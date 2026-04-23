import argparse
import csv
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import LABEL_ORDER, MultiLabelImageFolder, build_eval_transform, build_train_transform
from src.model import create_resnet18_multilabel


def sample_key(dataset, idx):
    img_path, _ = dataset.samples[idx]
    return img_path.relative_to(dataset.root).as_posix()


def create_split(dataset, train_fraction, val_fraction, test_fraction, seed):
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")

    keys = [sample_key(dataset, idx) for idx in range(len(dataset))]
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_items = len(keys)
    train_size = int(round(n_items * train_fraction))
    val_size = int(round(n_items * val_fraction))

    train_keys = keys[:train_size]
    val_keys = keys[train_size:train_size + val_size]
    test_keys = keys[train_size + val_size:]

    if not train_keys or not val_keys or not test_keys:
        raise ValueError("Split produced an empty partition; check dataset size/fractions.")

    return {"train": train_keys, "val": val_keys, "test": test_keys}


def load_or_create_split(path, dataset, args):
    path = Path(path)
    if path.exists():
        with path.open("r") as split_file:
            return json.load(split_file)

    split = create_split(dataset, args.train_fraction, args.val_fraction, args.test_fraction, args.seed)
    payload = {
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "label_order": LABEL_ORDER,
        "splits": split,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as split_file:
        json.dump(payload, split_file, indent=2)
    return payload


def indices_from_split(dataset, split_payload, split_name):
    wanted = set(split_payload["splits"][split_name])
    key_to_idx = {sample_key(dataset, idx): idx for idx in range(len(dataset))}
    missing = sorted(wanted - set(key_to_idx))
    if missing:
        raise ValueError(f"{len(missing)} {split_name} split files are missing from dataset; first missing: {missing[0]}")
    return [key_to_idx[key] for key in split_payload["splits"][split_name]]


def multilabel_metrics(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    exact_match = (preds == labels).all(dim=1).float().mean().item()
    hamming_acc = (preds == labels).float().mean().item()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    precision_micro = (tp / (tp + fp + 1e-8)).item()
    recall_micro = (tp / (tp + fn + 1e-8)).item()
    f1_micro = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

    intersection = (preds * labels).sum(dim=1)
    union = ((preds + labels) > 0).float().sum(dim=1)
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    mean_iou = iou.mean().item()

    per_class_tp = ((preds == 1) & (labels == 1)).sum(dim=0).float()
    per_class_fp = ((preds == 1) & (labels == 0)).sum(dim=0).float()
    per_class_fn = ((preds == 0) & (labels == 1)).sum(dim=0).float()
    per_class_f1 = 2 * per_class_tp / (2 * per_class_tp + per_class_fp + per_class_fn + 1e-8)
    f1_macro = per_class_f1.mean().item()

    return {
        "exact_match": exact_match,
        "hamming_acc": hamming_acc,
        "mean_iou": mean_iou,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def run_epoch(model, loader, criterion, device, optimizer=None, threshold=0.5):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = multilabel_metrics(logits, labels, threshold=threshold)
    metrics["loss"] = total_loss / total_samples
    return metrics


def save_checkpoint(path, model, args, val_metrics, epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_order": LABEL_ORDER,
            "image_size": args.image_size,
            "threshold": args.threshold,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "architecture": "resnet18",
            "pretrained": args.pretrained,
            "split_json": args.split_json,
        },
        path,
    )


def write_metrics_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_run_paths(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"resnet18_{timestamp}"
    if args.pretrained:
        run_name += "_pretrained"

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs") / run_name
    output = Path(args.output) if args.output else run_dir / "best_resnet18_multilabel.pth"
    log_csv = Path(args.log_csv) if args.log_csv else run_dir / "metrics.csv"
    return run_dir, output, log_csv


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 for multilabel image classification")
    parser.add_argument("--data_dir", type=str, default="static/data")
    parser.add_argument("--split_json", type=str, default="splits/split_seed42.json")
    parser.add_argument("--run_dir", type=str, default=None, help="Directory for this run's checkpoint and metrics")
    parser.add_argument("--output", type=str, default=None, help="Checkpoint path; defaults inside the run directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Use cached torchvision ImageNet weights if available")
    parser.add_argument("--log_csv", type=str, default=None, help="CSV path for per-epoch metrics; defaults inside the run directory")
    args = parser.parse_args()

    run_dir, output_path, log_csv_path = build_run_paths(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output = str(output_path)
    args.log_csv = str(log_csv_path)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {args.output}")
    print(f"metrics: {args.log_csv}")

    train_base = MultiLabelImageFolder(args.data_dir, transform=build_train_transform(args.image_size))
    eval_base = MultiLabelImageFolder(args.data_dir, transform=build_eval_transform(args.image_size))
    split_payload = load_or_create_split(args.split_json, eval_base, args)
    train_idx = indices_from_split(train_base, split_payload, "train")
    val_idx = indices_from_split(eval_base, split_payload, "val")
    test_idx = indices_from_split(eval_base, split_payload, "test")

    print(f"split_json: {args.split_json}")
    print(f"samples: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_loader = DataLoader(
        Subset(train_base, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(eval_base, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        Subset(eval_base, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = create_resnet18_multilabel(num_labels=len(LABEL_ORDER), pretrained=args.pretrained).to(device)
    param_count = count_trainable_params(model)
    print(f"trainable_params: {param_count}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_epoch = None
    best_val_metrics = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, args.threshold)
        val_metrics = run_epoch(model, val_loader, criterion, device, None, args.threshold)

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1_micro']:.4f} "
            f"val_macro_f1={val_metrics['f1_macro']:.4f} "
            f"val_hamming={val_metrics['hamming_acc']:.4f}"
        )

        saved_checkpoint = False
        if val_metrics["f1_micro"] > best_f1:
            best_f1 = val_metrics["f1_micro"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            save_checkpoint(args.output, model, args, val_metrics, epoch)
            saved_checkpoint = True
            print(f"saved checkpoint: {args.output}")

        write_metrics_row(
            args.log_csv,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_exact_match": train_metrics["exact_match"],
                "train_hamming_acc": train_metrics["hamming_acc"],
                "train_mean_iou": train_metrics["mean_iou"],
                "train_precision_micro": train_metrics["precision_micro"],
                "train_recall_micro": train_metrics["recall_micro"],
                "train_f1_micro": train_metrics["f1_micro"],
                "train_f1_macro": train_metrics["f1_macro"],
                "val_loss": val_metrics["loss"],
                "val_exact_match": val_metrics["exact_match"],
                "val_hamming_acc": val_metrics["hamming_acc"],
                "val_mean_iou": val_metrics["mean_iou"],
                "val_precision_micro": val_metrics["precision_micro"],
                "val_recall_micro": val_metrics["recall_micro"],
                "val_f1_micro": val_metrics["f1_micro"],
                "val_f1_macro": val_metrics["f1_macro"],
                "saved_checkpoint": saved_checkpoint,
                "checkpoint_path": args.output,
                "pretrained": args.pretrained,
                "seed": args.seed,
                "threshold": args.threshold,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "param_count": param_count,
            },
        )

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, criterion, device, None, args.threshold)
    print(
        f"best_epoch={best_epoch:03d} "
        f"best_val_f1={best_f1:.4f} "
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_f1={test_metrics['f1_micro']:.4f} "
        f"test_macro_f1={test_metrics['f1_macro']:.4f} "
        f"test_hamming={test_metrics['hamming_acc']:.4f}"
    )
    write_metrics_row(
        Path(run_dir) / "summary.csv",
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_metrics["loss"],
            "best_val_exact_match": best_val_metrics["exact_match"],
            "best_val_hamming_acc": best_val_metrics["hamming_acc"],
            "best_val_mean_iou": best_val_metrics["mean_iou"],
            "best_val_precision_micro": best_val_metrics["precision_micro"],
            "best_val_recall_micro": best_val_metrics["recall_micro"],
            "best_val_f1_micro": best_val_metrics["f1_micro"],
            "best_val_f1_macro": best_val_metrics["f1_macro"],
            "test_loss": test_metrics["loss"],
            "test_exact_match": test_metrics["exact_match"],
            "test_hamming_acc": test_metrics["hamming_acc"],
            "test_mean_iou": test_metrics["mean_iou"],
            "test_precision_micro": test_metrics["precision_micro"],
            "test_recall_micro": test_metrics["recall_micro"],
            "test_f1_micro": test_metrics["f1_micro"],
            "test_f1_macro": test_metrics["f1_macro"],
            "checkpoint_path": args.output,
            "split_json": args.split_json,
            "pretrained": args.pretrained,
            "seed": args.seed,
            "threshold": args.threshold,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "param_count": param_count,
        },
    )


if __name__ == "__main__":
    main()
