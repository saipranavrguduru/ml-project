import argparse
import csv
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import LABEL_ORDER, MultiLabelImageFolder, build_eval_transform, build_train_transform
from src.model import create_resnet18_multilabel


def split_indices(n_items, val_fraction, seed):
    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(n_items * val_fraction)))
    return indices[val_size:], indices[:val_size]


def multilabel_metrics(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    exact_match = (preds == labels).all(dim=1).float().mean().item()
    hamming_acc = (preds == labels).float().mean().item()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()
    f1_micro = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

    return {"exact_match": exact_match, "hamming_acc": hamming_acc, "f1_micro": f1_micro}


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
    parser.add_argument("--run_dir", type=str, default=None, help="Directory for this run's checkpoint and metrics")
    parser.add_argument("--output", type=str, default=None, help="Checkpoint path; defaults inside the run directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
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
    val_base = MultiLabelImageFolder(args.data_dir, transform=build_eval_transform(args.image_size))
    train_idx, val_idx = split_indices(len(train_base), args.val_fraction, args.seed)

    train_loader = DataLoader(
        Subset(train_base, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        Subset(val_base, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = create_resnet18_multilabel(num_labels=len(LABEL_ORDER), pretrained=args.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, args.threshold)
        val_metrics = run_epoch(model, val_loader, criterion, device, None, args.threshold)

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1_micro']:.4f} "
            f"val_hamming={val_metrics['hamming_acc']:.4f}"
        )

        saved_checkpoint = False
        if val_metrics["f1_micro"] > best_f1:
            best_f1 = val_metrics["f1_micro"]
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
                "train_f1_micro": train_metrics["f1_micro"],
                "val_loss": val_metrics["loss"],
                "val_exact_match": val_metrics["exact_match"],
                "val_hamming_acc": val_metrics["hamming_acc"],
                "val_f1_micro": val_metrics["f1_micro"],
                "saved_checkpoint": saved_checkpoint,
                "checkpoint_path": args.output,
                "pretrained": args.pretrained,
                "seed": args.seed,
                "threshold": args.threshold,
                "lr": args.lr,
                "batch_size": args.batch_size,
            },
        )


if __name__ == "__main__":
    main()
