# Multilabel Image Classification Project

PyTorch pipeline for offline multilabel classification of 128x128 PNG images
across 12 object classes.

## Task

Each image can contain more than one label. Folder names encode the labels using
underscores:

```text
pen_paper/img1.png
phone_laptop_keychain/img2.png
```

The fixed label order is:

```text
backpack, book, bottle, calculator, chair, clock, desk, keychain, laptop, paper, pen, phone
```

## Method

- Model: `torchvision.models.resnet18`
- Output head: final fully connected layer replaced with 12 logits
- Transfer learning: when `--pretrained` is used, ResNet-18 starts from
  ImageNet weights and all layers are fine-tuned on this dataset
- From-scratch fallback: without `--pretrained`, the same architecture trains
  from random initialization
- Loss: `BCEWithLogitsLoss`
- Inference: `sigmoid(logits)` followed by thresholding, default `0.5`
- Image preprocessing: RGB conversion, resize to `128x128`, tensor conversion,
  ImageNet mean/std normalization
- Training augmentation: random horizontal flip and light color jitter
- Validation: fixed random 80/20 train/validation split using seed `42`
- Checkpoint selection: best validation micro-F1
- Optimizer: AdamW over all model parameters

## Files

```text
eval.py                  # evaluation script
requirements.txt
src/
  dataset.py             # dataset loader and transforms
  model.py               # ResNet-18 model factory
  train.py               # training and checkpoint saving
checkpoints/
  best_resnet18_multilabel.pth
runs/
  resnet18_YYYYMMDD_HHMMSS/
    metrics.csv
    best_resnet18_multilabel.pth
report/
  report.tex
  refs.bib
  report.pdf
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Offline training from local/random initialization:

```powershell
python -m src.train --data_dir static/data --epochs 10
```

Transfer-learning run, if torchvision ResNet-18 pretrained weights are already
cached locally:

```powershell
python -m src.train --data_dir static/data --epochs 10 --pretrained
```

Each training command creates a timestamped run directory by default:

```text
runs/resnet18_YYYYMMDD_HHMMSS/
  metrics.csv
  best_resnet18_multilabel.pth
```


## Evaluate

Required grader command:

```powershell
python eval.py --model_path checkpoints/best_resnet18_multilabel.pth --test_data project_test_data --group_id YOUR_GROUP_ID --project_title "YOUR_PROJECT_TITLE"
```

Local sanity check:

```powershell
python eval.py --model_path checkpoints/best_resnet18_multilabel.pth --test_data static/data --group_id 47 --project_title "YOUR_PROJECT_TITLE"
```

The `static/data` result is not a true held-out test result because the same data
pool is used for training/validation.

## Checkpoint

The saved `.pth` file is a PyTorch checkpoint dictionary containing:

```python
{
    "model_state_dict": model.state_dict(),
    "label_order": LABEL_ORDER,
    "image_size": 128,
    "threshold": 0.5,
    "epoch": epoch,
    "val_metrics": val_metrics,
    "architecture": "resnet18",
    "pretrained": args.pretrained,
}
```

`.pt` and `.pth` are both common PyTorch checkpoint extensions. This project uses
`.pth`.

## Report

The provided LaTeX template files are in `report/`. The required report
submission is the compiled PDF.

## Exploration
- Freezing Bottom Layers - Will
- Full Fine Tuning - Pranav
- Training From Scratch - Addison
- Report - Ametoje
- Vision Transformer - (?)
## References

- PyTorch transfer learning tutorial:
  https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- Torchvision ResNet-18 weights and preprocessing:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- PyTorch saving/loading models:
  https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
