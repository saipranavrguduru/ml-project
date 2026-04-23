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

The fixed label order used by the code and checkpoints is:

```text
pen, paper, book, clock, phone, laptop, chair, desk, bottle, keychain, backpack, calculator
```

## Method

- Model: selectable torchvision backbone, currently `resnet18` or `densenet121`
- Output head: final classifier layer replaced with 12 logits
- Transfer learning: when `--pretrained` is used, the selected backbone starts
  from ImageNet weights and all layers are fine-tuned on this dataset
- From-scratch fallback: without `--pretrained`, the same architecture trains
  from random initialization
- Loss: `BCEWithLogitsLoss`
- Inference: `sigmoid(logits)` followed by thresholding, default `0.5`
- Image preprocessing: RGB conversion, resize to `128x128`, tensor conversion,
  ImageNet mean/std normalization
- Training augmentation: random horizontal flip and light color jitter
- Split: fixed random 70/15/15 train/val/test split using seed `42`
- Checkpoint selection: best validation micro-F1
- Optimizer: AdamW over all model parameters
- Logged metrics: loss, exact match, hamming accuracy, mean IoU, micro
  precision/recall/F1, macro-F1, and trainable parameter count

## Files

```text
eval.py                  # evaluation script
requirements.txt
src/
  dataset.py             # dataset loader and transforms
  model.py               # torchvision backbone factory
  train.py               # training and checkpoint saving
checkpoints/
  best_<arch>_multilabel.pth
splits/
  split_seed42.json
runs/
  <arch>_YYYYMMDD_HHMMSS/
    metrics.csv
    summary.csv
    best_<arch>_multilabel.pth
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

Select a backbone explicitly:

```powershell
python -m src.train --data_dir static/data --epochs 10 --arch densenet121
```

Transfer-learning run, if the selected torchvision pretrained weights are
already cached locally:

```powershell
python -m src.train --data_dir static/data --epochs 10 --pretrained
```

Example DenseNet transfer-learning run:

```powershell
python -m src.train --data_dir static/data --epochs 10 --arch densenet121 --pretrained
```

Example head-only DenseNet transfer-learning run:

```powershell
python -m src.train --data_dir static/data --epochs 15 --arch densenet121 --pretrained --freeze_backbone
```

Each training command creates a timestamped run directory by default:

```text
runs/<arch>_YYYYMMDD_HHMMSS/
  metrics.csv
  summary.csv
  best_<arch>_multilabel.pth
```

The first run also creates `splits/split_seed42.json`, which fixes the shared
70/15/15 train/val/test split for later experiments. Training uses the train
partition, checkpoint selection uses validation micro-F1, and the best
checkpoint is evaluated once on the local test partition for comparison across
models.

Important note: the current split is created at the individual image level, not
at the folder level. That means images from the same folder such as
`pen_paper/` can be divided across train, val, and test. This is convenient for
repeatable experiments, but it can make evaluation look a bit better than a
harder split where entire folders or capture sessions are held out.

To choose the run directory explicitly:

```powershell
python -m src.train --data_dir static/data --epochs 10 --run_dir runs/from_scratch_10ep
```


## Evaluate

Course grader command:

```powershell
python eval.py --model_path checkpoints/best_resnet18_multilabel.pth --test_data project_test_data --group_id YOUR_GROUP_ID --project_title "YOUR_PROJECT_TITLE"
```

The `eval.py` example is configured for the ResNet-18 checkpoint path
shown above. When evaluating a different backbone such as `densenet121`, the
model construction in `eval.py` should match the checkpoint architecture.

Local evaluation example:

```powershell
python eval.py --model_path checkpoints/best_resnet18_multilabel.pth --test_data static/data --group_id 47 --project_title "YOUR_PROJECT_TITLE"
```

The local test metrics in each run's `summary.csv` are better for comparing
experiments. The `eval.py` script is kept primarily for the grader command.

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
    "architecture": args.arch,
    "pretrained": args.pretrained,
    "freeze_backbone": args.freeze_backbone,
    "split_json": args.split_json,
}
```

`.pt` and `.pth` are both common PyTorch checkpoint extensions. This project uses
`.pth`.

## Report

The provided LaTeX template files are in `report/`. The required report
submission is the compiled PDF.

## Exploration
- Freezing Bottom Layers - Will
- ResNet Fine Tuning - Pranav
- maybe dendritic ResNet/DenseNet fine tuning - Addison
- EfficientNet fine tuning - Ametoje
- Vision Transformer - (?)
## References

- PyTorch transfer learning tutorial:
  https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- DenseNet paper: Huang et al., "Densely Connected Convolutional Networks"
  (CVPR 2017)
  https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html
- Torchvision DenseNet docs:
  https://docs.pytorch.org/vision/main/models/densenet.html
- Torchvision ResNet-18 weights and preprocessing:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Torchvision ResNet docs:
  https://docs.pytorch.org/vision/main/models/resnet.html
- PyTorch saving/loading models:
  https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
