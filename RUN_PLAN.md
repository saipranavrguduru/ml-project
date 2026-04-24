# Run Plan
Notes:

- if `vitb32` runs out of memory, rerun that same command with `--batch_size 16`
- if the best checkpoint is still from the last epoch, bump that run by another `10-20` epochs
- efficient net will also need to be tested once its implemented

## Full Fine-Tuning

Use:

- `--epochs 50`
- `--early_stopping_patience 5`
- LR sweep: `1e-4`, `3e-4`

```powershell
python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch resnet18 --pretrained --epochs 50 --lr 1e-4 --batch_size 32 --early_stopping_patience 5 --run_dir runs/resnet18_fullft_lr1e4
python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch resnet18 --pretrained --epochs 50 --lr 3e-4 --batch_size 32 --early_stopping_patience 5 --run_dir runs/resnet18_fullft_lr3e4

python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch densenet121 --pretrained --epochs 50 --lr 1e-4 --batch_size 32 --early_stopping_patience 5 --run_dir runs/densenet121_fullft_lr1e4
python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch densenet121 --pretrained --epochs 50 --lr 3e-4 --batch_size 32 --early_stopping_patience 5 --run_dir runs/densenet121_fullft_lr3e4

python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch vitb32 --pretrained --epochs 50 --lr 1e-4 --batch_size 32 --early_stopping_patience 5 --run_dir runs/vitb32_fullft_lr1e4
python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch vitb32 --pretrained --epochs 50 --lr 3e-4 --batch_size 32 --early_stopping_patience 5 --run_dir runs/vitb32_fullft_lr3e4
```

## Head-Only

Use:

- `--epochs 30`
- `--early_stopping_patience 5`
- `--freeze_backbone`
- LR sweep: `3e-4`, `1e-3`

```powershell
python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch resnet18 --pretrained --freeze_backbone --epochs 30 --lr 3e-4 --batch_size 32 --early_stopping_patience 5 --run_dir runs/resnet18_headonly_lr3e4
python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch resnet18 --pretrained --freeze_backbone --epochs 30 --lr 1e-3 --batch_size 32 --early_stopping_patience 5 --run_dir runs/resnet18_headonly_lr1e3

python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch densenet121 --pretrained --freeze_backbone --epochs 30 --lr 3e-4 --batch_size 32 --early_stopping_patience 5 --run_dir runs/densenet121_headonly_lr3e4
python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch densenet121 --pretrained --freeze_backbone --epochs 30 --lr 1e-3 --batch_size 32 --early_stopping_patience 5 --run_dir runs/densenet121_headonly_lr1e3

python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch vitb32 --pretrained --freeze_backbone --epochs 30 --lr 3e-4 --batch_size 32 --early_stopping_patience 5 --run_dir runs/vitb32_headonly_lr3e4
python -m src.train --data_dir static/data --split_json splits/split_seed42.json --arch vitb32 --pretrained --freeze_backbone --epochs 30 --lr 1e-3 --batch_size 32 --early_stopping_patience 5 --run_dir runs/vitb32_headonly_lr1e3
```

## Compare Afterward

Use each run's `summary.csv`to compare between models since thats based off our local-test split.
