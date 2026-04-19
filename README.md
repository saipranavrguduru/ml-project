# ML Project
## Folder Structure
```
.
├── README.md
├── requirements.txt
├── src                 # pipeline files
│   └── eval.py
└── static              # data folder should be under here
```

## Python Environment Setup

This project uses a local virtual environment at `.venv`.

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3) Install project dependencies

```bash
pip install -r requirements.txt
```

The dependency set includes core deep learning and foundational ML/data-science packages such as:

- torch
- torchvision
- pillow
- numpy
- pandas
- scikit-learn
- matplotlib

## Run Evaluation

From the project root:

```bash
python src/eval.py --model_path YOUR_SAVED_MODEL --test_data project_test_data --group_id YOUR_GROUP_ID --project_title "YOUR_PROJECT_TITLE"
```

If you do not activate the environment, use:

```bash
.venv/bin/python src/eval.py --model_path YOUR_SAVED_MODEL --test_data project_test_data --group_id YOUR_GROUP_ID --project_title "YOUR_PROJECT_TITLE"
```
