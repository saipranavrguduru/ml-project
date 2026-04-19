from pathlib import Path
from PIL import Image

DATASET_DIR = Path("formatted_output")

for folder in sorted(DATASET_DIR.iterdir()):
    if not folder.is_dir():
        continue
    for img_path in sorted(folder.glob("*.png")):
        with Image.open(img_path) as img:
            print(img_path, img.size, img.mode)