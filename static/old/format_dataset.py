from pathlib import Path
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("formatted_output")
TARGET_SIZE = (128, 128)
SKIP_NAMES = {"formatted_output", "__pycache__", ".git"}

def process_image(infile: Path, outfile: Path):
    with Image.open(infile) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img = ImageOps.fit(img, TARGET_SIZE, method=Image.Resampling.LANCZOS)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        img.save(outfile, format="PNG")

def main():
    folder_counts = {}

    for class_folder in sorted(INPUT_DIR.iterdir()):
        if not class_folder.is_dir():
            continue
        if class_folder.name in SKIP_NAMES:
            continue

        files = [f for f in sorted(class_folder.iterdir()) if f.is_file()]
        if not files:
            continue

        saved_count = 0
        print(f"\nProcessing folder: {class_folder.name}")

        for infile in files:
            try:
                outfile = OUTPUT_DIR / class_folder.name / f"img{saved_count}.png"
                process_image(infile, outfile)
                print(f"Saved: {outfile} <- from {infile.name}")
                saved_count += 1
            except Exception as e:
                print(f"Failed on {infile}: {e}")

        folder_counts[class_folder.name] = saved_count

    print("\nFinal saved counts:")
    for folder, count in folder_counts.items():
        print(f"{folder}: {count}")

if __name__ == "__main__":
    main()