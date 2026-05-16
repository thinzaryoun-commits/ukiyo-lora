import csv
import shutil
import random
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & Thresholds
# ---------------------------------------------------------------------------
# Pointing to MET Dataset
RAW_DIR    = Path("/kaggle/input/datasets/kengoichiki/the-metropolitan-museum-of-art-ukiyoe-dataset/images")
BASE_DIR   = Path("/kaggle/working/ukiyo-lora")
TRAIN_DIR  = BASE_DIR / "dataset" / "train"
REPORT_CSV = BASE_DIR / "dataset" / "selection_report.csv"

# Thresholds
MIN_RESOLUTION  = 400
MAX_RATIO        = 2.0
BRIGHTNESS_LOW  = 20
BRIGHTNESS_HIGH = 240
TARGET_MIN      = 150 # Diversity Target
SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png"}

Image.MAX_IMAGE_PIXELS = None # Allow large MET scans

def mean_brightness(img: Image.Image) -> float:
    gray = img.convert("L")
    pixels = list(gray.getdata())
    return sum(pixels) / len(pixels)

def run_selection(brightness_low: int):
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Get ALL images across all artist subfolders
    image_files = list(RAW_DIR.rglob('*'))
    image_files = [p for p in image_files if p.suffix.lower() in SUPPORTED_EXTS]
    
    # 2. Shuffle for Diversity
    random.seed(42)
    random.shuffle(image_files)

    accepted_hashes = set()
    rows = []

    print(f"Scanning {len(image_files)} images for a diverse selection...")

    for img_path in tqdm(image_files, desc="Filtering"):
        # We need to preserve the relative path so copy_accepted can find it
        # The filename in the dictionary should be the FULL path relative to RAW_DIR
        rel_path = img_path.relative_to(RAW_DIR)
        
        row = {
            "filename": str(rel_path),
            "accepted": False,
            "reject_reason": "",
        }

        try:
            # Metadata check (Fast)
            with Image.open(img_path) as img_brief:
                w, h = img_brief.size
                if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
                    row["reject_reason"] = "too_small"
                    rows.append(row); continue
                
            # Content check (Slower)
            img = Image.open(img_path).convert("RGB")
            brightness = mean_brightness(img)
            
            if brightness < brightness_low:
                row["reject_reason"] = "too_dark"
            elif brightness > BRIGHTNESS_HIGH:
                row["reject_reason"] = "too_bright"
            else:
                phash = str(imagehash.phash(img))
                if phash in accepted_hashes:
                    row["reject_reason"] = "duplicate"
                else:
                    row["accepted"] = True
                    accepted_hashes.add(phash)
            
            rows.append(row)

            # Diversity Cap
            if len(accepted_hashes) >= 300:
                print("\nTarget of 300 diverse images reached.")
                break

        except Exception as e:
            row["reject_reason"] = f"error:{e}"
            rows.append(row)

    return rows # Returning just the list of dictionaries

def copy_accepted(rows: list[dict]) -> None:
    for row in rows:
        if row.get("accepted"):
            src = RAW_DIR / row["filename"]
            # We flatten the output directory for easier training
            dst = TRAIN_DIR / Path(row["filename"]).name
            shutil.copy2(src, dst)

def main() -> None:
    # Pass 1
    rows = run_selection(brightness_low=BRIGHTNESS_LOW)
    
    accepted_count = sum(1 for r in rows if r["accepted"])
    print(f"\nFinal training set: {accepted_count} images.")
    
    # Save & Copy
    copy_accepted(rows)
    
    # Create CSV Report
    fieldnames = ["filename", "accepted", "reject_reason"]
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    main()
