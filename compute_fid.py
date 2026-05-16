import os
import csv
import shutil
import tempfile
from pathlib import Path
from PIL import Image
from cleanfid import fid

VAL_DIR = Path("/kaggle/working/ukiyo-lora/dataset/val")
RUNS = [
    {"run": "run_4",  "rank": 4,  "gen_dir": Path("/kaggle/working/ukiyo-lora/eval/generated/run_4")},
    {"run": "run_16", "rank": 16, "gen_dir": Path("/kaggle/working/ukiyo-lora/eval/generated/run_16")},
    {"run": "run_32", "rank": 32, "gen_dir": Path("/kaggle/working/ukiyo-lora/eval/generated/run_32")},
]
OUTPUT_CSV = Path("/kaggle/working/ukiyo-lora/eval/fid_scores.csv")
TARGET_SIZE = (512, 512)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resize_images_to_tmp(src_dir: Path, size: tuple[int, int]) -> str:
    tmp = tempfile.mkdtemp()
    for img_path in src_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        img = Image.open(img_path).convert("RGB").resize(size, Image.LANCZOS)
        img.save(Path(tmp) / (img_path.stem + ".png"))
    return tmp


def count_images(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in IMG_EXTS)


def main():
    if not VAL_DIR.exists():
        print(f"ERROR: val directory not found: {VAL_DIR}")
        return

    val_count = count_images(VAL_DIR)
    print(f"Val images found: {val_count} in {VAL_DIR}")

    val_tmp = resize_images_to_tmp(VAL_DIR, TARGET_SIZE)
    print(f"Resized val images to {TARGET_SIZE} in temp dir\n")

    results = []

    try:
        for entry in RUNS:
            run_name = entry["run"]
            rank = entry["rank"]
            gen_dir = entry["gen_dir"]

            if not gen_dir.exists():
                print(f"[SKIP] {run_name}: folder not found ({gen_dir})")
                results.append({"run": run_name, "rank": rank, "fid_score": "N/A"})
                continue

            gen_count = count_images(gen_dir)
            if gen_count == 0:
                print(f"[SKIP] {run_name}: no images in {gen_dir}")
                results.append({"run": run_name, "rank": rank, "fid_score": "N/A"})
                continue

            print(f"Computing FID for {run_name} (rank={rank}, {gen_count} generated images)...")
            score = fid.compute_fid(val_tmp, str(gen_dir), mode="clean", num_workers=0)
            print(f"  FID = {score:.4f}")
            results.append({"run": run_name, "rank": rank, "fid_score": round(score, 4)})

    finally:
        shutil.rmtree(val_tmp, ignore_errors=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "rank", "fid_score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {OUTPUT_CSV}")
    print("\n--- FID Results (lower is better) ---")
    print(f"{'Run':<10} {'Rank':<8} {'FID Score'}")
    print("-" * 32)
    for r in results:
        print(f"{r['run']:<10} {r['rank']:<8} {r['fid_score']}")


if __name__ == "__main__":
    main()
