"""
Split dataset/train/ into 80% train / 20% val.
Moves val images to dataset/val/ and updates both metadata.jsonl files.
"""

import json
import random
import shutil
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/kaggle/working/ukiyo-lora")
TRAIN_DIR  = BASE_DIR / "dataset" / "train"
VAL_DIR    = BASE_DIR / "dataset" / "val"
TRAIN_META = TRAIN_DIR / "metadata.jsonl"
VAL_META   = VAL_DIR   / "metadata.jsonl"
SPLIT_LOG  = BASE_DIR  / "split_log.txt"

SEED       = 42
VAL_RATIO  = 0.20

# ── Load metadata ─────────────────────────────────────────────────────────────
records = []
with open(TRAIN_META, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

total = len(records)
print(f"Total records found: {total}")

# ── Split ─────────────────────────────────────────────────────────────────────
random.seed(SEED)
shuffled = records.copy()
random.shuffle(shuffled)

n_val   = max(1, round(total * VAL_RATIO))
n_train = total - n_val

val_records   = shuffled[:n_val]
train_records = shuffled[n_val:]

# ── Move val images ───────────────────────────────────────────────────────────
VAL_DIR.mkdir(parents=True, exist_ok=True)

moved = []
missing = []
for rec in val_records:
    fname = rec["file_name"]
    src   = TRAIN_DIR / fname
    dst   = VAL_DIR   / fname
    if src.exists():
        shutil.move(str(src), str(dst))
        moved.append(fname)
    else:
        missing.append(fname)
        print(f"  WARNING: image not found, skipping move: {fname}")

# ── Write val metadata.jsonl ──────────────────────────────────────────────────
with open(VAL_META, "w", encoding="utf-8") as f:
    for rec in val_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ── Update train metadata.jsonl ───────────────────────────────────────────────
with open(TRAIN_META, "w", encoding="utf-8") as f:
    for rec in train_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ── Save split_log.txt ────────────────────────────────────────────────────────
train_fnames = [r["file_name"] for r in train_records]
val_fnames   = [r["file_name"] for r in val_records]

with open(SPLIT_LOG, "w", encoding="utf-8") as log:
    log.write(f"Split log — seed={SEED}, val_ratio={VAL_RATIO}\n")
    log.write(f"Total: {total}  |  Train: {n_train}  |  Val: {n_val}\n")
    log.write("\n── TRAIN FILES ──────────────────────────────────────────\n")
    for name in sorted(train_fnames):
        log.write(f"  {name}\n")
    log.write("\n── VAL FILES ────────────────────────────────────────────\n")
    for name in sorted(val_fnames):
        log.write(f"  {name}\n")
    if missing:
        log.write("\n── MISSING (not moved) ──────────────────────────────────\n")
        for name in missing:
            log.write(f"  {name}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Train : {n_train} images")
print(f"  Val   : {n_val}   images")
print(f"{'='*55}")
print(f"\nFiles moved to dataset/val/  ({len(moved)} total):")
for name in sorted(moved):
    print(f"  {name}")
if missing:
    print(f"\nWARNING — {len(missing)} file(s) listed in metadata but not found on disk:")
    for name in missing:
        print(f"  {name}")
print(f"\nSplit log saved → {SPLIT_LOG}")
