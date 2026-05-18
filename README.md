# SD 1.5 LoRA Fine-tuning — Ukiyo-e Art Style

**Course:** CPE494/663 Generative AI | KMUTT  
**Team:** Youn Thinzar (Implementation) · Kyawhtin Khaung Soe (Theory & Data QA)  

---

## Description

Fine-tunes Stable Diffusion 1.5 via LoRA on 300 curated Ukiyo-e (Japanese woodblock print) images from the Metropolitan Museum of Art collection.
Three rank configurations (4 / 16 / 32) are compared using FID and CLIP scores.
Trigger word: `ukiyoe style`.

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/select_images.py` | Filter raw images by resolution, brightness, and perceptual hash deduplication into `dataset/train/` |
| `scripts/caption_images.py` | Generate `metadata.jsonl` captions using BLIP with trigger word prefix |
| `scripts/split_dataset.py` | 80/20 train-val split — 240 train / 60 val images |
| `scripts/train_lora.py` | LoRA fine-tuning loop (rank=4, rank=16, rank=32 at LR=1e-4, 2000 steps) |
| `scripts/generate_eval_images.py` | Run inference with trained checkpoints, save 20 images per run to `eval/generated/` |
| `scripts/compute_fid.py` | Compute FID score between generated and val images using clean-fid |
| `scripts/compute_clip.py` | Compute CLIP score between generated images and prompts using open-clip |
| `scripts/compare_runs.py` | Plot loss curves, FID/CLIP bar charts, and sample comparison grid |

---

## How to Run

### 0. Setup (Kaggle Notebook)
```bash
pip install -r requirements.txt
export HF_HOME=/kaggle/working/hf_cache
```

### 1. Select & QA images
```bash
python scripts/select_images.py
```

### 2. Caption images
```bash
python scripts/caption_images.py
```

### 3. Train-val split (80/20)
```bash
python scripts/split_dataset.py
```

### 4. Train (change RANK env var per run)
```bash
RANK=4  python scripts/train_lora.py
RANK=16 python scripts/train_lora.py
RANK=32 python scripts/train_lora.py
```

### 5. Generate evaluation images
```bash
python scripts/generate_eval_images.py
```

### 6. Compute FID and CLIP scores
```bash
python scripts/compute_fid.py
python scripts/compute_clip.py
```

### 7. Compare runs and generate figures
```bash
python scripts/compare_runs.py
```

---

## Project Layout

```
ukiyo-lora/
├── dataset/
│   ├── raw/          # original MET collection images
│   ├── train/        # 240 selected + captioned images + metadata.jsonl
│   └── val/          # 60 val images + metadata.jsonl
├── output/           # LoRA checkpoints per run  [git-ignored]
├── eval/             # FID + CLIP score CSVs and figures
├── scripts/          # all Python scripts
├── report/           # final report assets (figures, tables)
├── requirements.txt
└── README.md
```

---

## Results

| Run | Rank | LR | Final Loss | FID ↓ | Avg CLIP ↑ |
|-----|------|----|-----------|-------|-----------|
| run_4 ★ | 4 | 1e-4 | 0.5894 | 233.05 | 0.3402 |
| run_16 | 16 | 1e-4 | — | — | — |
| run_32 | 32 | 1e-4 | 0.5149 | 242.68 | 0.3396 |

★ Best quantitative results. Rank=16 excluded due to training instability.

---

## Base Model

`runwayml/stable-diffusion-v1-5` — LoRA applied to cross-attention layers:
`to_q`, `to_k`, `to_v`, `to_out.0`  
Platform: Kaggle T4 GPU (16GB VRAM) · Checkpoint every 500 steps
