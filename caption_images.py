# ---------------------------------------------------------------------------
# Paths (relative to repo root, matching Kaggle layout)
# ---------------------------------------------------------------------------

TRAIN_DIR = Path("/kaggle/working/ukiyo-lora/dataset/train")
METADATA_OUT = TRAIN_DIR / "metadata.jsonl"
CSV_OUT = TRAIN_DIR / "captions_preview.csv"

TRIGGER = "ukiyoe style"
FALLBACK_CAPTION = f"{TRIGGER}, japanese woodblock print artwork"
MODEL_ID = "Salesforce/blip-image-captioning-base"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Phrases to strip from BLIP output before prepending the trigger word
REDUNDANT_PHRASES = [
    "a painting of",
    "an painting of",
    "a picture of",
    "an picture of",
    "a photo of",
    "an photo of",
    "a image of",
    "an image of",
    "a drawing of",
    "an drawing of",
    "a close up of",
    "an close up of",
    "a close-up of",
    "there is",
    "this is",
    "it is",
]


# ---------------------------------------------------------------------------
# Caption cleaning
# ---------------------------------------------------------------------------
def clean_caption(raw: str) -> str:
    text = raw.lower().strip()
    for phrase in REDUNDANT_PHRASES:
        # Remove phrase at start of string or after a comma/period
        text = re.sub(rf"(?:^|(?<=,\s)|(?<=\.\s)){re.escape(phrase)}\s*", "", text)
    # Collapse multiple spaces / strip
    text = re.sub(r"\s+", " ", text).strip().strip(",").strip()
    return text


def build_caption(raw: str) -> str:
    cleaned = clean_caption(raw)
    return f"{TRIGGER}, {cleaned}" if cleaned else FALLBACK_CAPTION


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading BLIP model ({MODEL_ID})...")
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    image_paths = sorted(
        p for p in TRAIN_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )

    if not image_paths:
        print(f"No images found in {TRAIN_DIR}. Exiting.")
        return

    print(f"Found {len(image_paths)} images. Captioning...")

    records = []

    for img_path in tqdm(image_paths, desc="Captioning"):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            if device == "cuda":
                inputs = {k: v.to(torch.float16) for k, v in inputs.items() if v.dtype == torch.float}

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50)

            raw_caption = processor.decode(out[0], skip_special_tokens=True)
            caption = build_caption(raw_caption)

        except Exception as exc:
            print(f"  [WARN] Failed on {img_path.name}: {exc!r} — using fallback.")
            caption = FALLBACK_CAPTION

        records.append({"file_name": img_path.name, "text": caption})

    # Write metadata.jsonl
    with METADATA_OUT.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(records)} entries → {METADATA_OUT}")

    # Write captions_preview.csv
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote preview CSV → {CSV_OUT}")

    # Print 5 random samples
    samples = random.sample(records, min(5, len(records)))
    print("\n--- 5 random sample captions ---")
    for s in samples:
        print(f"  {s['file_name']}: {s['text']}")


if __name__ == "__main__":
    main()
