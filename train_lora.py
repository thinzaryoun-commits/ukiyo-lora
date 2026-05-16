"""
SD 1.5 LoRA fine-tuning — Ukiyo-e art
fp32 params, no GradScaler, no autocast
"""

import gc
import os
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------------
# Hyperparameters (overridable via env)
# ---------------------------------------------------------------------------
RANK             = int(os.environ.get("RANK", 4))
LR               = float(os.environ.get("LR", 1e-4))
BATCH_SIZE       = int(os.environ.get("BATCH_SIZE", 1))
GRAD_ACCUM       = int(os.environ.get("GRAD_ACCUM", 4))
MAX_STEPS        = int(os.environ.get("MAX_STEPS", 2000))
WARMUP_STEPS     = int(os.environ.get("WARMUP_STEPS", 100))
CKPT_EVERY       = int(os.environ.get("CKPT_EVERY", 500))
LOG_EVERY        = int(os.environ.get("LOG_EVERY", 50))
MAX_NORM         = float(os.environ.get("MAX_NORM", 1.0))
RESUME_FROM_STEP = int(os.environ.get("RESUME_FROM_STEP", 0))
IMAGE_SIZE       = 512
TRIGGER_WORD     = "ukiyoe style"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_MODEL    = "runwayml/stable-diffusion-v1-5"
WORKING_DIR   = Path("/kaggle/working/ukiyo-lora")
TRAIN_DIR     = WORKING_DIR / "dataset" / "train"
VAL_DIR       = WORKING_DIR / "dataset" / "val"
OUTPUT_DIR    = WORKING_DIR / "output"
GENERATED_DIR = WORKING_DIR / "generated"
SAMPLES_DIR   = OUTPUT_DIR / f"run_{RANK}_{LR:.0e}" / "samples"
LOSS_LOG      = OUTPUT_DIR / f"run_{RANK}_{LR:.0e}" / "loss_log.csv"
HF_CACHE      = "/kaggle/working/hf_cache"

os.environ.setdefault("HF_HOME", HF_CACHE)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class UkiyoeDataset(Dataset):
    def __init__(self, data_dir: Path, tokenizer: CLIPTokenizer, image_size: int = IMAGE_SIZE):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_size = image_size

        meta_path = data_dir / "metadata.jsonl"
        self.samples = []
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = self.data_dir / item["file_name"]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        caption = item["text"]
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.squeeze(0)

        return {"pixel_values": pixel_values, "input_ids": input_ids}


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------
def apply_lora(unet: UNet2DConditionModel, rank: int) -> UNet2DConditionModel:
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)

    # Cast LoRA params (and only them) to fp32
    for name, param in unet.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    unet.print_trainable_parameters()
    return unet


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------
def generate_samples(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    scheduler: DDPMScheduler,
    device: torch.device,
    step: int,
    prompts: list[str],
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
):
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    unet.eval()

    # Determine a safe dtype for inference
    inf_dtype = torch.float16 if device.type == "cuda" else torch.float32

    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            tokens = tokenizer(
                [prompt, ""],
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).to(device)
            text_embeds = text_encoder(tokens.input_ids)[0].to(inf_dtype)

            latents = torch.randn(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8, device=device, dtype=inf_dtype)
            scheduler.set_timesteps(num_inference_steps)
            latents = latents * scheduler.init_noise_sigma

            for t in scheduler.timesteps:
                latent_input = torch.cat([latents] * 2)
                latent_input = scheduler.scale_model_input(latent_input, t)

                noise_pred = unet(
                    latent_input.to(inf_dtype),
                    t,
                    encoder_hidden_states=text_embeds,
                ).sample

                uncond, cond = noise_pred.chunk(2)
                noise_pred = uncond + guidance_scale * (cond - uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents = latents / vae.config.scaling_factor
            image_tensor = vae.decode(latents.to(vae.dtype)).sample
            image_tensor = (image_tensor.clamp(-1, 1) + 1) / 2
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            image_np = (image_np * 255).astype("uint8")
            img = Image.fromarray(image_np)
            img.save(GENERATED_DIR / f"step{step:05d}_sample{i}.png")

    unet.train()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    # ── Clear GPU ─────────────────────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | RANK={RANK} | LR={LR} | RESUME_FROM_STEP={RESUME_FROM_STEP}")

    # ── Load models ───────────────────────────────────────────────────────────
    tokenizer = CLIPTokenizer.from_pretrained(
        BASE_MODEL, subfolder="tokenizer", cache_dir=HF_CACHE)

    # VAE in fp16 to save memory (not trained, just encoding)
    vae = AutoencoderKL.from_pretrained(
        BASE_MODEL, subfolder="vae",
        cache_dir=HF_CACHE, torch_dtype=torch.float16).to(device)
    vae.requires_grad_(False)

    # Text encoder in fp16 to save memory (frozen)
    text_encoder = CLIPTextModel.from_pretrained(
        BASE_MODEL, subfolder="text_encoder",
        cache_dir=HF_CACHE, torch_dtype=torch.float16).to(device)
    text_encoder.requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler", cache_dir=HF_CACHE)

    # UNet in fp32 — LoRA params will also be fp32, no conflict
    unet = UNet2DConditionModel.from_pretrained(
        BASE_MODEL, subfolder="unet", cache_dir=HF_CACHE, torch_dtype=torch.float32).to(device)
    unet = apply_lora(unet, rank=RANK)

    # Resume from checkpoint if requested
    if RESUME_FROM_STEP > 0:
        ckpt_path = OUTPUT_DIR / f"run_{RANK}_{LR:.0e}" / f"step_{RESUME_FROM_STEP}"
        if ckpt_path.exists():
            from peft import PeftModel
            unet = PeftModel.from_pretrained(unet.base_model.model, str(ckpt_path))
            print(f"Resumed from {ckpt_path}")
        else:
            print(f"Warning: checkpoint {ckpt_path} not found — starting fresh")

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataset = UkiyoeDataset(TRAIN_DIR, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=2, pin_memory=True, drop_last=True)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-2)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS,
        num_training_steps=MAX_STEPS)

    # Fast-forward scheduler if resuming
    if RESUME_FROM_STEP > 0:
        for _ in range(RESUME_FROM_STEP):
            lr_scheduler.step()

    # ── Training loop ─────────────────────────────────────────────────────────
    run_output_dir = OUTPUT_DIR / f"run_{RANK}_{LR:.0e}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    LOSS_LOG.parent.mkdir(parents=True, exist_ok=True)

    loss_csv = open(LOSS_LOG, "w", newline="")
    csv_writer = csv.writer(loss_csv)
    csv_writer.writerow(["step", "loss"])

    global_step  = 0
    micro_step   = 0
    running_loss = 0.0
    optimizer.zero_grad()

    unet.train()
    text_encoder.eval()
    vae.eval()

    pbar = tqdm(total=MAX_STEPS, desc=f"Training rank={RANK}")

    eval_prompts = [
        f"{TRIGGER_WORD}, Mount Fuji at sunrise",
        f"{TRIGGER_WORD}, samurai warrior portrait",
        f"{TRIGGER_WORD}, ocean waves crashing",
    ]

    while global_step < MAX_STEPS:
        for batch in loader:
            if global_step >= MAX_STEPS:
                break

            # Forward — encode with fp16 frozen models, train in fp32
            pixel_values = batch["pixel_values"].to(device)        # fp32
            input_ids    = batch["input_ids"].to(device)

            with torch.no_grad():
                latents = vae.encode(pixel_values.half()).latent_dist.sample()
                latents = latents.float() * vae.config.scaling_factor
                encoder_hidden = text_encoder(input_ids)[0].float()

            noise     = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy     = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast("cuda", dtype=torch.float16):
                noise_pred = unet(noisy, timesteps, encoder_hidden).sample

            loss = F.mse_loss(noise_pred.float(), noise.float()) / GRAD_ACCUM

            loss.backward()
            micro_step += 1
            running_loss += loss.item() * GRAD_ACCUM

            if micro_step % GRAD_ACCUM != 0:
                continue

            torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], MAX_NORM)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            pbar.update(1)

            if global_step % LOG_EVERY == 0:
                avg = running_loss / LOG_EVERY
                pbar.set_postfix({"loss": f"{avg:.4f}"})
                csv_writer.writerow([global_step, f"{avg:.6f}"])
                loss_csv.flush()
                running_loss = 0.0

            if global_step % CKPT_EVERY == 0:
                ckpt = run_output_dir / f"step_{global_step}"
                unet.save_pretrained(str(ckpt))
                pbar.write(f"✅ Checkpoint saved → {ckpt}")

                generate_samples(
                    unet, vae, text_encoder, tokenizer,
                    noise_scheduler, device, global_step, eval_prompts,
                )

    pbar.close()
    loss_csv.close()

    # Final save
    final = run_output_dir / "step_2000"
    unet.save_pretrained(str(final))
    print(f"\nTraining complete → {final}")


if __name__ == "__main__":
    main()