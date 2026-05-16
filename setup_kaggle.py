"""
Kaggle environment setup and verification for SD 1.5 LoRA training.
Safe to re-run — all steps are idempotent.
"""

import os
import subprocess
import sys
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str]) -> int:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "Success" if ok else "Fails"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    return ok


# ── 1. Install packages ───────────────────────────────────────────────────────

print("\n── 1. Installing packages ──────────────────────────────────────────")

req_path = Path("/kaggle/working/ukiyo-lora/requirements.txt")
pip_base = [sys.executable, "-m", "pip", "install", "--quiet"]

if req_path.exists():
    rc = run(pip_base + ["-r", str(req_path)])
    check("requirements.txt", rc == 0, str(req_path))
else:
    check("requirements.txt", False, f"not found at {req_path}")

for pkg in ["imagehash", "clean-fid"]:
    rc = run(pip_base + [pkg])
    check(pkg, rc == 0)


# ── 2. HuggingFace cache dir ──────────────────────────────────────────────────

print("\n── 2. HuggingFace cache ────────────────────────────────────────────")

HF_CACHE = "/kaggle/working/hf_cache"
os.environ["HF_HOME"] = HF_CACHE
Path(HF_CACHE).mkdir(parents=True, exist_ok=True)
check("HF_HOME set", os.environ.get("HF_HOME") == HF_CACHE, HF_CACHE)


# ── 3. Download model weights ─────────────────────────────────────────────────

print("\n── 3. Downloading SD 1.5 weights (UNet + text encoder) ─────────────")

MODEL_ID = "runwayml/stable-diffusion-v1-5"
unet = None
text_encoder = None

try:
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    check("UNet downloaded", True, MODEL_ID)
except Exception as e:
    check("UNet downloaded", False, str(e))

try:
    from transformers import CLIPTextModel
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    check("Text encoder downloaded", True, MODEL_ID)
except Exception as e:
    check("Text encoder downloaded", False, str(e))


# ── 4. Verification ───────────────────────────────────────────────────────────

print("\n── 4. Verification ─────────────────────────────────────────────────")

import torch

# torch + CUDA
torch_ok = check("torch importable", True, torch.__version__)
cuda_ok  = check("CUDA available", torch.cuda.is_available())

gpu_name = ""
vram_gb  = 0.0
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    check("GPU detected", True, f"{gpu_name}  {vram_gb:.1f} GB VRAM")
else:
    check("GPU detected", False, "no CUDA device found")

# UNet parameter count
unet_params = 0
if unet is not None:
    unet_params = sum(p.numel() for p in unet.parameters())
    check("UNet param count", unet_params > 0, f"{unet_params / 1e6:.1f} M params")
else:
    check("UNet param count", False, "UNet not loaded")

# LoRA applicability via peft
lora_ok = False
try:
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    # Clone so we don't mutate the cached unet
    import copy
    unet_test = copy.deepcopy(unet) if unet is not None else None
    if unet_test is not None:
        lora_model = get_peft_model(unet_test, lora_cfg)
        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        lora_ok = trainable > 0
        check("LoRA applicable (peft)", lora_ok, f"{trainable:,} trainable params")
        del lora_model, unet_test
    else:
        check("LoRA applicable (peft)", False, "UNet not loaded")
except Exception as e:
    check("LoRA applicable (peft)", False, str(e))


# ── 5. Collect versions ───────────────────────────────────────────────────────

def pkg_version(name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return "n/a"


versions = {
    "torch":        torch.__version__,
    "diffusers":    pkg_version("diffusers"),
    "peft":         pkg_version("peft"),
    "accelerate":   pkg_version("accelerate"),
    "transformers": pkg_version("transformers"),
    "torchvision":  pkg_version("torchvision"),
    "clean-fid":    pkg_version("clean-fid"),
    "imagehash":    pkg_version("ImageHash"),
    "Pillow":       pkg_version("Pillow"),
    "scipy":        pkg_version("scipy"),
    "pandas":       pkg_version("pandas"),
    "matplotlib":   pkg_version("matplotlib"),
}

cuda_version = torch.version.cuda or "n/a"

info_lines = [
    "=== environment_info.txt ===",
    f"python:       {sys.version.split()[0]}",
    f"cuda:         {cuda_version}",
    f"gpu_name:     {gpu_name or 'n/a'}",
    f"vram_gb:      {vram_gb:.1f}",
    f"unet_params:  {unet_params / 1e6:.1f} M",
    "",
    "--- package versions ---",
] + [f"{k:<14}{v}" for k, v in versions.items()]

info_text = "\n".join(info_lines)

# ── 6. Save environment_info.txt ──────────────────────────────────────────────

out_path = Path("/kaggle/working/environment_info.txt")
try:
    out_path.write_text(info_text + "\n")
    check("environment_info.txt saved", True, str(out_path))
except Exception as e:
    # Fallback: save next to this script (local dev / non-Kaggle)
    fallback = Path(__file__).parent.parent / "environment_info.txt"
    fallback.write_text(info_text + "\n")
    check("environment_info.txt saved", True, f"(fallback) {fallback}")

print("\n" + info_text)
print("\n── Setup complete ──────────────────────────────────────────────────\n")
