import os
import gc
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from peft import PeftModel

os.environ["HF_HOME"] = "/kaggle/working/hf_cache"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
OUTPUT_BASE = "/kaggle/working/ukiyo-lora/eval/generated"

RUNS = [
    {"name": "run_4",  "checkpoint": "/kaggle/working/ukiyo-lora/output/run_4_1e-4/step_2000"},
    {"name": "run_16", "checkpoint": "/kaggle/working/ukiyo-lora/output/run_16_1e-04/step_2000"},
    {"name": "run_32", "checkpoint": "/kaggle/working/ukiyo-lora/output/run_32_1e-04/step_2000"},
]

PROMPTS = [
    "ukiyoe style, Mount Fuji at sunrise with cherry blossoms",
    "ukiyoe style, ocean waves crashing on rocks",
    "ukiyoe style, woman in kimono standing by river",
    "ukiyoe style, pine trees in snow, winter landscape",
    "ukiyoe style, samurai warrior on horseback",
]

IMAGES_PER_PROMPT = 4  # 5 prompts × 4 = 20 images per run
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
SEED = 42


def make_blank_image():
    return Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))


def run_generation(run_name, checkpoint_path):
    out_dir = os.path.join(OUTPUT_BASE, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Loading pipeline for {run_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint_path)
    pipe.unet = pipe.unet.to(torch.float16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    image_idx = 1
    for prompt_idx, prompt in enumerate(PROMPTS):
        for img_in_prompt in range(IMAGES_PER_PROMPT):
            filename = f"{image_idx:03d}.png"
            out_path = os.path.join(out_dir, filename)
            print(f"Generating {run_name} image {image_idx}/20...")

            try:
                generator = torch.Generator("cuda").manual_seed(SEED + img_in_prompt)
                with torch.autocast("cuda", dtype=torch.float16):
                    result = pipe(
                        prompt,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=generator,
                    )
                image = result.images[0]
            except Exception as e:
                print(f"  ERROR on {filename} (prompt {prompt_idx + 1}, img {img_in_prompt + 1}): {e}")
                image = make_blank_image()

            image.save(out_path)
            image_idx += 1

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nDone with {run_name}. GPU cache cleared.")


if __name__ == "__main__":
    for run in RUNS:
        run_generation(run["name"], run["checkpoint"])

    print("\nAll runs complete.")
    print(f"Images saved to: {OUTPUT_BASE}")
