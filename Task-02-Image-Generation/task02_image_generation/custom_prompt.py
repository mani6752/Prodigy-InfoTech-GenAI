# custom_prompt.py
# Interactive Stable Diffusion Image Generator (GPU Optimized)

import os
import time
import torch
from diffusers import StableDiffusionPipeline

# =====================================================
# CONFIG
# =====================================================

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEIGHT = 512
WIDTH = 512
GUIDANCE_SCALE = 7.5
DEFAULT_STEPS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# CUDA OPTIMIZATION
# =====================================================

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# =====================================================
# LOAD PIPELINE
# =====================================================

print(f"\nLoading Stable Diffusion on {DEVICE.upper()}...")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None,
)

pipe = pipe.to(DEVICE)

# Memory optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

print("Pipeline Loaded Successfully!")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

print()

# =====================================================
# INTERACTIVE MODE
# =====================================================

print("=" * 60)
print(" Stable Diffusion Image Generator ")
print("=" * 60)
print("Type 'quit' anytime to exit.")
print()

counter = 1

while True:

    print(f"\nImage #{counter}")

    prompt = input("Prompt: ").strip()

    if prompt.lower() in ["quit", "exit", "q"]:
        break

    if prompt == "":
        continue

    negative = input(
        "Negative Prompt (Enter = default): "
    ).strip()

    if negative == "":
        negative = (
            "blurry, low quality, bad anatomy, distorted, "
            "extra fingers, watermark, text"
        )

    steps = input(f"Inference Steps [{DEFAULT_STEPS}]: ").strip()

    if steps == "":
        steps = DEFAULT_STEPS
    else:
        steps = int(steps)

    seed = input("Seed (Enter = random): ").strip()

    if seed == "":
        generator = None
    else:
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    print("\nGenerating...\n")

    start = time.time()

    with torch.inference_mode():

        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=steps,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).images[0]

    elapsed = time.time() - start

    filename = f"custom_{counter:03d}.png"

    path = os.path.join(OUTPUT_DIR, filename)

    image.save(path)

    print(f"Saved: {path}")
    print(f"Time : {elapsed:.1f} seconds")

    counter += 1

print("\nDone!")