# generate_images.py
# Task-02: Image Generation with Pre-trained Models
# Optimized for NVIDIA RTX 3050 GPU

import os
import time
import torch
from diffusers import StableDiffusionPipeline

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HEIGHT = 512
WIDTH = 512

NUM_STEPS = 20          # 20 = much faster, excellent quality
GUIDANCE_SCALE = 7.5

SEED = 42               # Set to None for random images

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print(f"Loading Stable Diffusion on {DEVICE.upper()}...")
print("=" * 60)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)

pipe = pipe.to(DEVICE)

# ---------------- GPU Optimizations ----------------

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# Enable xFormers if installed
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("✓ xFormers enabled")
except Exception:
    print("xFormers not installed (this is okay)")

# Enable TF32 on RTX GPUs
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("✓ Pipeline Loaded Successfully!\n")

# ============================================================
# PROMPTS
# ============================================================

prompts = [
    {
        "name": "futuristic_city",
        "prompt": "A futuristic cyberpunk city at night, neon lights, flying cars, cinematic lighting, ultra detailed, masterpiece, 8k",
        "negative_prompt": "blurry, low quality, distorted, watermark, text"
    },
    {
        "name": "enchanted_forest",
        "prompt": "An enchanted forest with glowing mushrooms, magical fireflies, mystical fog, fantasy art, vibrant colors, ultra detailed",
        "negative_prompt": "blurry, dark, low quality"
    },
    {
        "name": "astronaut_ocean",
        "prompt": "An astronaut exploring an alien ocean with glowing bioluminescent creatures, underwater, cinematic lighting, ultra realistic",
        "negative_prompt": "ugly, blurry, low quality"
    },
]

# ============================================================
# RANDOM SEED
# ============================================================

if SEED is not None:
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
else:
    generator = None

# ============================================================
# GENERATE IMAGES
# ============================================================

print("=" * 60)
print("Generating Images")
print("=" * 60)

total_start = time.time()

for i, item in enumerate(prompts):

    print(f"\n[{i+1}/{len(prompts)}] {item['name']}")
    print("-" * 60)

    start = time.time()

    image = pipe(
        prompt=item["prompt"],
        negative_prompt=item["negative_prompt"],
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    save_path = os.path.join(OUTPUT_DIR, f"{item['name']}.png")
    image.save(save_path)

    elapsed = time.time() - start

    print(f"✓ Saved: {save_path}")
    print(f"✓ Time : {elapsed:.2f} seconds")

total = time.time() - total_start

print("\n" + "=" * 60)
print("All images generated successfully!")
print(f"Output Folder : {OUTPUT_DIR}")
print(f"Total Time    : {total:.2f} seconds")
print("=" * 60)