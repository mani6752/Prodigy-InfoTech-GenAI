# Task-02: Image Generation with Pre-trained Models
### Prodigy InfoTech — Generative AI Internship (Track: GA)

---

## Objective
Use a pre-trained generative model (Stable Diffusion v1.5) to create images from
text prompts — demonstrating text-to-image generation using diffusion models.

---

## Project Structure
```
task02_image_generation/
├── outputs/                    ← generated images saved here
├── generate_images.py          ← Step 1: generate 3 demo images
├── custom_prompt.py            ← Step 2: interactive custom prompts
├── requirements.txt
└── README.md
```

---

## Setup & Run (VS Code)

### Step 0 — Create & activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate demo images (3 pre-set prompts)
```bash
python generate_images.py
```
First run downloads ~4GB model weights (one-time only).

### Step 3 — Try your own prompts
```bash
python custom_prompt.py
```

---

## How It Works

```
Text Prompt
    ↓
Text Encoder (CLIP)  →  converts text to embeddings
    ↓
U-Net Denoiser  →  iteratively removes noise (30 steps)
    ↓
VAE Decoder  →  converts latent space to pixel image
    ↓
Output Image (512×512 PNG)
```

| Component | Role |
|-----------|------|
| CLIP encoder | Understands the meaning of the text prompt |
| U-Net | The core denoising neural network |
| VAE decoder | Converts compressed representation to image |
| Scheduler (PNDM) | Controls the denoising step schedule |

---

## Key Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `num_inference_steps` | 30 | More steps = higher quality (slower) |
| `guidance_scale` | 7.5 | Higher = follows prompt more strictly |
| `height / width` | 512×512 | Output image resolution |
| `negative_prompt` | "blurry, low quality..." | What to avoid in the image |
| `seed` | 42 | Fixed seed for reproducibility |

---

## Sample Prompts Used

1. **Futuristic city** — `"A futuristic city at night with neon lights, flying cars, cyberpunk style, highly detailed, 4k"`
2. **Enchanted forest** — `"An enchanted forest with glowing mushrooms, fireflies, mystical fog, fantasy art"`
3. **Astronaut ocean** — `"An astronaut exploring an alien ocean with bioluminescent creatures, cinematic lighting"`

---

## Tips for Better Prompts

| Do | Example |
|----|---------|
| Add art style | `oil painting`, `digital art`, `watercolor` |
| Add quality tags | `4k`, `highly detailed`, `sharp focus` |
| Add lighting | `golden hour`, `neon lights`, `cinematic` |
| Use negative prompts | `blurry`, `distorted`, `low quality` |

---

## Note on Speed
- **CPU:** ~5–10 minutes per image
- **GPU (CUDA):** ~15–30 seconds per image
- For faster results, use Google Colab with a free T4 GPU

---

## Tech Stack
- **Model:** Stable Diffusion v1.5 (runwayml)
- **Framework:** HuggingFace Diffusers + PyTorch
- **Libraries:** `diffusers`, `transformers`, `Pillow`
- **Architecture:** Latent Diffusion Model (LDM)

---

## References
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [HuggingFace Diffusers Docs](https://huggingface.co/docs/diffusers)
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [DALL-E mini (Craiyon)](https://www.craiyon.com/)
