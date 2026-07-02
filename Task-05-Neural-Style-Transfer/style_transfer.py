# style_transfer.py
# Task-05: Neural Style Transfer
# Applies artistic style of one image to the content of another using VGG19

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy
import os

# ============================================================
# CONFIG
# ============================================================

CONTENT_IMG   = "images/content.jpg"
STYLE_IMG     = "images/style.jpg"
OUTPUT_DIR    = "outputs"
IMAGE_SIZE    = 512
NUM_STEPS     = 300
CONTENT_WEIGHT = 1
STYLE_WEIGHT   = 1000000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Task-05: Neural Style Transfer")
print("=" * 60)
print(f"Device  : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"Steps   : {NUM_STEPS}")
print()

# ============================================================
# IMAGE LOADER
# ============================================================

loader = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = loader(img).unsqueeze(0)
    return img.to(DEVICE, torch.float)

def save_image(tensor, path):
    img = tensor.cpu().clone().squeeze(0)
    img = transforms.ToPILImage()(img.clamp(0, 1))
    img.save(path)
    print(f"✓ Saved: {path}")

content_img = load_image(CONTENT_IMG)
style_img   = load_image(STYLE_IMG)

print(f"Content : {CONTENT_IMG}")
print(f"Style   : {STYLE_IMG}")
print()

# ============================================================
# VGG19 FEATURE EXTRACTOR
# ============================================================

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(2)])   # relu1_1
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(2, 7)])  # relu2_1
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(7, 12)]) # relu3_1
        self.slice4 = nn.Sequential(*[vgg[i] for i in range(12, 21)])# relu4_1
        self.slice5 = nn.Sequential(*[vgg[i] for i in range(21, 30)])# relu5_1
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return h1, h2, h3, h4, h5

vgg = VGG().to(DEVICE).eval()
print("✓ VGG19 loaded\n")

# ============================================================
# LOSS FUNCTIONS
# ============================================================

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(c, h * w)
    return torch.mm(features, features.t()) / (c * h * w)

def content_loss(gen, content):
    return nn.functional.mse_loss(gen, content)

def style_loss(gen_feats, style_feats):
    loss = 0
    for g, s in zip(gen_feats, style_feats):
        loss += nn.functional.mse_loss(gram_matrix(g), gram_matrix(s))
    return loss

# ============================================================
# EXTRACT FEATURES
# ============================================================

with torch.no_grad():
    content_feats = vgg(content_img)
    style_feats   = vgg(style_img)

# Use relu4_1 for content, all layers for style
content_target = content_feats[3].detach()
style_targets  = [f.detach() for f in style_feats]

# ============================================================
# OPTIMIZATION
# ============================================================

# Start from content image
generated = content_img.clone().requires_grad_(True)
optimizer = optim.LBFGS([generated])

print("Starting style transfer optimization...")
print("-" * 60)

step = [0]

while step[0] < NUM_STEPS:
    def closure():
        with torch.no_grad():
            generated.clamp_(0, 1)

        optimizer.zero_grad()
        gen_feats = vgg(generated)

        c_loss = content_loss(gen_feats[3], content_target) * CONTENT_WEIGHT
        s_loss = style_loss(gen_feats, style_targets) * STYLE_WEIGHT
        total  = c_loss + s_loss
        total.backward()

        step[0] += 1
        if step[0] % 50 == 0:
            print(f"Step [{step[0]:03d}/{NUM_STEPS}] "
                  f"Content: {c_loss.item():.2f} | "
                  f"Style: {s_loss.item():.2f} | "
                  f"Total: {total.item():.2f}")
        return total

    optimizer.step(closure)

# ============================================================
# SAVE RESULTS
# ============================================================

with torch.no_grad():
    generated.clamp_(0, 1)

save_image(generated, f"{OUTPUT_DIR}/stylized.png")

# Save side by side comparison
def to_pil(tensor):
    return transforms.ToPILImage()(tensor.cpu().squeeze(0).clamp(0, 1))

content_pil = to_pil(content_img)
style_pil   = to_pil(style_img)
result_pil  = to_pil(generated)

comparison = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE))
comparison.paste(content_pil, (0, 0))
comparison.paste(style_pil,   (IMAGE_SIZE, 0))
comparison.paste(result_pil,  (IMAGE_SIZE * 2, 0))
comparison.save(f"{OUTPUT_DIR}/comparison.png")
print(f"✓ Saved: {OUTPUT_DIR}/comparison.png")

print("\n" + "=" * 60)
print("Neural Style Transfer Complete!")
print(f"  stylized.png    → final result")
print(f"  comparison.png  → content | style | result")
print("=" * 60)