# pix2pix.py
# Task-04: Image-to-Image Translation with cGAN (pix2pix)
# Facades dataset: architectural labels → real building photos

import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ============================================================
# CONFIG
# ============================================================

DATA_DIR    = "data/facades"
OUTPUT_DIR  = "outputs"
CHECKPOINT  = "checkpoints"

IMAGE_SIZE  = 256
BATCH_SIZE  = 4       # safe for 6GB VRAM
NUM_EPOCHS  = 50
LR          = 0.0002
LAMBDA_L1   = 100     # L1 loss weight (pix2pix paper value)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT, exist_ok=True)

print("=" * 60)
print("Task-04: pix2pix cGAN — Facades Dataset")
print("=" * 60)
print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
print()

# ============================================================
# DATASET
# ============================================================

class FacadesDataset(Dataset):
    def __init__(self, root, split="train"):
        self.files = sorted([
            os.path.join(root, split, f)
            for f in os.listdir(os.path.join(root, split))
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size
        # pix2pix images are side-by-side: [input | target]
        input_img  = img.crop((0, 0, w//2, h))
        target_img = img.crop((w//2, 0, w, h))
        return self.transform(input_img), self.transform(target_img)

train_ds = FacadesDataset(DATA_DIR, "train")
val_ds   = FacadesDataset(DATA_DIR, "val")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

print(f"Train: {len(train_ds)} images")
print(f"Val  : {len(val_ds)} images")

# ============================================================
# GENERATOR (U-Net)
# ============================================================

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, bn=True, dropout=False, act=True):
        super().__init__()
        layers = []
        if down:
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
        else:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False))
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)
        self.act   = nn.LeakyReLU(0.2) if down else nn.ReLU()
        self.use_act = act

    def forward(self, x):
        return self.block(self.act(x) if self.use_act else x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.e1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.e2 = UNetBlock(64,  128)
        self.e3 = UNetBlock(128, 256)
        self.e4 = UNetBlock(256, 512)
        self.e5 = UNetBlock(512, 512)
        self.e6 = UNetBlock(512, 512)
        self.e7 = UNetBlock(512, 512)
        self.e8 = UNetBlock(512, 512, bn=False)
        # Decoder
        self.d1 = UNetBlock(512,  512, down=False, dropout=True)
        self.d2 = UNetBlock(1024, 512, down=False, dropout=True)
        self.d3 = UNetBlock(1024, 512, down=False, dropout=True)
        self.d4 = UNetBlock(1024, 512, down=False)
        self.d5 = UNetBlock(1024, 256, down=False)
        self.d6 = UNetBlock(512,  128, down=False)
        self.d7 = UNetBlock(256,  64,  down=False)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        return self.out(torch.cat([d7, e1], 1))

# ============================================================
# DISCRIMINATOR (PatchGAN)
# ============================================================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], 1))

# ============================================================
# INIT MODELS
# ============================================================

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

G.apply(weights_init)
D.apply(weights_init)

# ============================================================
# LOSS & OPTIMIZERS
# ============================================================

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1  = nn.L1Loss()

opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# ============================================================
# SAVE SAMPLE IMAGES
# ============================================================

def save_samples(epoch, loader):
    G.eval()
    with torch.no_grad():
        inp, tgt = next(iter(loader))
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        gen = G(inp)

    def denorm(t):
        return ((t * 0.5 + 0.5) * 255).clamp(0, 255).byte()

    inp_img = denorm(inp[0]).cpu().permute(1,2,0).numpy()
    gen_img = denorm(gen[0]).cpu().permute(1,2,0).numpy()
    tgt_img = denorm(tgt[0]).cpu().permute(1,2,0).numpy()

    combined = np.concatenate([inp_img, gen_img, tgt_img], axis=1)
    Image.fromarray(combined).save(f"{OUTPUT_DIR}/epoch_{epoch:03d}.png")
    G.train()

# ============================================================
# TRAINING LOOP
# ============================================================

print("\nStarting Training...")
print(f"Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | Device: {DEVICE}\n")

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    d_losses, g_losses = [], []

    for inp, tgt in train_loader:
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)

        # ── Train Discriminator ──
        opt_D.zero_grad()
        fake     = G(inp).detach()
        real_out = D(inp, tgt)
        fake_out = D(inp, fake)
        loss_D   = (
            criterion_GAN(real_out, torch.ones_like(real_out)) +
            criterion_GAN(fake_out, torch.zeros_like(fake_out))
        ) * 0.5
        loss_D.backward()
        opt_D.step()

        # ── Train Generator ──
        opt_G.zero_grad()
        fake     = G(inp)
        fake_out = D(inp, fake)
        loss_G   = (
            criterion_GAN(fake_out, torch.ones_like(fake_out)) +
            LAMBDA_L1 * criterion_L1(fake, tgt)
        )
        loss_G.backward()
        opt_G.step()

        d_losses.append(loss_D.item())
        g_losses.append(loss_G.item())

    elapsed = time.time() - epoch_start
    print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
          f"D: {np.mean(d_losses):.4f} | "
          f"G: {np.mean(g_losses):.4f} | "
          f"Time: {elapsed:.1f}s")

    # Save samples every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        save_samples(epoch, val_loader)
        torch.save(G.state_dict(), f"{CHECKPOINT}/G_epoch_{epoch:03d}.pth")
        print(f"  ✓ Saved sample + checkpoint at epoch {epoch}")

print("\n" + "=" * 60)
print("Training Complete!")
print(f"Outputs saved to: {OUTPUT_DIR}/")
print("=" * 60)
