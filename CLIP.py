# -*- coding: utf-8 -*-
"""Untitled5.ipynb
# ─────────────────────────────────────────────
#  USER CONFIGURATION  ← change these settings
# ─────────────────────────────────────────────

DATASET_ROOT   = '/content/dataset_images/train'
AUTO_SPLIT     = True
SPLIT_RATIO    = 0.8

EPOCHS         = 4
BATCH_SIZE     = 16
LEARNING_RATE  = 1e-6              # UPDATED: Lowered for numerical stability
CLIP_MODEL     = "ViT-B/32"
FREEZE_CLIP    = False
OUTPUT_DIR     = "./outputs"

CLASS_NAMES    = ["REAL", "FAKE"]

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────

import os, sys, random, shutil, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

try:
    import clip
except ImportError:
    print("\n[ERROR] CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
#  DATASET CLASS
# ─────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(self, root, class_names, transform=None):
        self.transform   = transform
        self.class_names = class_names
        self.samples     = []

        for label, cls in enumerate(class_names):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                print(f"  [WARNING] Folder not found: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if os.path.splitext(fname)[1].lower() in self.EXTENSIONS:
                    self.samples.append((os.path.join(cls_dir, fname), label))

        random.shuffle(self.samples)
        print(f"  Loaded {len(self.samples)} images from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label

# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=2, freeze_clip=True):
        super().__init__()
        self.clip       = clip_model
        self.freeze_clip = freeze_clip

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        embed_dim = clip_model.visual.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512), # Increased slightly for better learning
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images):
        # We ensure images are in Float32 before passing to CLIP
        features = self.clip.encode_image(images.type(self.clip.dtype))
        logits = self.classifier(features.float())
        return logits

# ─────────────────────────────────────────────
#  TRAIN FUNCTION (WITH GRADIENT CLIPPING)
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)

        # Check for nan loss before proceeding
        if torch.isnan(loss):
            continue

        loss.backward()

        # FIX 1: Gradient Clipping to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total

# ─────────────────────────────────────────────
#  EVALUATION & METRICS (Simplified for brevity)
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)

        total_loss  += loss.item() * labels.size(0)
        total       += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, np.array(all_preds), np.array(all_labels)

# ─────────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────────

def main():
    print(f"\n{'='*50}\n CLIP AI Image Detector (Stable Version)\n{'='*50}")

    # 1. Load CLIP and Force Float32
    print("[1/4] Loading CLIP model...")
    clip_model, preprocess = clip.load(CLIP_MODEL, device=DEVICE, jit=False)

    # FIX 2: Force model to Float32 to avoid NaN issues with FP16
    clip_model = clip_model.float()

    # 2. Build Datasets
    print("[2/4] Building datasets...")
    full_ds = ImageFolderDataset(DATASET_ROOT, CLASS_NAMES, preprocess)
    n_train = int(len(full_ds) * SPLIT_RATIO)
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # 3. Build Model
    model = CLIPClassifier(clip_model, num_classes=len(CLASS_NAMES), freeze_clip=FREEZE_CLIP).to(DEVICE)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    print(f"[3/4] Training for {EPOCHS} epochs...")
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_preds, v_labels = evaluate(model, val_loader)
        v_acc = accuracy_score(v_labels, v_preds)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"Epoch {epoch}: Train Loss {t_loss:.4f} | Val Acc {v_acc*100:.2f}% (Saved!)")
        else:
            print(f"Epoch {epoch}: Train Loss {t_loss:.4f} | Val Acc {v_acc*100:.2f}%")

    print(f"\n{'='*50}\n DONE! Model saved in {OUTPUT_DIR}\n{'='*50}")

if __name__ == "__main__":
    main()
