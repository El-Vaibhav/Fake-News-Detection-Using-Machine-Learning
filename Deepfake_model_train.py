
import os
import torch
import timm
import numpy as np
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= FIX CORRUPT IMAGES =================
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= CONFIG =================
DATASET_PATH = "/kaggle/input/datasets/superpotato9/dalle-recognition-dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 2   # increase later if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", DEVICE)

# ================= LOAD DATA =================
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

paths, labels = [], []

for cls in ["real", "fakeV2/fake-v2"]:
    folder = os.path.join(DATASET_PATH, cls)

    for f in os.listdir(folder):
        if not f.lower().endswith(VALID_EXTENSIONS):
            continue

        full_path = os.path.join(folder, f)
        paths.append(full_path)
        labels.append(0 if "real" in cls else 1)

print("Total images:", len(paths))

# ================= SPLIT =================
train_p, val_p, train_l, val_l = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# ================= AUGMENTATION =================
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),

    # robustness
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.RandomApply([transforms.ColorJitter(0.3,0.3)], p=0.5),

    transforms.RandomApply([
        transforms.Lambda(lambda x: x.convert("L").convert("RGB"))
    ], p=0.2),

    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),

    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ================= DATASET =================
class ImgDataset(Dataset):
    def __init__(self, paths, labels, tfm):
        self.paths = paths
        self.labels = labels
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        try:
            img = Image.open(path)

            if img.mode != "RGB":
                img = img.convert("RGB")

        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        return self.tfm(img), torch.tensor(self.labels[idx])

# ================= DATALOADER =================
train_loader = DataLoader(
    ImgDataset(train_p, train_l, train_tfms),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    ImgDataset(val_p, val_l, val_tfms),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ================= LIGHTWEIGHT MODEL =================
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
model = model.to(DEVICE)

# ================= LOSS =================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ================= TRAIN =================
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)   # ✅ FIXED

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ================= VALIDATION =================
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    scheduler.step()

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "/kaggle/working/model.pth")

print("\n🔥 Best Validation Accuracy:", best_acc)

# ================= OPTIONAL QUANTIZATION =================
model.cpu()

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), "/kaggle/working/model_quantized.pth")

print("✅ Saved lightweight quantized model (~10MB)")