# =========================
# 1. IMPORTS
# =========================
import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. LOAD DATA
# =========================
csv_path = "/kaggle/input/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized/Data_Entry_2017.csv"
data = pd.read_csv(csv_path)


# =========================
# 3. FIXED IMAGE PATH MAPPING (ROBUST)
# =========================
image_root = "/kaggle/input/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized/"

all_image_files = glob(os.path.join(image_root, "**/*.png"), recursive=True)

print("Total image files found:", len(all_image_files))

image_paths = {
    os.path.basename(x): x
    for x in all_image_files
}

data["path"] = data["Image Index"].map(image_paths.get)

print("Total rows in CSV:", len(data))
print("Mapped image paths:", data["path"].notnull().sum())

# Drop missing
data = data.dropna(subset=["path"])

print("After dropna:", len(data))

# Safety check
assert len(data) > 0, "Dataset is empty! Fix image path."


# =========================
# 4. CREATE MULTI-LABEL TARGETS
# =========================
from itertools import chain

data["Finding Labels"] = data["Finding Labels"].map(
    lambda x: x.replace("No Finding", "")
)

all_labels = np.unique(
    list(chain(*data["Finding Labels"].map(lambda x: x.split("|"))))
)

all_labels = [l for l in all_labels if len(l) > 0]

for label in all_labels:
    data[label] = data["Finding Labels"].map(
        lambda x: 1.0 if label in x else 0
    )

print("Number of labels:", len(all_labels))


# =========================
# 5. PREPARE ARRAYS
# =========================
image_list = data["path"].values
label_matrix = data[all_labels].values.astype(np.float32)

train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    image_list,
    label_matrix,
    test_size=0.2,
    random_state=42
)

print("Train samples:", len(train_imgs))
print("Val samples:", len(val_imgs))


# =========================
# 6. DATASET
# =========================
class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx])

        return img, label


# =========================
# 7. TRANSFORMS
# =========================
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


# =========================
# 8. DATALOADER
# =========================
train_ds = ChestXrayDataset(train_imgs, train_labels, train_tf)
val_ds   = ChestXrayDataset(val_imgs, val_labels, val_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)


# =========================
# 9. MODEL (EFFICIENTNET-B0)
# =========================
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights="DEFAULT")
        
        # Get input features of classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# Initialize model
model = EfficientNetModel(len(all_labels)).to(device)


# =========================
# 10. FOCAL LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()


criterion = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


# =========================
# 11. METRICS
# =========================
def evaluate(logits, targets):

    probs = torch.sigmoid(logits).cpu().numpy()
    targets = targets.cpu().numpy()

    preds = (probs > 0.5).astype(int)

    mAP = average_precision_score(targets, probs, average='macro')

    TP = (preds * targets).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()

    OP = TP / (TP + FP + 1e-8)
    OR = TP / (TP + FN + 1e-8)
    OF1 = 2 * OP * OR / (OP + OR + 1e-8)

    CP, CR = [], []
    for i in range(targets.shape[1]):
        tp = ((preds[:, i] == 1) & (targets[:, i] == 1)).sum()
        fp = ((preds[:, i] == 1) & (targets[:, i] == 0)).sum()
        fn = ((preds[:, i] == 0) & (targets[:, i] == 1)).sum()

        CP.append(tp / (tp + fp + 1e-8))
        CR.append(tp / (tp + fn + 1e-8))

    CP = np.mean(CP)
    CR = np.mean(CR)
    CF1 = 2 * CP * CR / (CP + CR + 1e-8)

    Macro_AUC = roc_auc_score(targets, probs, average='macro')
    Micro_F1 = f1_score(targets.flatten(), preds.flatten())

    return {
        "mAP": mAP,
        "OP": OP, "OR": OR, "OF1": OF1,
        "CP": CP, "CR": CR, "CF1": CF1,
        "Macro_AUC": Macro_AUC,
        "Micro_F1": Micro_F1
    }


# =========================
# 12. TRAIN LOOP
# =========================
epochs = 50
best_of1 = 0

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for imgs, labels in train_loader:

        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(imgs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # ===== VALIDATION =====
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)

            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    metrics = evaluate(all_logits, all_targets)

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {train_loss:.4f}")
    print(metrics)

    # Save best model
    if metrics["OF1"] > best_of1:
        best_of1 = metrics["OF1"]
        torch.save(model.state_dict(), "/kaggle/working/best_efficientnet.pth")


# =========================
# 13. LOAD BEST MODEL (OPTIONAL)
# =========================
# model.load_state_dict(torch.load("/kaggle/working/best_densenet.pth"))
