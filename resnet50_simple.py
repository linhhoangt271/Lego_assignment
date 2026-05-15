"""
Simple ResNet50 Fine-tuning on Top 20 Classes
- ResNet50 pretrained backbone
- Top 20 classes only (78.9% of data = 13.7K samples)
- Standard cross-entropy loss
- Two-stage training: frozen backbone → full fine-tuning
- ~30 min training on GPU, ~2-3 hours on CPU
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from collections import Counter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "resnet50_top20_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMG_SIZE = 224  # ResNet standard
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS_FROZEN = 5
NUM_EPOCHS_FULL = 15
NUM_WORKERS = 4

# Top 20 classes
TOP_20_CLASSES = [
    'Town', 'Star Wars', 'Super Heroes', 'DUPLO', 'NINJAGO',
    'Friends', 'Collectible Minifigures', 'Harry Potter', 'Castle', 'Holiday & Event',
    'Disney', 'LEGO Ideas', 'BrickLink Designer Program', 'Minecraft', 'Sports',
    'Super Mario', 'Space', 'Pirates', 'Monkie Kid', 'LEGO Brand'
]

print(f"Training on {len(TOP_20_CLASSES)} classes: {', '.join(TOP_20_CLASSES)}")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
class MinifigDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
        except:
            # Fallback: create gray image if loading fails
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=128)

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]

def load_data():
    """Load minifigs.json and filter to top 20 classes."""
    print("Loading data...")
    with open(os.path.join(BASE_DIR, 'minifigs.json')) as f:
        minifigs = json.load(f)

    # Filter to top 20 classes
    class_to_idx = {cls: i for i, cls in enumerate(TOP_20_CLASSES)}

    image_paths = []
    labels = []
    skipped = 0

    for entry in minifigs:
        category = entry.get('category')
        if category not in class_to_idx:
            continue

        img_local_path = entry.get('img_local_path')
        if not img_local_path:
            skipped += 1
            continue

        img_path = os.path.join(BASE_DIR, img_local_path)
        if not os.path.exists(img_path):
            skipped += 1
            continue

        image_paths.append(img_path)
        labels.append(class_to_idx[category])

    print(f"Loaded {len(image_paths)} images, skipped {skipped}")
    print(f"Class distribution: {Counter(labels)}")

    return image_paths, labels, class_to_idx

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def evaluate(model, loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()

    # Load data
    image_paths, labels, class_to_idx = load_data()

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_dataset = MinifigDataset(X_train, y_train, train_transform)
    val_dataset = MinifigDataset(X_val, y_val, val_transform)

    # Weighted sampler for class balance
    class_counts = Counter(y_train)
    weights = [1.0 / class_counts[label] for label in y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Model: ResNet50 pretrained
    print("\nLoading ResNet50 pretrained on ImageNet...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(TOP_20_CLASSES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Phase 1: Frozen backbone, train only head (5 epochs)
    print("\n" + "="*70)
    print("PHASE 1: Frozen backbone - optimizing fc layer only (5 epochs)")
    print("="*70)

    # Only optimize the fc (head) parameters
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    best_val_f1 = 0.0

    for epoch in range(NUM_EPOCHS_FROZEN):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS_FROZEN}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"✓ Best model saved (F1={val_f1:.4f})")

    # Phase 2: Full fine-tuning (15 epochs)
    print("\n" + "="*70)
    print("PHASE 2: Full fine-tuning - all parameters (15 epochs)")
    print("="*70)

    # Now optimize all parameters with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)  # Lower LR for full training

    for epoch in range(NUM_EPOCHS_FULL):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS_FULL}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f"✓ Best model saved (F1={val_f1:.4f})")

    # Final evaluation on validation set
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
    _, _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, DEVICE)

    # Classification report
    report = classification_report(val_labels, val_preds, target_names=TOP_20_CLASSES, zero_division=0)
    print(report)

    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, xticklabels=TOP_20_CLASSES, yticklabels=TOP_20_CLASSES,
                cmap='Blues', annot=False, cbar=True)
    plt.title('Confusion Matrix - ResNet50 Top 20 Classes')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    print(f"\n✓ Confusion matrix saved")

    # Per-class F1 scores
    per_class_f1 = f1_score(val_labels, val_preds, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(TOP_20_CLASSES)), per_class_f1, color='steelblue')
    ax.set_xticks(range(len(TOP_20_CLASSES)))
    ax.set_xticklabels(TOP_20_CLASSES, rotation=45, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_f1.png'), dpi=150)
    print(f"✓ Per-class F1 chart saved")

    # Summary
    elapsed = time.time() - start_time
    acc = accuracy_score(val_labels, val_preds)
    macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

    summary = f"""
ResNet50 Fine-tuning Results
============================
Classes: {len(TOP_20_CLASSES)}
Total samples: {len(X_train) + len(X_val)}
Training time: {elapsed/60:.1f} minutes

Final Validation Metrics:
- Accuracy: {acc:.4f}
- Macro F1: {macro_f1:.4f}
- Weighted F1: {weighted_f1:.4f}

Per-class F1 scores:
"""
    for cls, f1 in zip(TOP_20_CLASSES, per_class_f1):
        summary += f"\n  {cls:30s}: {f1:.4f}"

    print(summary)

    with open(os.path.join(OUTPUT_DIR, 'results_summary.txt'), 'w') as f:
        f.write(summary)

    print(f"\n✓ All results saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
