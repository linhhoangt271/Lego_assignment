"""
Option B Improved — Techniques to handle class imbalance & boost val accuracy:
  1. Stronger data augmentation (RandAugment + CutMix + MixUp)
  2. Focal Loss (designed for class imbalance, down-weights easy examples)
  3. Larger image size (224 vs 128)
  4. Unfreeze more layers (fine-tune from layer 4 instead of 6)
  5. More epochs + early stopping + cosine annealing LR
  6. Label smoothing
  7. Softer class weighting (sqrt of inverse freq instead of raw inverse)
"""

import json, os, time, warnings, random, math
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, top_k_accuracy_score
)
from collections import Counter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = "/Users/linh/Downloads/KUL /11. Advanced Analytics/big_data_assignment_2"
OUTPUT_DIR = os.path.join(BASE_DIR, "optionB_improved_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

IMG_SIZE = 224       # Larger images → more detail
BATCH_SIZE = 32      # Smaller batch for larger images
NUM_EPOCHS = 20
LR = 5e-4
NUM_WORKERS = 0
PATIENCE = 5         # Early stopping patience
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0    # Focal loss gamma
MIXUP_ALPHA = 0.2    # MixUp alpha
CUTMIX_ALPHA = 1.0   # CutMix alpha
CUTMIX_PROB = 0.3    # Probability of applying CutMix per batch
MIXUP_PROB = 0.3     # Probability of applying MixUp per batch

# ============================================================
# MERGE MAP (same as Option B)
# ============================================================
MERGE_MAP = {
    'The LEGO NINJAGO Movie': 'NINJAGO',
    'The LEGO Movie': 'LEGO Movies', 'The LEGO Movie 2': 'LEGO Movies',
    'DC Super Hero Girls': 'Super Heroes', 'Spider-Man': 'Super Heroes', 'Batman I': 'Super Heroes',
    'Elves': 'Friends & Fantasy', 'Friends': 'Friends & Fantasy',
    'DUPLO': 'Preschool', 'Primo': 'Preschool', 'Belville': 'Preschool', 'Scala': 'Preschool',
    'For Juniors': 'Preschool', 'Homemaker': 'Preschool', 'Fabuland': 'Preschool',
    'Pirates of the Caribbean': 'Pirates',
    'Castle': 'Castle & Medieval', 'Vikings': 'Castle & Medieval', 'Ninja': 'Castle & Medieval',
    'NEXO KNIGHTS': 'Castle & Medieval',
    'Space': 'Space & Sci-Fi', 'Aquazone': 'Space & Sci-Fi', 'Alpha Team': 'Space & Sci-Fi',
    'Exo-Force': 'Space & Sci-Fi', 'Power Miners': 'Space & Sci-Fi', 'Rock Raiders': 'Space & Sci-Fi',
    'Agents': 'Space & Sci-Fi', 'Ultra Agents': 'Space & Sci-Fi', 'Atlantis': 'Space & Sci-Fi',
    'Adventurers': 'Adventure', 'Indiana Jones': 'Adventure', "Pharaoh's Quest": 'Adventure',
    'Dino Attack': 'Adventure', 'Dino': 'Adventure', 'Western': 'Adventure',
    'SPEED CHAMPIONS': 'Racing & Vehicles', 'Racers': 'Racing & Vehicles',
    'World Racers': 'Racing & Vehicles', 'Speed Racer': 'Racing & Vehicles', 'Cars': 'Racing & Vehicles',
    'LEGENDS OF CHIMA': 'Fantasy', 'DREAMZzz': 'Fantasy', 'Hidden Side': 'Fantasy',
    'Monster Fighters': 'Fantasy',
    'The Hobbit and The Lord of the Rings': 'Middle-Earth',
    'Teenage Mutant Ninja Turtles': 'Licensed Entertainment',
    'SpongeBob SquarePants': 'Licensed Entertainment', 'Scooby-Doo': 'Licensed Entertainment',
    'The Simpsons': 'Licensed Entertainment', 'Ghostbusters': 'Licensed Entertainment',
    'Stranger Things': 'Licensed Entertainment', 'The Angry Birds Movie': 'Licensed Entertainment',
    'Trolls World Tour': 'Licensed Entertainment', 'The Incredibles': 'Licensed Entertainment',
    'Toy Story': 'Licensed Entertainment', 'Wednesday': 'Licensed Entertainment',
    'Wicked': 'Licensed Entertainment', 'Despicable Me and Minions': 'Licensed Entertainment',
    'The Lone Ranger': 'Licensed Entertainment', 'Prince of Persia': 'Licensed Entertainment',
    'Island Xtreme Stunts': 'Licensed Entertainment', 'The Powerpuff Girls': 'Licensed Entertainment',
    'Back to the Future': 'Licensed Entertainment', 'Friends TV Series': 'Licensed Entertainment',
    'Queer Eye': 'Licensed Entertainment', 'Lightyear': 'Licensed Entertainment',
    'Bluey': 'Licensed Entertainment', "Gabby's Dollhouse": 'Licensed Entertainment',
    'Dune': 'Licensed Entertainment', 'Star Trek': 'Licensed Entertainment',
    'Overwatch': 'Video Games', 'Fortnite': 'Video Games', 'Sonic the Hedgehog': 'Video Games',
    'Horizon': 'Video Games', 'The Legend of Zelda': 'Video Games', 'Animal Crossing': 'Video Games',
    'Avatar The Last Airbender': 'Video Games', 'Avatar': 'Video Games', 'One Piece': 'Video Games',
    'Collectible Minifigures': 'Collectible & Special', 'Holiday & Event': 'Collectible & Special',
    'Dimensions': 'Collectible & Special', 'Vidiyo': 'Collectible & Special',
    'Games': 'Collectible & Special',
    'LEGO Ideas': 'LEGO Promotional', 'BrickLink Designer Program': 'LEGO Promotional',
    'LEGO Brand': 'LEGO Promotional', 'LEGOLAND': 'LEGO Promotional',
    'LEGOLAND Parks': 'LEGO Promotional', 'Promotional': 'LEGO Promotional',
    'Studios': 'LEGO Promotional', 'FIRST LEGO League': 'LEGO Promotional',
    'Educational & Dacta': 'Education & Technical', 'BIONICLE': 'Education & Technical',
    'Hero Factory': 'Education & Technical', 'Technic': 'Education & Technical',
    'Basic': 'Education & Technical', 'FreeStyle': 'Education & Technical',
    'Master Builder Academy': 'Education & Technical', 'Building Bigger Thinking': 'Education & Technical',
    'Discovery': 'Education & Technical', 'Quatro': 'Education & Technical',
    'Universe': 'Education & Technical', 'Fusion': 'Education & Technical',
    'Nike': 'Education & Technical', 'Architecture': 'Education & Technical',
    'Clikits': 'Education & Technical', 'Unikitty!': 'Education & Technical',
}

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss: down-weights well-classified examples, focuses on hard ones."""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Apply label smoothing
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        if self.label_smoothing > 0:
            # Focal with smoothed labels
            focal_weight = (1 - probs).pow(self.gamma)
            loss = -(focal_weight * smooth_targets * log_probs).sum(dim=1)
        else:
            # Standard focal
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt).pow(self.gamma)
            loss = focal_weight * ce_loss

        # Apply class weights
        if self.weight is not None and self.label_smoothing == 0:
            pass  # already applied in ce_loss
        elif self.weight is not None:
            w = self.weight[targets]
            loss = loss * w

        return loss.mean()


# ============================================================
# CUTMIX & MIXUP
# ============================================================
def rand_bbox(size, lam):
    """Random bounding box for CutMix."""
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_rat), int(W * cut_rat)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    return y1, y2, x1, x2


def apply_cutmix(images, targets, alpha=1.0):
    """Apply CutMix: cut a patch from one image, paste onto another."""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    y1, y2, x1, x2 = rand_bbox(images.size(), lam)
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    # Adjust lambda for actual area
    lam = 1 - ((y2 - y1) * (x2 - x1) / (images.size(-1) * images.size(-2)))
    targets_a, targets_b = targets, targets[index]
    return images, targets_a, targets_b, lam


def apply_mixup(images, targets, alpha=0.2):
    """Apply MixUp: blend two images and their labels."""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, lam


# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
with open(os.path.join(BASE_DIR, "minifigs.json")) as f:
    data = json.load(f)

valid_data = [d for d in data if d.get('img_local_path') and
              os.path.exists(os.path.join(BASE_DIR, d['img_local_path']))]

for d in valid_data:
    d['merged_category'] = MERGE_MAP.get(d['category'], d['category'])

categories = [d['merged_category'] for d in valid_data]
cat_counts = Counter(categories)
cat_names = sorted(cat_counts.keys())
cat2idx = {c: i for i, c in enumerate(cat_names)}
idx2cat = {i: c for c, i in cat2idx.items()}
num_classes = len(cat_names)
print(f"Valid images: {len(valid_data)}, Merged classes: {num_classes}")

# Split (same as Option B for fair comparison)
labels = [cat2idx[d['merged_category']] for d in valid_data]
label_counts = Counter(labels)
small_idx = [i for i, l in enumerate(labels) if label_counts[l] < 7]
big_idx = [i for i, l in enumerate(labels) if label_counts[l] >= 7]
big_data = [valid_data[i] for i in big_idx]
big_labels = [labels[i] for i in big_idx]
train_data_big, temp_data, train_labels_big, temp_labels = train_test_split(
    big_data, big_labels, test_size=0.3, random_state=42, stratify=big_labels)
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
train_data = train_data_big + [valid_data[i] for i in small_idx]
train_labels = train_labels_big + [labels[i] for i in small_idx]
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# ============================================================
# DATASET with stronger augmentation
# ============================================================
class MinifigDataset(Dataset):
    def __init__(self, records, labels, transform):
        self.records, self.labels, self.transform = records, labels, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        try: img = Image.open(os.path.join(BASE_DIR, self.records[idx]['img_local_path'])).convert('RGB')
        except: img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.transform(img), self.labels[idx]

# Strong augmentation with RandAugment
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),   # Resize slightly larger
    transforms.RandomCrop(IMG_SIZE),                      # Random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(20),
    transforms.RandAugment(num_ops=2, magnitude=9),       # RandAugment
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # Random erasing
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = MinifigDataset(train_data, train_labels, train_transform)
val_dataset = MinifigDataset(val_data, val_labels, eval_transform)
test_dataset = MinifigDataset(test_data, test_labels, eval_transform)

# ============================================================
# SOFTER CLASS WEIGHTS (sqrt of inverse freq)
# ============================================================
train_label_counts = Counter(train_labels)

# Sqrt weighting: less aggressive than pure inverse, prevents rare class domination
class_weights = torch.zeros(num_classes)
for i in range(num_classes):
    count = train_label_counts.get(i, 1)
    class_weights[i] = 1.0 / math.sqrt(count)  # sqrt inverse instead of raw inverse
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = class_weights.to(DEVICE)

# Sampler also uses sqrt weighting
sample_weights = [1.0 / math.sqrt(train_label_counts[l]) for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ============================================================
# MODEL — Unfreeze more layers
# ============================================================
print("Building model...")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze only first 4 layers (vs 6 before) → more fine-tuning
for param in model.features[:4].parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.4),    # Slightly higher dropout
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable ({trainable_params/total_params*100:.1f}%)")

# Focal Loss with label smoothing
criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
criterion_val = nn.CrossEntropyLoss()  # Standard CE for val (comparable metric)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# ============================================================
# TRAINING with CutMix + MixUp
# ============================================================
print(f"\nTraining for up to {NUM_EPOCHS} epochs (early stopping patience={PATIENCE})...")
print(f"Techniques: RandAugment, CutMix(p={CUTMIX_PROB}), MixUp(p={MIXUP_PROB}), "
      f"FocalLoss(γ={FOCAL_GAMMA}), LabelSmoothing({LABEL_SMOOTHING}), sqrt-weighting")
print("-" * 80)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for images, lbl in train_loader:
        images, lbl = images.to(DEVICE), lbl.to(DEVICE)

        # Randomly apply CutMix or MixUp
        r = random.random()
        if r < CUTMIX_PROB:
            images, targets_a, targets_b, lam = apply_cutmix(images, lbl, CUTMIX_ALPHA)
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        elif r < CUTMIX_PROB + MIXUP_PROB:
            images, targets_a, targets_b, lam = apply_mixup(images, lbl, MIXUP_ALPHA)
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, lbl)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        train_correct += preds.eq(lbl).sum().item()
        train_total += images.size(0)

    scheduler.step(epoch)
    train_loss /= train_total
    train_acc = train_correct / train_total

    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, lbl in val_loader:
            images, lbl = images.to(DEVICE), lbl.to(DEVICE)
            outputs = model(images)
            loss = criterion_val(outputs, lbl)
            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            val_correct += preds.eq(lbl).sum().item()
            val_total += images.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    lr = optimizer.param_groups[0]['lr']
    elapsed = time.time() - t0
    marker = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        patience_counter = 0
        marker = " ★ best"
    else:
        patience_counter += 1

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {lr:.6f} | {elapsed:.0f}s{marker}")

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

print(f"\nBest validation accuracy: {best_val_acc:.4f}")

# ============================================================
# EVALUATION ON TEST SET
# ============================================================
print("\n" + "=" * 80)
print("EVALUATING ON TEST SET")
print("=" * 80)

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=DEVICE, weights_only=True))
model.eval()

all_preds, all_labels_arr, all_probs = [], [], []
with torch.no_grad():
    for images, lbl in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels_arr.extend(lbl.numpy())
        all_probs.extend(probs)

all_preds = np.array(all_preds)
all_labels_arr = np.array(all_labels_arr)
all_probs = np.array(all_probs)

# Metrics
acc = accuracy_score(all_labels_arr, all_preds)
macro_f1 = f1_score(all_labels_arr, all_preds, average='macro', zero_division=0)
weighted_f1 = f1_score(all_labels_arr, all_preds, average='weighted', zero_division=0)
macro_prec = precision_score(all_labels_arr, all_preds, average='macro', zero_division=0)
macro_rec = recall_score(all_labels_arr, all_preds, average='macro', zero_division=0)
top3_acc = top_k_accuracy_score(all_labels_arr, all_probs, k=3, labels=range(num_classes))
top5_acc = top_k_accuracy_score(all_labels_arr, all_probs, k=5, labels=range(num_classes))

print(f"\n{'Metric':<30} {'Value':>10}")
print("-" * 42)
for name, val in [("Accuracy", acc), ("Top-3 Accuracy", top3_acc), ("Top-5 Accuracy", top5_acc),
                   ("Macro F1", macro_f1), ("Weighted F1", weighted_f1),
                   ("Macro Precision", macro_prec), ("Macro Recall", macro_rec)]:
    print(f"{name:<30} {val:>10.4f}")

present_labels = sorted(set(all_labels_arr) | set(all_preds))
present_names = [idx2cat[i] for i in present_labels]

report = classification_report(all_labels_arr, all_preds, labels=present_labels,
                                target_names=present_names, zero_division=0, output_dict=True)
report_str = classification_report(all_labels_arr, all_preds, labels=present_labels,
                                    target_names=present_names, zero_division=0)
print("\n" + report_str)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
    f.write(f"Accuracy: {acc:.4f}\nTop-3: {top3_acc:.4f}\nTop-5: {top5_acc:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\nWeighted F1: {weighted_f1:.4f}\n\n{report_str}")

# ============================================================
# PLOTS
# ============================================================

# 1. Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='Train'); ax1.plot(history['val_loss'], label='Val')
ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
ax2.plot(history['train_acc'], label='Train'); ax2.plot(history['val_acc'], label='Val')
ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150); plt.close()

# 2. Confusion matrix
cm = confusion_matrix(all_labels_arr, all_preds, labels=present_labels)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
fig, ax = plt.subplots(figsize=(22, 18))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=present_names, yticklabels=present_names, ax=ax, vmin=0, vmax=1)
ax.set_title('Normalized Confusion Matrix — Option B Improved')
ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150); plt.close()

# 3. Per-class F1
per_class_f1 = [(cat, report[cat]['f1-score'], report[cat]['support']) for cat in present_names]
per_class_f1.sort(key=lambda x: -x[1])
fig, ax = plt.subplots(figsize=(14, 10))
names_p = [x[0] for x in per_class_f1]
f1s = [x[1] for x in per_class_f1]
sups = [x[2] for x in per_class_f1]
colors = ['green' if f > 0.5 else 'orange' if f > 0.2 else 'red' for f in f1s]
ax.barh(range(len(names_p)), f1s, color=colors)
ax.set_yticks(range(len(names_p)))
ax.set_yticklabels([f"{n} (n={s})" for n, s in zip(names_p, sups)], fontsize=8)
ax.set_xlabel('F1 Score'); ax.set_title('Per-Class F1 — Option B Improved')
ax.invert_yaxis(); ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "per_class_f1.png"), dpi=150); plt.close()

# 4. Confused pairs
print(f"\n{'='*80}")
print("TOP 15 MOST CONFUSED PAIRS")
print(f"{'='*80}")
full_cm = confusion_matrix(all_labels_arr, all_preds, labels=range(num_classes))
confused = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and full_cm[i][j] > 0:
            confused.append((idx2cat[i], idx2cat[j], full_cm[i][j]))
confused.sort(key=lambda x: -x[2])
print(f"{'True':<35} {'Predicted As':<35} {'Count':>5}")
print("-" * 77)
for t, p, c in confused[:15]:
    print(f"{t:<35} {p:<35} {c:>5}")

# 5. Zero F1
zero_f1 = [(n, s) for n, f, s in per_class_f1 if f == 0]
print(f"\nCategories with F1=0: {len(zero_f1)}")
for n, s in zero_f1: print(f"  {n} (test: {s})")

# 6. Size buckets
print(f"\n{'='*80}")
print("PERFORMANCE BY CATEGORY SIZE")
print(f"{'='*80}")
test_counts = Counter(all_labels_arr)
buckets = {'<10': [], '10-49': [], '50-99': [], '100-299': [], '300+': []}
for cat in present_names:
    idx = cat2idx[cat]
    support = test_counts.get(idx, 0)
    f1 = report[cat]['f1-score']
    if support < 10: buckets['<10'].append(f1)
    elif support < 50: buckets['10-49'].append(f1)
    elif support < 100: buckets['50-99'].append(f1)
    elif support < 300: buckets['100-299'].append(f1)
    else: buckets['300+'].append(f1)

print(f"{'Bucket':<15} {'#Classes':>10} {'Avg F1':>10} {'Min F1':>10} {'Max F1':>10}")
print("-" * 57)
for bucket, f1s_list in buckets.items():
    if f1s_list:
        print(f"{bucket:<15} {len(f1s_list):>10} {np.mean(f1s_list):>10.4f} {min(f1s_list):>10.4f} {max(f1s_list):>10.4f}")

# ============================================================
# 3-WAY COMPARISON
# ============================================================
print(f"\n{'='*80}")
print("COMPARISON: BASELINE vs OPTION B vs OPTION B IMPROVED")
print(f"{'='*80}")
print(f"{'Metric':<25} {'Baseline':>12} {'Option B':>12} {'B Improved':>12} {'B→Imp Δ':>10}")
print("-" * 73)
baseline = {'Accuracy': 0.2930, 'Top-3 Acc': 0.4821, 'Top-5 Acc': 0.5818,
            'Macro F1': 0.3224, 'Weighted F1': 0.2730}
optB = {'Accuracy': 0.5885, 'Top-3 Acc': 0.8534, 'Top-5 Acc': 0.9290,
        'Macro F1': 0.5730, 'Weighted F1': 0.5777}
improved = {'Accuracy': acc, 'Top-3 Acc': top3_acc, 'Top-5 Acc': top5_acc,
            'Macro F1': macro_f1, 'Weighted F1': weighted_f1}
for metric in baseline:
    b, o, im = baseline[metric], optB[metric], improved[metric]
    delta = im - o
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
    print(f"{metric:<25} {b:>12.4f} {o:>12.4f} {im:>12.4f} {arrow}{abs(delta):>8.4f}")

print(f"{'Num Classes':<25} {'122':>12} {'28':>12} {'28':>12}")
print(f"\nAll results saved to: {OUTPUT_DIR}")
