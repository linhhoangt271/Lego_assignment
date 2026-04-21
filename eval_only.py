"""Evaluate the saved baseline model — no retraining needed."""
import json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

BASE_DIR = "/Users/linh/Downloads/KUL /11. Advanced Analytics/big_data_assignment_2"
OUTPUT_DIR = os.path.join(BASE_DIR, "baseline_results")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 128; BATCH_SIZE = 64

# Load data
with open(os.path.join(BASE_DIR, "minifigs.json")) as f:
    data = json.load(f)

valid_data = [d for d in data if d.get('img_local_path') and
              os.path.exists(os.path.join(BASE_DIR, d['img_local_path']))]

categories = [d['category'] for d in valid_data]
cat_counts = Counter(categories)
cat_names = sorted(cat_counts.keys())
cat2idx = {c: i for i, c in enumerate(cat_names)}
idx2cat = {i: c for c, i in cat2idx.items()}
num_classes = len(cat_names)

# Reproduce same split
labels = [cat2idx[d['category']] for d in valid_data]
label_counts = Counter(labels)
small_idx = [i for i, l in enumerate(labels) if label_counts[l] < 7]
big_idx = [i for i, l in enumerate(labels) if label_counts[l] >= 7]
big_data = [valid_data[i] for i in big_idx]
big_labels = [labels[i] for i in big_idx]
_, temp_data, _, temp_labels = train_test_split(big_data, big_labels, test_size=0.3, random_state=42, stratify=big_labels)
_, test_data, _, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
print(f"Test set: {len(test_data)} samples, {num_classes} total classes")

class MinifigDataset(Dataset):
    def __init__(self, records, labels, transform):
        self.records, self.labels, self.transform = records, labels, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        try: img = Image.open(os.path.join(BASE_DIR, self.records[idx]['img_local_path'])).convert('RGB')
        except: img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128,128,128))
        return self.transform(img), self.labels[idx]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_loader = DataLoader(MinifigDataset(test_data, test_labels, eval_transform), batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, num_classes))
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"), map_location=DEVICE, weights_only=True))
model = model.to(DEVICE)
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

# Overall metrics
acc = accuracy_score(all_labels_arr, all_preds)
macro_f1 = f1_score(all_labels_arr, all_preds, average='macro', zero_division=0)
weighted_f1 = f1_score(all_labels_arr, all_preds, average='weighted', zero_division=0)
macro_prec = precision_score(all_labels_arr, all_preds, average='macro', zero_division=0)
macro_rec = recall_score(all_labels_arr, all_preds, average='macro', zero_division=0)
top3_acc = top_k_accuracy_score(all_labels_arr, all_probs, k=3, labels=range(num_classes))
top5_acc = top_k_accuracy_score(all_labels_arr, all_probs, k=5, labels=range(num_classes))

print(f"\n{'='*65}")
print("OVERALL METRICS")
print(f"{'='*65}")
print(f"{'Metric':<30} {'Value':>10}")
print("-" * 42)
for name, val in [("Accuracy", acc), ("Top-3 Accuracy", top3_acc), ("Top-5 Accuracy", top5_acc),
                   ("Macro F1", macro_f1), ("Weighted F1", weighted_f1),
                   ("Macro Precision", macro_prec), ("Macro Recall", macro_rec)]:
    print(f"{name:<30} {val:>10.4f}")

# Per-class report
present_labels = sorted(set(all_labels_arr) | set(all_preds))
present_names = [idx2cat[i] for i in present_labels]

report = classification_report(all_labels_arr, all_preds, labels=present_labels,
                                target_names=present_names, zero_division=0, output_dict=True)
report_str = classification_report(all_labels_arr, all_preds, labels=present_labels,
                                    target_names=present_names, zero_division=0)
print("\n" + report_str)

# Save report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
    f.write(f"Accuracy: {acc:.4f}\nTop-3: {top3_acc:.4f}\nTop-5: {top5_acc:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\nWeighted F1: {weighted_f1:.4f}\n\n{report_str}")

# Per-class F1
per_class_f1 = [(cat, report[cat]['f1-score'], report[cat]['support']) for cat in present_names]
per_class_f1.sort(key=lambda x: -x[1])

# Plot 1: Per-class F1 bar chart
fig, ax = plt.subplots(figsize=(16, 22))
names_plot = [x[0] for x in per_class_f1]
f1s = [x[1] for x in per_class_f1]
supports = [x[2] for x in per_class_f1]
colors = ['green' if f > 0.5 else 'orange' if f > 0.2 else 'red' for f in f1s]
ax.barh(range(len(names_plot)), f1s, color=colors)
ax.set_yticks(range(len(names_plot)))
ax.set_yticklabels([f"{n} (n={s})" for n, s in zip(names_plot, supports)], fontsize=6)
ax.set_xlabel('F1 Score'); ax.set_title('Per-Class F1 (green>0.5, orange>0.2, red≤0.2)')
ax.invert_yaxis(); ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "per_class_f1.png"), dpi=150); plt.close()

# Plot 2: Confusion matrix (top 20)
top20_cats = [c for c, _ in Counter(all_labels_arr).most_common(20)]
mask = np.isin(all_labels_arr, top20_cats)
cm = confusion_matrix(all_labels_arr[mask], all_preds[mask], labels=top20_cats)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
top20_names = [idx2cat[i] for i in top20_cats]

fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=top20_names, yticklabels=top20_names, ax=ax, vmin=0, vmax=1)
ax.set_title('Normalized Confusion Matrix (Top 20)'); ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_top20.png"), dpi=150); plt.close()

# Most confused pairs
print(f"\n{'='*65}")
print("TOP 20 MOST CONFUSED CATEGORY PAIRS")
print(f"{'='*65}")
full_cm = confusion_matrix(all_labels_arr, all_preds, labels=range(num_classes))
confused = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and full_cm[i][j] > 0:
            confused.append((idx2cat[i], idx2cat[j], full_cm[i][j]))
confused.sort(key=lambda x: -x[2])
print(f"{'True':<40} {'Predicted As':<40} {'Count':>5}")
print("-" * 87)
for t, p, c in confused[:20]:
    print(f"{t:<40} {p:<40} {c:>5}")

# Categories with F1=0
zero_f1 = [(n, s) for n, f, s in per_class_f1 if f == 0]
print(f"\n{'='*65}")
print(f"CATEGORIES WITH F1=0 ({len(zero_f1)} total)")
print(f"{'='*65}")
for n, s in zero_f1:
    print(f"  {n} (test samples: {s})")

# Performance by size
print(f"\n{'='*65}")
print("PERFORMANCE BY CATEGORY SIZE")
print(f"{'='*65}")
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

print(f"\nAll results saved to: {OUTPUT_DIR}")
