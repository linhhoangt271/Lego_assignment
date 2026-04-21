"""
Fast TensorBoard Visualization: All Trained Models Comparison
- Performance metrics comparison
- Model rankings
- Confusion matrices (sample)
- Embedding scatterplots (smaller sample size for speed)
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 380
BATCH_SIZE = 32  # Reduced batch size
DEVICE = torch.device("cpu")  # Use CPU to avoid GPU memory issues with HP tuning
LOG_DIR = os.path.join(BASE_DIR, "runs", "all_models_comparison")
TEST_SAMPLES = 500  # Use 500 samples for speed
os.makedirs(LOG_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Test samples: {TEST_SAMPLES}")
print(f"TensorBoard logs: {LOG_DIR}")

# ============================================================
# LOAD DATA
# ============================================================
with open(os.path.join(BASE_DIR, "minifigs.json")) as f:
    minifigs_list = json.load(f)

valid_images = []
theme_map = {}
theme_path_map = {}

for minifig in minifigs_list:
    img_path = minifig.get("img_local_path")
    theme = minifig.get("themes", [None])[0]
    if img_path and theme and os.path.exists(img_path):
        valid_images.append(img_path)
        theme_map[img_path] = theme
        theme_path_map[img_path] = img_path

# Subsample for speed
valid_images = valid_images[:TEST_SAMPLES]

MERGE_MAP = {
    'The LEGO NINJAGO Movie': 'NINJAGO', 'The LEGO Movie': 'LEGO Movies',
    'The LEGO Movie 2': 'LEGO Movies', 'DC Super Hero Girls': 'Super Heroes',
    'Spider-Man': 'Super Heroes', 'Batman I': 'Super Heroes',
    'Elves': 'Friends & Fantasy', 'Friends': 'Friends & Fantasy',
}

def map_theme(theme):
    return MERGE_MAP.get(theme, theme)

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MinifigDataset(Dataset):
    def __init__(self, image_list, labels, transform=None):
        self.image_list = image_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = theme_path_map[img_name]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, img_name

# Prepare data
test_labels = [map_theme(theme_map[img]) for img in valid_images]
class_names = sorted(list(set(test_labels)))
class_to_idx = {c: i for i, c in enumerate(class_names)}
test_labels_idx = [class_to_idx[c] for c in test_labels]

test_dataset = MinifigDataset(valid_images, test_labels_idx, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Test set: {len(valid_images)} samples, {len(class_names)} classes")

# ============================================================
# MODELS
# ============================================================
MODELS = {
    'Baseline': 'baseline_results/best_model.pth',
    'OptionB': 'optionB_results/best_model.pth',
    'V2': 'optionB_v2_results/best_model.pth',
    'V3-Focal': 'optionB_v3_results/best_model_focal.pth',
    'V3-ArcFace': 'optionB_v3_results/best_model_arcface.pth',
    'V3-SupCon': 'optionB_v3_results/best_model_supcon.pth',
}

def load_model(model_path):
    """Load model with flexible architecture detection"""
    if not os.path.exists(model_path):
        return None

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            # Try common architectures
            for arch_fn in [
                lambda: models.efficientnet_b4(pretrained=False),
                lambda: models.resnet50(pretrained=False),
            ]:
                try:
                    model = arch_fn()
                    model.load_state_dict(checkpoint, strict=False)
                    break
                except:
                    continue
            else:
                return None

        model.to(DEVICE)
        model.eval()
        return model
    except:
        return None

# ============================================================
# EVALUATE MODELS
# ============================================================
print("\n" + "="*80)
print("EVALUATING MODELS")
print("="*80)

writer = SummaryWriter(LOG_DIR)
results = {}

for model_name, model_path in MODELS.items():
    full_path = os.path.join(BASE_DIR, model_path)
    print(f"\n{model_name}:", end=" ")

    model = load_model(full_path)
    if model is None:
        print("❌ Failed to load")
        continue

    print("✅ Loaded", end="")

    # Get predictions
    all_preds = []
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(test_loader):
            images = images.to(DEVICE)

            # Get embeddings (average pool features)
            if hasattr(model, 'avgpool'):  # ResNet
                feat = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(images))))))))
                feat = torch.flatten(feat, 1)
            elif hasattr(model, 'classifier'):  # EfficientNet
                feat = model.avgpool(model.features(images))
                feat = torch.flatten(feat, 1)
            else:
                feat = images  # Fallback

            all_embeddings.append(feat.cpu().numpy())

            # Get predictions
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 5 == 0:
                print(".", end="", flush=True)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_embeddings = np.vstack(all_embeddings) if all_embeddings else None

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    results[model_name] = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'predictions': all_preds,
        'labels': all_labels,
        'embeddings': all_embeddings
    }

    print(f" | Acc: {accuracy:.4f} | F1: {macro_f1:.4f}")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Comparison Bar Chart
print("\n1. Creating comparison chart...")
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
macro_f1s = [results[m]['macro_f1'] for m in model_names]
weighted_f1s = [results[m]['weighted_f1'] for m in model_names]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(model_names))
width = 0.25

bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', alpha=0.85, color='#1f77b4')
bars2 = ax.bar(x, macro_f1s, width, label='Macro F1', alpha=0.85, color='#ff7f0e')
bars3 = ax.bar(x + width, weighted_f1s, width, label='Weighted F1', alpha=0.85, color='#2ca02c')

ax.set_ylabel('Score', fontsize=11, fontweight='bold')
ax.set_title('All Models Performance Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
writer.add_figure('01_Comparison/PerformanceChart', fig, global_step=0)
plt.close()
print("  ✅ Comparison chart saved")

# 2. Ranking Table
print("2. Creating ranking table...")
sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

fig, ax = plt.subplots(figsize=(10, len(sorted_models) * 0.4 + 1))
ax.axis('tight')
ax.axis('off')

table_data = [[m, f"{v['accuracy']:.4f}", f"{v['macro_f1']:.4f}", f"{v['weighted_f1']:.4f}"]
              for m, v in sorted_models]

table = ax.table(cellText=table_data,
                colLabels=['Model', 'Accuracy', 'Macro F1', 'Weighted F1'],
                cellLoc='center',
                loc='center',
                colColours=['#f0f0f0']*4)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Color best performance
for i in range(1, len(sorted_models) + 1):
    table[(i, 1)].set_facecolor('#90EE90' if i == 1 else '#ffffff')

ax.set_title('Model Performance Ranking (Sorted by Accuracy)', fontsize=12, fontweight='bold', pad=15)
writer.add_figure('02_Ranking/Table', fig, global_step=0)
plt.close()
print("  ✅ Ranking table saved")

# 3. Confusion Matrices (sample)
print("3. Creating confusion matrices...")
for model_name in model_names[:3]:  # Top 3 for speed
    cm = confusion_matrix(results[model_name]['labels'], results[model_name]['predictions'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(cm_norm, cmap='Blues', ax=ax, cbar_kws={'label': 'Proportion'},
                xticklabels=False, yticklabels=False, vmin=0, vmax=1)
    ax.set_xlabel('Predicted Class', fontsize=10)
    ax.set_ylabel('True Class', fontsize=10)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=11, fontweight='bold')

    writer.add_figure(f'03_ConfusionMatrix/{model_name}', fig, global_step=0)
    plt.close()
    print(f"  ✅ Confusion matrix: {model_name}")

# 4. Embedding Scatterplots (PCA 2D)
print("4. Creating embedding scatterplots...")
for model_name in model_names:
    if results[model_name]['embeddings'] is None:
        continue

    embeddings = results[model_name]['embeddings']

    # PCA reduction
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(11, 9))

    labels = results[model_name]['labels']
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels,
                        cmap='tab20c', alpha=0.5, s=15, edgecolors='none')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
    ax.set_title(f'{model_name} - Embedding Space (PCA)\nAcc: {results[model_name]["accuracy"]:.4f}',
                fontsize=11, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Class', fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    writer.add_figure(f'04_Embeddings/{model_name}', fig, global_step=0)
    plt.close()
    print(f"  ✅ Embedding plot: {model_name}")

# 5. Summary Text
print("5. Creating summary...")
best_model = sorted_models[0]
summary_text = f"""
# Model Comparison Summary

## Best Model: {best_model[0]}
- **Accuracy**: {best_model[1]['accuracy']:.4f}
- **Macro F1**: {best_model[1]['macro_f1']:.4f}
- **Weighted F1**: {best_model[1]['weighted_f1']:.4f}

## Top 3 Models:
"""

for i, (name, metrics) in enumerate(sorted_models[:3], 1):
    summary_text += f"\n{i}. **{name}** - Acc: {metrics['accuracy']:.4f}, F1: {metrics['macro_f1']:.4f}"

writer.add_text('00_Summary/Overview', summary_text)

print("  ✅ Summary created")

# ============================================================
# CLOSE
# ============================================================
writer.close()

print("\n" + "="*80)
print("✅ ALL VISUALIZATIONS COMPLETE!")
print("="*80)
print(f"\n📊 Logs: {LOG_DIR}")
print(f"📈 View: tensorboard --logdir {LOG_DIR}")
print(f"🌐 Open: http://localhost:6006")
print("\n" + "="*80)

# Save summary report
summary_file = os.path.join(LOG_DIR, "model_comparison_summary.txt")
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MODEL PERFORMANCE SUMMARY\n")
    f.write("="*80 + "\n\n")
    for name, metrics in sorted_models:
        f.write(f"{name}:\n")
        f.write(f"  Accuracy:    {metrics['accuracy']:.4f}\n")
        f.write(f"  Macro F1:    {metrics['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1: {metrics['weighted_f1']:.4f}\n\n")

print(f"📄 Report saved to: {summary_file}\n")
