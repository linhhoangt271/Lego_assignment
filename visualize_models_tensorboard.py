"""
TensorBoard Visualization: All Trained Models Comparison
- Embedding scatterplots (UMAP/TSNE)
- Model predictions vs ground truth
- Confusion matrices
- Performance metrics
- 3D projector visualization
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  UMAP not available, will use PCA instead")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 380
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = os.path.join(BASE_DIR, "runs", "all_models_comparison")
os.makedirs(LOG_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"TensorBoard logs: {LOG_DIR}")

# ============================================================
# LOAD METADATA & TRANSFORMS
# ============================================================
with open(os.path.join(BASE_DIR, "minifigs.json")) as f:
    minifigs_list = json.load(f)

valid_images = set()
theme_map = {}
theme_path_map = {}

for minifig in minifigs_list:
    img_path = minifig.get("img_local_path")
    theme = minifig.get("themes", [None])[0]
    if img_path and theme and os.path.exists(img_path):
        valid_images.add(img_path)
        theme_map[img_path] = theme
        theme_path_map[img_path] = img_path

MERGE_MAP = {
    'The LEGO NINJAGO Movie': 'NINJAGO', 'The LEGO Movie': 'LEGO Movies',
    'The LEGO Movie 2': 'LEGO Movies', 'DC Super Hero Girls': 'Super Heroes',
    'Spider-Man': 'Super Heroes', 'Batman I': 'Super Heroes',
    'Elves': 'Friends & Fantasy', 'Friends': 'Friends & Fantasy',
    'DUPLO': 'Preschool', 'Primo': 'Preschool', 'Belville': 'Preschool', 'Scala': 'Preschool',
}

def map_theme(theme):
    return MERGE_MAP.get(theme, theme)

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# DATASET
# ============================================================
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

# ============================================================
# LOAD TEST DATA
# ============================================================
print("\nLoading test data...")
test_data = list(valid_images)[:5000]  # Use 5000 samples for visualization
test_labels = [map_theme(theme_map[img]) for img in test_data]

class_names = sorted(list(set(test_labels)))
class_to_idx = {c: i for i, c in enumerate(class_names)}
test_labels_idx = [class_to_idx[c] for c in test_labels]

print(f"Test samples: {len(test_data)}")
print(f"Classes: {len(class_names)}")

test_dataset = MinifigDataset(test_data, test_labels_idx, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================
# MODELS TO EVALUATE
# ============================================================
MODELS = {
    'Baseline': {
        'path': 'baseline_results/best_model.pth',
        'type': 'resnet50',
        'num_classes': len(class_names)
    },
    'OptionB': {
        'path': 'optionB_results/best_model.pth',
        'type': 'effnet_b4',
        'num_classes': len(class_names)
    },
    'V2': {
        'path': 'optionB_v2_results/best_model.pth',
        'type': 'effnet_b4',
        'num_classes': len(class_names)
    },
    'V3-Focal': {
        'path': 'optionB_v3_results/best_model_focal.pth',
        'type': 'effnet_b4',
        'num_classes': len(class_names)
    },
    'V3-ArcFace': {
        'path': 'optionB_v3_results/best_model_arcface.pth',
        'type': 'effnet_b4',
        'num_classes': len(class_names)
    },
    'V3-SupCon': {
        'path': 'optionB_v3_results/best_model_supcon.pth',
        'type': 'effnet_b4',
        'num_classes': len(class_names)
    },
}

# ============================================================
# MODEL LOADING
# ============================================================
def load_model(model_name, model_info):
    model_path = os.path.join(BASE_DIR, model_info['path'])
    if not os.path.exists(model_path):
        print(f"  ⚠️  {model_name} not found: {model_path}")
        return None

    try:
        # Load model directly - handles both full models and state dicts
        checkpoint = torch.load(model_path, map_location=DEVICE)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            # Try common architectures
            try:
                model = models.efficientnet_b4(pretrained=False)
                model.load_state_dict(checkpoint, strict=False)
            except:
                try:
                    model = models.resnet50(pretrained=False)
                    model.load_state_dict(checkpoint, strict=False)
                except:
                    print(f"  ⚠️  Could not match architecture, skipping {model_name}")
                    return None

        model.to(DEVICE)
        model.eval()
        print(f"  ✅ {model_name} loaded")
        return model
    except Exception as e:
        print(f"  ❌ Error loading {model_name}: {e}")
        return None

# ============================================================
# EXTRACT EMBEDDINGS & PREDICTIONS
# ============================================================
def get_embeddings_and_predictions(model, model_name):
    print(f"\n  Extracting embeddings for {model_name}...")

    embeddings = []
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(DEVICE)

            # Get embeddings (before final FC layer)
            if isinstance(model, torch.nn.Sequential):
                # For V3 models with Sequential wrapper
                feat = model[0](images) if len(model) > 1 else model(images)
            else:
                feat = model(images) if hasattr(model, '__call__') else model.fc(images)

            # Get predictions
            with torch.no_grad():
                logits = model(images)

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            embeddings.append(feat.cpu().numpy())
            predictions.extend(preds)
            ground_truth.extend(labels.cpu().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings, np.array(predictions), np.array(ground_truth)

# ============================================================
# DIMENSIONALITY REDUCTION
# ============================================================
def reduce_embeddings(embeddings, method='umap'):
    print(f"  Reducing dimensions with {method}...")

    if method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, metric='cosine', n_neighbors=15, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    else:
        # Fallback to PCA
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)

    return reduced

# ============================================================
# WRITER & VISUALIZATION
# ============================================================
writer = SummaryWriter(LOG_DIR)

print("\n" + "="*80)
print("LOADING TRAINED MODELS")
print("="*80)

model_results = {}

for model_name, model_info in MODELS.items():
    print(f"\n{model_name}:")
    model = load_model(model_name, model_info)

    if model is None:
        continue

    embeddings, preds, ground_truth = get_embeddings_and_predictions(model, model_name)

    # Compute metrics
    accuracy = accuracy_score(ground_truth, preds)
    macro_f1 = f1_score(ground_truth, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(ground_truth, preds, average='weighted', zero_division=0)

    model_results[model_name] = {
        'embeddings': embeddings,
        'predictions': preds,
        'ground_truth': ground_truth,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }

    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Macro F1: {macro_f1:.4f}")
    print(f"    Weighted F1: {weighted_f1:.4f}")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

for model_name, results in model_results.items():
    print(f"\n{model_name}:")

    # Reduce dimensions
    reduced = reduce_embeddings(results['embeddings'])

    # Create scatterplot
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter_colors = results['ground_truth']
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=scatter_colors,
                        cmap='tab20c', alpha=0.6, s=20, edgecolors='none')

    ax.set_xlabel('Dimension 1', fontsize=10)
    ax.set_ylabel('Dimension 2', fontsize=10)
    ax.set_title(f'{model_name} - Embedding Space Visualization\n' +
                f'Accuracy: {results["accuracy"]:.4f} | Macro F1: {results["macro_f1"]:.4f}',
                fontsize=12, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Class Index', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    writer.add_figure(f'01_EmbeddingScatter/{model_name}', fig, global_step=0)
    plt.close()
    print(f"  ✅ Scatterplot saved")

    # Add metrics as text
    metrics_text = f"""
    **Accuracy**: {results['accuracy']:.4f}
    **Macro F1**: {results['macro_f1']:.4f}
    **Weighted F1**: {results['weighted_f1']:.4f}
    """
    writer.add_text(f'02_Metrics/{model_name}', metrics_text)

# ============================================================
# COMPARISON TABLE
# ============================================================
print("\n3. Creating comparison table...")
fig, ax = plt.subplots(figsize=(10, 6))

model_names_list = list(model_results.keys())
accuracies = [model_results[m]['accuracy'] for m in model_names_list]
macro_f1s = [model_results[m]['macro_f1'] for m in model_names_list]
weighted_f1s = [model_results[m]['weighted_f1'] for m in model_names_list]

x = np.arange(len(model_names_list))
width = 0.25

bars1 = ax.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
bars2 = ax.bar(x, macro_f1s, width, label='Macro F1', alpha=0.8)
bars3 = ax.bar(x + width, weighted_f1s, width, label='Weighted F1', alpha=0.8)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names_list, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
writer.add_figure('03_Comparison/AllModels', fig, global_step=0)
plt.close()
print(f"  ✅ Comparison chart saved")

# ============================================================
# CONFUSION MATRICES
# ============================================================
print("\n4. Creating confusion matrices...")
for model_name, results in model_results.items():
    cm = confusion_matrix(results['ground_truth'], results['predictions'])

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_normalized, cmap='Blues', ax=ax, cbar_kws={'label': 'Proportion'},
                xticklabels=False, yticklabels=False)
    ax.set_xlabel('Predicted Class', fontsize=10)
    ax.set_ylabel('True Class', fontsize=10)
    ax.set_title(f'{model_name} - Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    writer.add_figure(f'04_ConfusionMatrix/{model_name}', fig, global_step=0)
    plt.close()
    print(f"  ✅ Confusion matrix saved for {model_name}")

# ============================================================
# RANKING TABLE
# ============================================================
print("\n5. Creating ranking table...")
ranking_data = {
    'Model': [],
    'Accuracy': [],
    'Macro F1': [],
    'Weighted F1': []
}

for model_name in sorted(model_results.keys(),
                         key=lambda x: model_results[x]['accuracy'],
                         reverse=True):
    ranking_data['Model'].append(model_name)
    ranking_data['Accuracy'].append(f"{model_results[model_name]['accuracy']:.4f}")
    ranking_data['Macro F1'].append(f"{model_results[model_name]['macro_f1']:.4f}")
    ranking_data['Weighted F1'].append(f"{model_results[model_name]['weighted_f1']:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=[[ranking_data['Model'][i],
                           ranking_data['Accuracy'][i],
                           ranking_data['Macro F1'][i],
                           ranking_data['Weighted F1'][i]]
                          for i in range(len(ranking_data['Model']))],
                colLabels=['Model', 'Accuracy', 'Macro F1', 'Weighted F1'],
                cellLoc='center',
                loc='center',
                colColours=['#f0f0f0']*4)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Color code best performance
for i in range(1, len(ranking_data['Model']) + 1):
    table[(i, 1)].set_facecolor('#90EE90' if i == 1 else '#FFFFFF')

ax.set_title('Model Performance Ranking', fontsize=13, fontweight='bold', pad=20)
writer.add_figure('05_Ranking/PerformanceTable', fig, global_step=0)
plt.close()
print(f"  ✅ Ranking table saved")

# ============================================================
# SUMMARY
# ============================================================
writer.add_text('00_Summary/BestModel',
               f"**Best Model**: {max(model_results.items(), key=lambda x: x[1]['accuracy'])[0]}\n" +
               f"**Accuracy**: {max(model_results.items(), key=lambda x: x[1]['accuracy'])[1]['accuracy']:.4f}")

print("\n" + "="*80)
print("✅ ALL MODELS VISUALIZATION COMPLETE!")
print("="*80)
print(f"\n📊 Logs saved to: {LOG_DIR}")
print(f"\n🎯 To view results, run:")
print(f"   tensorboard --logdir {LOG_DIR}")
print(f"\n📈 Then open: http://localhost:6006")
print("\n" + "="*80)

writer.close()

# ============================================================
# SAVE SUMMARY REPORT
# ============================================================
summary_file = os.path.join(LOG_DIR, "model_comparison_summary.txt")
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MODEL PERFORMANCE COMPARISON SUMMARY\n")
    f.write("="*80 + "\n\n")

    for model_name in sorted(model_results.keys(),
                            key=lambda x: model_results[x]['accuracy'],
                            reverse=True):
        r = model_results[model_name]
        f.write(f"\n{model_name}:\n")
        f.write(f"  Accuracy:    {r['accuracy']:.4f}\n")
        f.write(f"  Macro F1:    {r['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1: {r['weighted_f1']:.4f}\n")

print(f"\n📄 Summary report saved to: {summary_file}")
