"""Evaluate all 4 trained models on the test set (no retraining)."""
import json, os, warnings, math
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, top_k_accuracy_score
)
from collections import Counter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "/home/test/big_data_assignment_2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
NUM_WORKERS = 2

# ============================================================
# MERGE MAPS
# ============================================================
MERGE_MAP_B = {
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

def split_town(subcategory):
    sub = subcategory.lower()
    if 'police' in sub: return 'Town - Police'
    if 'fire' in sub: return 'Town - Fire'
    if 'airport' in sub: return 'Town - Airport'
    if 'hospital' in sub or 'rescue' in sub: return 'Town - Rescue'
    if 'space' in sub: return 'Town - Space'
    if 'race' in sub or 'stuntz' in sub: return 'Town - Racing'
    if 'coast guard' in sub: return 'Town - Coast Guard'
    if 'construction' in sub: return 'Town - Construction'
    if any(x in sub for x in ['arctic', 'jungle', 'volcano', 'ocean', 'deep sea', 'exploration']): return 'Town - Exploration'
    return 'Town - General'

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
with open(os.path.join(BASE_DIR, "minifigs.json")) as f:
    data = json.load(f)

valid_data = [d for d in data if d.get('img_local_path') and
              os.path.exists(os.path.join(BASE_DIR, d['img_local_path']))]
print(f"Valid images: {len(valid_data)}")

# ============================================================
# HELPER: build labels + split for a given merge strategy
# ============================================================
def build_split(valid_data, merge_map=None, town_split=False):
    """Apply merge, build cat2idx, reproduce the same stratified split."""
    import copy
    vd = copy.deepcopy(valid_data)

    for d in vd:
        if town_split and d['category'] == 'Town':
            d['merged_category'] = split_town(d['subcategory'])
        elif merge_map:
            d['merged_category'] = merge_map.get(d['category'], d['category'])
        else:
            d['merged_category'] = d['category']

    cat_key = 'merged_category'
    categories = [d[cat_key] for d in vd]
    cat_names = sorted(set(categories))
    cat2idx = {c: i for i, c in enumerate(cat_names)}
    idx2cat = {i: c for c, i in cat2idx.items()}
    num_classes = len(cat_names)

    labels = [cat2idx[d[cat_key]] for d in vd]
    label_counts = Counter(labels)
    small_idx = [i for i, l in enumerate(labels) if label_counts[l] < 7]
    big_idx = [i for i, l in enumerate(labels) if label_counts[l] >= 7]

    big_data = [vd[i] for i in big_idx]
    big_labels = [labels[i] for i in big_idx]

    _, temp_data, _, temp_labels = train_test_split(
        big_data, big_labels, test_size=0.3, random_state=42, stratify=big_labels)
    _, test_data, _, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

    return test_data, test_labels, cat2idx, idx2cat, num_classes

# ============================================================
# DATASET
# ============================================================
class MinifigDataset(Dataset):
    def __init__(self, records, labels, transform, img_size):
        self.records, self.labels, self.transform, self.img_size = records, labels, transform, img_size
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        try: img = Image.open(os.path.join(BASE_DIR, self.records[idx]['img_local_path'])).convert('RGB')
        except: img = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))
        return self.transform(img), self.labels[idx]

class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        return ImageOps.pad(img, (max_side, max_side), color=(255, 255, 255))

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model(model, test_loader, num_classes, idx2cat):
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

    acc = accuracy_score(all_labels_arr, all_preds)
    macro_f1 = f1_score(all_labels_arr, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels_arr, all_preds, average='weighted', zero_division=0)
    macro_prec = precision_score(all_labels_arr, all_preds, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels_arr, all_preds, average='macro', zero_division=0)
    top3_acc = top_k_accuracy_score(all_labels_arr, all_probs, k=3, labels=range(num_classes))
    top5_acc = top_k_accuracy_score(all_labels_arr, all_probs, k=5, labels=range(num_classes))

    return {
        'Accuracy': acc, 'Top-3 Accuracy': top3_acc, 'Top-5 Accuracy': top5_acc,
        'Macro F1': macro_f1, 'Weighted F1': weighted_f1,
        'Macro Precision': macro_prec, 'Macro Recall': macro_rec
    }, all_preds, all_labels_arr, all_probs

def evaluate_with_tta(model, test_data, test_labels, num_classes, idx2cat, img_size, batch_size):
    """Run TTA evaluation for V2 model."""
    tta_transforms_list = [
        transforms.Compose([PadToSquare(), transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([PadToSquare(), transforms.Resize((img_size, img_size)),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([PadToSquare(), transforms.Resize((img_size + 20, img_size + 20)),
                            transforms.CenterCrop(img_size), transforms.RandomRotation(10),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([PadToSquare(), transforms.Resize((img_size + 40, img_size + 40)),
                            transforms.CenterCrop(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        transforms.Compose([PadToSquare(), transforms.Resize((img_size, img_size)),
                            transforms.ColorJitter(brightness=0.15, contrast=0.15),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    ]

    model.eval()
    # Get labels from first pass
    first_ds = MinifigDataset(test_data, test_labels, tta_transforms_list[0], img_size)
    first_loader = DataLoader(first_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    all_labels_arr = []
    all_probs_tta = None

    for t_idx, tta_t in enumerate(tta_transforms_list):
        ds = MinifigDataset(test_data, test_labels, tta_t, img_size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        probs_this = []
        labels_this = []
        with torch.no_grad():
            for images, lbl in loader:
                images = images.to(DEVICE)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probs_this.extend(probs)
                if t_idx == 0:
                    labels_this.extend(lbl.numpy())
        if t_idx == 0:
            all_labels_arr = np.array(labels_this)
            all_probs_tta = np.array(probs_this)
        else:
            all_probs_tta += np.array(probs_this)
        print(f"  TTA view {t_idx+1}/{len(tta_transforms_list)} done")

    all_probs_tta /= len(tta_transforms_list)
    all_preds_tta = all_probs_tta.argmax(axis=1)

    acc = accuracy_score(all_labels_arr, all_preds_tta)
    macro_f1 = f1_score(all_labels_arr, all_preds_tta, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels_arr, all_preds_tta, average='weighted', zero_division=0)
    macro_prec = precision_score(all_labels_arr, all_preds_tta, average='macro', zero_division=0)
    macro_rec = recall_score(all_labels_arr, all_preds_tta, average='macro', zero_division=0)
    top3_acc = top_k_accuracy_score(all_labels_arr, all_probs_tta, k=3, labels=range(num_classes))
    top5_acc = top_k_accuracy_score(all_labels_arr, all_probs_tta, k=5, labels=range(num_classes))

    return {
        'Accuracy': acc, 'Top-3 Accuracy': top3_acc, 'Top-5 Accuracy': top5_acc,
        'Macro F1': macro_f1, 'Weighted F1': weighted_f1,
        'Macro Precision': macro_prec, 'Macro Recall': macro_rec
    }, all_preds_tta, all_labels_arr, all_probs_tta


# ============================================================
# 1. BASELINE — EfficientNet-B0, 122 classes, IMG_SIZE=128
# ============================================================
print("\n" + "="*80)
print("MODEL 1: BASELINE (EfficientNet-B0, 122 classes, 128px)")
print("="*80)

test_data_bl, test_labels_bl, cat2idx_bl, idx2cat_bl, nc_bl = build_split(valid_data)
print(f"Test set: {len(test_data_bl)} samples, {nc_bl} classes")

eval_tf_128 = transforms.Compose([
    transforms.Resize((128, 128)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_loader_bl = DataLoader(MinifigDataset(test_data_bl, test_labels_bl, eval_tf_128, 128),
                            batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

model_bl = models.efficientnet_b0(weights=None)
model_bl.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model_bl.classifier[1].in_features, nc_bl))
model_bl.load_state_dict(torch.load(os.path.join(BASE_DIR, "baseline_results", "best_model.pth"),
                                     map_location=DEVICE, weights_only=True))
model_bl = model_bl.to(DEVICE)

metrics_bl, preds_bl, labels_bl, probs_bl = evaluate_model(model_bl, test_loader_bl, nc_bl, idx2cat_bl)
print(f"\n{'Metric':<30} {'Value':>10}")
print("-" * 42)
for k, v in metrics_bl.items():
    print(f"{k:<30} {v:>10.4f}")
del model_bl
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# 2. OPTION B — EfficientNet-B0, merged classes, IMG_SIZE=128
# ============================================================
print("\n" + "="*80)
print("MODEL 2: OPTION B (EfficientNet-B0, merged classes, 128px)")
print("="*80)

test_data_ob, test_labels_ob, cat2idx_ob, idx2cat_ob, nc_ob = build_split(valid_data, MERGE_MAP_B)
print(f"Test set: {len(test_data_ob)} samples, {nc_ob} classes")

test_loader_ob = DataLoader(MinifigDataset(test_data_ob, test_labels_ob, eval_tf_128, 128),
                            batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

model_ob = models.efficientnet_b0(weights=None)
model_ob.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model_ob.classifier[1].in_features, nc_ob))
model_ob.load_state_dict(torch.load(os.path.join(BASE_DIR, "optionB_results", "best_model.pth"),
                                     map_location=DEVICE, weights_only=True))
model_ob = model_ob.to(DEVICE)

metrics_ob, preds_ob, labels_ob, probs_ob = evaluate_model(model_ob, test_loader_ob, nc_ob, idx2cat_ob)
print(f"\n{'Metric':<30} {'Value':>10}")
print("-" * 42)
for k, v in metrics_ob.items():
    print(f"{k:<30} {v:>10.4f}")
del model_ob
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# 3. OPTION B IMPROVED — EfficientNet-B0, merged, IMG_SIZE=224
# ============================================================
print("\n" + "="*80)
print("MODEL 3: OPTION B IMPROVED (EfficientNet-B0, merged, 224px, Focal+CutMix+MixUp)")
print("="*80)

test_data_oi, test_labels_oi, cat2idx_oi, idx2cat_oi, nc_oi = build_split(valid_data, MERGE_MAP_B)
print(f"Test set: {len(test_data_oi)} samples, {nc_oi} classes")

eval_tf_224 = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_loader_oi = DataLoader(MinifigDataset(test_data_oi, test_labels_oi, eval_tf_224, 224),
                            batch_size=32, shuffle=False, num_workers=NUM_WORKERS)

model_oi = models.efficientnet_b0(weights=None)
model_oi.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(model_oi.classifier[1].in_features, nc_oi))
model_oi.load_state_dict(torch.load(os.path.join(BASE_DIR, "optionB_improved_results", "best_model.pth"),
                                     map_location=DEVICE, weights_only=True))
model_oi = model_oi.to(DEVICE)

metrics_oi, preds_oi, labels_oi, probs_oi = evaluate_model(model_oi, test_loader_oi, nc_oi, idx2cat_oi)
print(f"\n{'Metric':<30} {'Value':>10}")
print("-" * 42)
for k, v in metrics_oi.items():
    print(f"{k:<30} {v:>10.4f}")
del model_oi
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# 4. OPTION B V2 — EfficientNet-B2, Town split, IMG_SIZE=260, TTA
# ============================================================
print("\n" + "="*80)
print("MODEL 4: OPTION B V2 (EfficientNet-B2, Town split, 260px, deeper head)")
print("="*80)

test_data_v2, test_labels_v2, cat2idx_v2, idx2cat_v2, nc_v2 = build_split(valid_data, MERGE_MAP_B, town_split=True)
print(f"Test set: {len(test_data_v2)} samples, {nc_v2} classes")

eval_tf_260 = transforms.Compose([
    PadToSquare(), transforms.Resize((260, 260)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_loader_v2 = DataLoader(MinifigDataset(test_data_v2, test_labels_v2, eval_tf_260, 260),
                            batch_size=24, shuffle=False, num_workers=NUM_WORKERS)

model_v2 = models.efficientnet_b2(weights=None)
in_features_v2 = model_v2.classifier[1].in_features  # 1408
model_v2.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features_v2, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, nc_v2)
)
model_v2.load_state_dict(torch.load(os.path.join(BASE_DIR, "optionB_v2_results", "best_model.pth"),
                                     map_location=DEVICE, weights_only=True))
model_v2 = model_v2.to(DEVICE)

# Standard eval (no TTA)
metrics_v2, preds_v2, labels_v2, probs_v2 = evaluate_model(model_v2, test_loader_v2, nc_v2, idx2cat_v2)
print(f"\n--- Without TTA ---")
print(f"{'Metric':<30} {'Value':>10}")
print("-" * 42)
for k, v in metrics_v2.items():
    print(f"{k:<30} {v:>10.4f}")

# TTA eval
print(f"\n--- With TTA (5 views) ---")
metrics_v2_tta, preds_v2_tta, labels_v2_tta, probs_v2_tta = evaluate_with_tta(
    model_v2, test_data_v2, test_labels_v2, nc_v2, idx2cat_v2, 260, 24)
print(f"\n{'Metric':<30} {'No TTA':>10} {'With TTA':>10} {'Delta':>8}")
print("-" * 60)
for k in metrics_v2:
    v1, v2t = metrics_v2[k], metrics_v2_tta[k]
    d = v2t - v1
    print(f"{k:<30} {v1:>10.4f} {v2t:>10.4f} {'+' if d>=0 else ''}{d:>7.4f}")

del model_v2
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# FULL COMPARISON TABLE
# ============================================================
print("\n" + "="*80)
print("FULL COMPARISON: ALL MODELS")
print("="*80)
print(f"{'Metric':<25} {'Baseline':>10} {'Opt B':>10} {'B Improv':>10} {'V2':>10} {'V2+TTA':>10}")
print("-" * 77)
for metric in metrics_bl:
    b = metrics_bl[metric]
    o = metrics_ob[metric]
    i = metrics_oi[metric]
    v = metrics_v2[metric]
    vt = metrics_v2_tta[metric]
    print(f"{metric:<25} {b:>10.4f} {o:>10.4f} {i:>10.4f} {v:>10.4f} {vt:>10.4f}")

print(f"{'Num Classes':<25} {nc_bl:>10} {nc_ob:>10} {nc_oi:>10} {nc_v2:>10} {nc_v2:>10}")

# ============================================================
# SAVE COMPARISON PLOT
# ============================================================
output_dir = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(output_dir, exist_ok=True)

# Bar chart comparison
metrics_to_plot = ['Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Macro F1', 'Weighted F1']
model_names = ['Baseline', 'Option B', 'B Improved', 'V2', 'V2+TTA']
all_metrics = [metrics_bl, metrics_ob, metrics_oi, metrics_v2, metrics_v2_tta]

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(metrics_to_plot))
width = 0.15
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

for i, (name, mets) in enumerate(zip(model_names, all_metrics)):
    vals = [mets[m] for m in metrics_to_plot]
    bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_ylabel('Score')
ax.set_title('Model Comparison — All Metrics')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics_to_plot)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150)
plt.close()

# Save results to text file
with open(os.path.join(output_dir, "comparison_results.txt"), 'w') as f:
    f.write("FULL MODEL COMPARISON\n")
    f.write("="*80 + "\n")
    f.write(f"{'Metric':<25} {'Baseline':>10} {'Opt B':>10} {'B Improv':>10} {'V2':>10} {'V2+TTA':>10}\n")
    f.write("-" * 77 + "\n")
    for metric in metrics_bl:
        b = metrics_bl[metric]
        o = metrics_ob[metric]
        i = metrics_oi[metric]
        v = metrics_v2[metric]
        vt = metrics_v2_tta[metric]
        f.write(f"{metric:<25} {b:>10.4f} {o:>10.4f} {i:>10.4f} {v:>10.4f} {vt:>10.4f}\n")
    f.write(f"{'Num Classes':<25} {nc_bl:>10} {nc_ob:>10} {nc_oi:>10} {nc_v2:>10} {nc_v2:>10}\n")

print(f"\nComparison chart saved to: {os.path.join(output_dir, 'model_comparison.png')}")
print(f"Results saved to: {os.path.join(output_dir, 'comparison_results.txt')}")
print("\nDone!")
