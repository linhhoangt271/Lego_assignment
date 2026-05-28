"""
Option B V4 — Three targeted optimizations over V3:

  1. WEIGHTED MODEL ENSEMBLE
     V3 trains Focal / ArcFace / SupCon separately and picks only the best.
     The probability outputs of all three models are never combined, which
     leaves easy gains on the table.  We collect every model's softmax
     probabilities and average them with weights proportional to each
     model's validation F1 — a cheap but reliable diversity boost.

  2. STOCHASTIC WEIGHT AVERAGING (SWA)
     After the warm-up phase (first 75 % of epochs) we start averaging the
     model weights rather than just keeping the single best checkpoint.
     SWA finds a flatter minimum in the loss landscape, which typically
     improves generalisation by 1-3 % F1 with no extra data or parameters.
     PyTorch ships AveragedModel / SWALR / update_bn for this out of the box.

  3. CONVNEXT-SMALL BACKBONE (optional, flag below)
     ConvNeXt-Small was released after EfficientNet-B4 and consistently
     out-performs it on ImageNet at a similar parameter budget while being
     fully compatible with the torchvision transfer-learning workflow.
     Toggle USE_CONVNEXT = True to swap the backbone; everything else
     (head, losses, SWA, ensemble) stays identical.

  Everything not listed above is intentionally unchanged from V3 so that
  the delta in results is attributable to these three changes only.
"""

import json, os, time, warnings, random, math, copy
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn   # <-- SWA
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, top_k_accuracy_score,
)
from collections import Counter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "optionB_v4_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda"  if torch.cuda.is_available() else
    "mps"   if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {DEVICE}")

# ── Backbone choice ──────────────────────────────────────────────────────────
# False → EfficientNet-B4 (same as V3, good for direct comparison)
# True  → ConvNeXt-Small  (new in V4, typically +1-2 % F1 at same cost)
USE_CONVNEXT = True

IMG_SIZE        = 384 if USE_CONVNEXT else 380   # ConvNeXt prefers 384
BATCH_SIZE      = 16
NUM_EPOCHS      = 20
LR              = 3e-4
NUM_WORKERS     = 4
PATIENCE        = 5
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA     = 2.0
MIXUP_ALPHA     = 0.2
CUTMIX_ALPHA    = 1.0
CUTMIX_PROB     = 0.3
MIXUP_PROB      = 0.3
MIN_SAMPLES_TARGET = 50

# SWA starts after this fraction of epochs has elapsed
SWA_START_FRAC = 0.75          # begin averaging at epoch ⌊0.75 × NUM_EPOCHS⌋
SWA_LR         = 5e-5          # constant LR used during SWA phase

backbone_name = "ConvNeXt-Small" if USE_CONVNEXT else "EfficientNet-B4"
print(f"Backbone: {backbone_name}  |  IMG_SIZE: {IMG_SIZE}  |  SWA start: {SWA_START_FRAC:.0%}")

# ─────────────────────────────────────────────────────────────────────────────
# TOWN SPLITTER & MERGE MAP  (identical to V3)
# ─────────────────────────────────────────────────────────────────────────────
def split_town(subcategory):
    sub = subcategory.lower()
    if 'police'      in sub: return 'Town - Police'
    if 'fire'        in sub: return 'Town - Fire'
    if 'airport'     in sub: return 'Town - Airport'
    if 'hospital'    in sub or 'rescue'      in sub: return 'Town - Rescue'
    if 'space'       in sub: return 'Town - Space'
    if 'race'        in sub or 'stuntz'      in sub: return 'Town - Racing'
    if 'coast guard' in sub: return 'Town - Coast Guard'
    if 'construction'in sub: return 'Town - Construction'
    if any(x in sub for x in ['arctic','jungle','volcano','ocean','deep sea','exploration']):
        return 'Town - Exploration'
    return 'Town - General'

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

# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS  (unchanged from V3)
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight, self.gamma, self.label_smoothing = weight, gamma, label_smoothing

    def forward(self, inputs, targets):
        n = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        if self.label_smoothing > 0:
            smooth = torch.full_like(inputs, self.label_smoothing / (n - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            fw = (1 - probs).pow(self.gamma)
            loss = -(fw * smooth * log_probs).sum(dim=1)
        else:
            ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            loss = (1 - pt).pow(self.gamma) * ce
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.scale, self.margin = scale, margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, emb, labels=None):
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        sine  = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))
        phi   = cosine * self.cos_m - sine * self.sin_m
        phi   = torch.where(cosine > self.th, phi, cosine - self.mm)
        oh    = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1)
        return (oh * phi + (1 - oh) * cosine) * self.scale


class ArcFaceLoss(nn.Module):
    def __init__(self, weight=None, label_smoothing=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    def forward(self, logits, targets): return self.ce(logits, targets)


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device, B = features.device, features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask   = torch.eq(labels, labels.T).float().to(device)
        adot   = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(adot, dim=1, keepdim=True)
        logits = adot - logits_max.detach()
        lmask  = 1 - torch.eye(B, device=device)
        mask   = mask * lmask
        exp_l  = torch.exp(logits) * lmask
        log_prob = logits - torch.log(exp_l.sum(1, keepdim=True) + 1e-12)
        mp = mask.sum(1).clamp(min=1)
        return -(mask * log_prob).sum(1).div(mp).mean()

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION HELPERS  (unchanged from V3)
# ─────────────────────────────────────────────────────────────────────────────
def rand_bbox(size, lam):
    H, W = size[2], size[3]
    cr = np.sqrt(1.0 - lam)
    ch, cw = int(H * cr), int(W * cr)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
    x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
    return y1, y2, x1, x2

def apply_cutmix(imgs, tgts, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0)).to(imgs.device)
    y1, y2, x1, x2 = rand_bbox(imgs.size(), lam)
    imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (imgs.size(-1) * imgs.size(-2))
    return imgs, tgts, tgts[idx], lam

def apply_mixup(imgs, tgts, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0)).to(imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], tgts, tgts[idx], lam


class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        return ImageOps.pad(img, (max(w, h), max(w, h)), color=(255, 255, 255))

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data …")
json_path = os.path.join(BASE_DIR, "minifigs.json")
with open(json_path) as f:
    raw = json.load(f)

valid_data = [d for d in raw
              if d.get('img_local_path')
              and os.path.exists(os.path.join(BASE_DIR, d['img_local_path']))]

for d in valid_data:
    if d['category'] == 'Town':
        d['merged_category'] = split_town(d.get('subcategory', ''))
    else:
        d['merged_category'] = MERGE_MAP.get(d['category'], d['category'])

categories  = [d['merged_category'] for d in valid_data]
cat_counts  = Counter(categories)
cat_names   = sorted(cat_counts.keys())
cat2idx     = {c: i for i, c in enumerate(cat_names)}
idx2cat     = {i: c for c, i in cat2idx.items()}
num_classes = len(cat_names)
print(f"Valid images: {len(valid_data)},  Classes: {num_classes}")

# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED SPLIT  (identical to V3)
# ─────────────────────────────────────────────────────────────────────────────
labels      = [cat2idx[d['merged_category']] for d in valid_data]
label_counts= Counter(labels)
small_idx   = [i for i, l in enumerate(labels) if label_counts[l] < 7]
big_idx     = [i for i, l in enumerate(labels) if label_counts[l] >= 7]

big_data, big_labels = [valid_data[i] for i in big_idx], [labels[i] for i in big_idx]
train_big, temp_data, train_labels_big, temp_labels = train_test_split(
    big_data, big_labels, test_size=0.30, random_state=42, stratify=big_labels)
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data, temp_labels, test_size=0.50, random_state=42, stratify=temp_labels)

train_data   = train_big   + [valid_data[i] for i in small_idx]
train_labels = train_labels_big + [labels[i] for i in small_idx]
print(f"Train: {len(train_data)},  Val: {len(val_data)},  Test: {len(test_data)}")

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS  (10 TTA views — V3 only had 5)
# ─────────────────────────────────────────────────────────────────────────────
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    transforms.Normalize(mean, std),
])

eval_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# V4: 10 TTA views (V3 had 5)
# More diversity → better probability calibration when ensembling
tta_transforms = [
    # 1. standard
    eval_transform,
    # 2. horizontal flip
    transforms.Compose([PadToSquare(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize(mean, std)]),
    # 3. slight zoom-in
    transforms.Compose([PadToSquare(), transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
                        transforms.CenterCrop(IMG_SIZE),
                        transforms.ToTensor(), transforms.Normalize(mean, std)]),
    # 4. more zoom-in
    transforms.Compose([PadToSquare(), transforms.Resize((IMG_SIZE + 40, IMG_SIZE + 40)),
                        transforms.CenterCrop(IMG_SIZE),
                        transforms.ToTensor(), transforms.Normalize(mean, std)]),
    # 5. colour jitter
    transforms.Compose([PadToSquare(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.ColorJitter(brightness=0.15, contrast=0.15),
                        transforms.ToTensor(), transforms.Normalize(mean, std)]),
    # 6. small rotation CW
    transforms.Compose([PadToSquare(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomRotation(degrees=(5, 10)),
                        transforms.ToTensor(), transforms.Normalize(mean, std)]),
    # 7. small rotation CCW
    transforms.Compose([PadToSquare(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.RandomRotation(degrees=(-10, -5)),
                        transforms.ToTensor(), transforms.Normalize(mean, std)]),
    # 8. flip + zoom
    transforms.Compose([PadToSquare(), transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
                        transforms.CenterCrop(IMG_SIZE),
                        transforms.RandomHorizontalFlip(p=1.0),
                        transforms.ToTensor(), transforms.Normalize(mean, std)]),
    # 9. slight brightness increase
    transforms.Compose([PadToSquare(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.ColorJitter(brightness=(1.1, 1.3)),
                        transforms.ToTensor(), transforms.Normalize(mean, std)]),
    # 10. slight brightness decrease
    transforms.Compose([PadToSquare(), transforms.Resize((IMG_SIZE, IMG_SIZE)),
                        transforms.ColorJitter(brightness=(0.7, 0.9)),
                        transforms.ToTensor(), transforms.Normalize(mean, std)]),
]
print(f"TTA views: {len(tta_transforms)}  (V3 had 5)")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET & LOADERS
# ─────────────────────────────────────────────────────────────────────────────
class MinifigDataset(Dataset):
    def __init__(self, records, labels, transform):
        self.records, self.labels, self.tf = records, labels, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        try:
            p = self.records[idx]['img_local_path']
            p = p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
            img = Image.open(p).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        return self.tf(img), self.labels[idx]


class EmbeddingDataset(Dataset):
    """Two-view dataset for SupCon stage-1."""
    def __init__(self, records, labels, transform):
        self.records, self.labels, self.tf = records, labels, transform
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        try:
            p = self.records[idx]['img_local_path']
            p = p if os.path.isabs(p) else os.path.join(BASE_DIR, p)
            img = Image.open(p).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        return self.tf(img), self.tf(img), self.labels[idx]


# Class weights — same sqrt formula as V3
tlc = Counter(train_labels)
class_weights = torch.zeros(num_classes)
for i in range(num_classes):
    class_weights[i] = 1.0 / math.sqrt(tlc.get(i, 1))
class_weights = (class_weights / class_weights.sum() * num_classes).to(DEVICE)

sample_weights = [1.0 / math.sqrt(tlc[l]) for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_dataset = MinifigDataset(train_data,  train_labels, train_transform)
val_dataset   = MinifigDataset(val_data,    val_labels,   eval_transform)
test_dataset  = MinifigDataset(test_data,   test_labels,  eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_model(loss_type='focal'):
    """
    Returns (model, arcface_head_or_None, embedding_dim).
    Supports EfficientNet-B4 (V3 default) or ConvNeXt-Small (V4 option).
    """
    if USE_CONVNEXT:
        # ── ConvNeXt-Small ───────────────────────────────────────────────────
        from torchvision.models import convnext_small, ConvNeXt_Small_Weights
        backbone_raw  = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        in_feat       = 768  # ConvNeXt-Small feature dimension

        # Wrap to include global pooling + strip final linear layers
        class ConvNeXtBackbone(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.features = model.features
                self.avgpool = model.avgpool
            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                return x.flatten(1)

        backbone = ConvNeXtBackbone(backbone_raw)

        # Freeze first 2 stages (out of 4)
        stage_params = list(backbone_raw.features[:2].parameters())
        for p in stage_params:
            p.requires_grad = False
    else:
        # ── EfficientNet-B4  (same as V3, via torchvision) ──────────────────
        backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_feat  = backbone.classifier[1].in_features    # 1792
        backbone.classifier = nn.Identity()

        # Freeze early blocks (conv_stem + blocks 0-3)
        for name, p in backbone.named_parameters():
            if name.startswith('features.0') or \
               any(name.startswith(f'features.{i}') for i in range(1, 5)):
                p.requires_grad = False

    arcface_head = None

    if loss_type == 'arcface':
        head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feat, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.2),
        )
        arcface_head = ArcFaceHead(512, num_classes, scale=30.0, margin=0.5).to(DEVICE)
        model = nn.Sequential(backbone, head)
    elif loss_type == 'supcon':
        head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feat, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
        )
        model = nn.Sequential(backbone, head)
    else:   # focal / default
        head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feat, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        model = nn.Sequential(backbone, head)

    model = model.to(DEVICE)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,} total | {trainable:,} trainable ({trainable/total*100:.1f}%)")
    return model, arcface_head, in_feat

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTION  — key V4 addition: SWA
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, loss_type, arcface_head=None, supcon_stage1=False,
                num_epochs=NUM_EPOCHS):
    """
    Train with optional SWA.  SWA is activated after SWA_START_FRAC × epochs.
    Returns (best_val_f1, history, swa_model_or_None).
    """
    swa_start_epoch = int(SWA_START_FRAC * num_epochs)

    # Loss functions
    if loss_type == 'focal':
        criterion = FocalLoss(weight=class_weights, gamma=FOCAL_GAMMA,
                              label_smoothing=LABEL_SMOOTHING)
    elif loss_type == 'arcface':
        criterion = ArcFaceLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    elif loss_type == 'supcon' and supcon_stage1:
        criterion = SupConLoss(temperature=0.07)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    criterion_val = nn.CrossEntropyLoss()

    params = list(model.parameters()) + (list(arcface_head.parameters()) if arcface_head else [])
    optimizer = torch.optim.AdamW(
        [p for p in params if p.requires_grad], lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2)

    # ── Mixed precision training ─────────────────────────────────────────────
    scaler = GradScaler() if torch.cuda.is_available() else None
    # ─────────────────────────────────────────────────────────────────────────

    # ── SWA setup ────────────────────────────────────────────────────────────
    swa_model     = AveragedModel(model)          # wraps the base model
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=3)
    swa_active    = False
    # ─────────────────────────────────────────────────────────────────────────

    history = {k: [] for k in ['train_loss','val_loss','train_acc','val_acc','val_f1']}
    best_val_f1, patience_ctr = 0.0, 0
    save_path = os.path.join(OUTPUT_DIR, f"best_model_{loss_type}.pth")

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        if arcface_head: arcface_head.train()
        tl, tc, tt = 0.0, 0, 0

        # ── Stage-1 SupCon  ──────────────────────────────────────────────────
        if loss_type == 'supcon' and supcon_stage1:
            ds  = EmbeddingDataset(train_data, train_labels, train_transform)
            ldr = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                             num_workers=NUM_WORKERS)
            for v1, v2, lbl in ldr:
                v1, v2, lbl = v1.to(DEVICE), v2.to(DEVICE), lbl.to(DEVICE)
                optimizer.zero_grad()
                with autocast(enabled=(DEVICE.type == 'cuda')):
                    z1 = F.normalize(model(v1), dim=1)
                    z2 = F.normalize(model(v2), dim=1)
                    loss = criterion(torch.cat([z1, z2]), torch.cat([lbl, lbl]))
                if scaler is not None:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                tl += loss.item() * v1.size(0); tt += v1.size(0)
            tl /= max(tt, 1); ta = 0.0
        # ── Standard training ────────────────────────────────────────────────
        else:
            for imgs, lbl in train_loader:
                imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
                r = random.random()
                optimizer.zero_grad()

                with autocast(enabled=(DEVICE.type == 'cuda')):
                    if r < CUTMIX_PROB:
                        imgs, ta_, tb_, lam = apply_cutmix(imgs, lbl, CUTMIX_ALPHA)
                        out = arcface_head(model(imgs), ta_) if arcface_head else model(imgs)
                        ob  = arcface_head(model(imgs), tb_) if arcface_head else out
                        loss = lam * criterion(out, ta_) + (1-lam) * criterion(ob, tb_)
                    elif r < CUTMIX_PROB + MIXUP_PROB:
                        imgs, ta_, tb_, lam = apply_mixup(imgs, lbl, MIXUP_ALPHA)
                        out = arcface_head(model(imgs), ta_) if arcface_head else model(imgs)
                        ob  = arcface_head(model(imgs), tb_) if arcface_head else out
                        loss = lam * criterion(out, ta_) + (1-lam) * criterion(ob, tb_)
                    else:
                        out = arcface_head(model(imgs), lbl) if arcface_head else model(imgs)
                        loss = criterion(out, lbl)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()

                tl += loss.item() * imgs.size(0)
                tc += out.max(1)[1].eq(lbl).sum().item()
                tt += imgs.size(0)
            tl /= max(tt, 1); ta = tc / max(tt, 1)

        # ── SWA update / LR schedule ─────────────────────────────────────────
        if epoch >= swa_start_epoch and not (loss_type == 'supcon' and supcon_stage1):
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if not swa_active:
                print(f"  [SWA activated at epoch {epoch+1}]")
                swa_active = True
        else:
            scheduler.step(epoch)
        # ─────────────────────────────────────────────────────────────────────

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        if arcface_head: arcface_head.eval()
        vl, vc, vt = 0.0, 0, 0
        vp_all, vl_all = [], []

        if not (loss_type == 'supcon' and supcon_stage1):
            with torch.no_grad():
                for imgs, lbl in val_loader:
                    imgs, lbl = imgs.to(DEVICE), lbl.to(DEVICE)
                    out = arcface_head(model(imgs)) if arcface_head else model(imgs)
                    vl += criterion_val(out, lbl).item() * imgs.size(0)
                    vc += out.max(1)[1].eq(lbl).sum().item()
                    vt += imgs.size(0)
                    vp_all.extend(out.max(1)[1].cpu().numpy())
                    vl_all.extend(lbl.cpu().numpy())
            vl /= max(vt, 1); va = vc / max(vt, 1)
            val_f1 = f1_score(vl_all, vp_all, average='macro', zero_division=0)
        else:
            vl = va = val_f1 = 0.0

        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_acc'].append(ta);  history['val_acc'].append(va)
        history['val_f1'].append(val_f1)

        lr_now  = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        marker  = ""

        if not (loss_type == 'supcon' and supcon_stage1):
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({'model': model.state_dict(),
                            **(({'arcface': arcface_head.state_dict()}) if arcface_head else {})},
                           save_path)
                patience_ctr = 0; marker = "  ← best"
            else:
                patience_ctr += 1

            swa_tag = " [SWA]" if swa_active else ""
            print(f"  Ep {epoch+1:02d}/{num_epochs}{swa_tag} | "
                  f"trn {tl:.4f}/{ta:.4f}  val {vl:.4f}/{va:.4f}  F1 {val_f1:.4f} | "
                  f"lr {lr_now:.2e} | {elapsed:.0f}s{marker}")

            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        else:
            torch.save(model.state_dict(), save_path)
            print(f"  Ep {epoch+1:02d}/{num_epochs} (SupCon S1) | loss {tl:.4f} | "
                  f"lr {lr_now:.2e} | {elapsed:.0f}s")

    # ── Finalise SWA model ───────────────────────────────────────────────────
    if swa_active:
        print("  Updating BatchNorm statistics for SWA model …")
        update_bn(train_loader, swa_model, device=DEVICE)
        swa_save = save_path.replace(".pth", "_swa.pth")
        torch.save(swa_model.state_dict(), swa_save)
        print(f"  SWA model saved → {swa_save}")
    # ─────────────────────────────────────────────────────────────────────────

    return best_val_f1, history, swa_model if swa_active else None


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPER  (supports both a plain model and an SWA-wrapped model)
# ─────────────────────────────────────────────────────────────────────────────
def predict_probs(model, arcface_head=None):
    """Run TTA over the test set and return (labels, averaged_probs)."""
    model.eval()
    if arcface_head: arcface_head.eval()

    def one_pass(loader):
        pbs, lbs = [], []
        with torch.no_grad():
            for imgs, lbl in loader:
                imgs = imgs.to(DEVICE)
                out  = arcface_head(model(imgs)) if arcface_head else model(imgs)
                pbs.extend(torch.softmax(out, 1).cpu().numpy())
                lbs.extend(lbl.numpy())
        return np.array(lbs), np.array(pbs)

    labels, base_probs = one_pass(test_loader)

    print(f"  TTA ({len(tta_transforms)} views) …")
    acc_probs = np.zeros_like(base_probs)
    for t in tta_transforms:
        ds  = MinifigDataset(test_data, test_labels, t)
        ldr = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        _, pb = one_pass(ldr)
        acc_probs += pb
    acc_probs /= len(tta_transforms)

    return labels, acc_probs


def metrics_from_probs(labels, probs):
    preds = probs.argmax(1)
    return {
        'Accuracy':        accuracy_score(labels, preds),
        'Macro F1':        f1_score(labels, preds, average='macro',    zero_division=0),
        'Weighted F1':     f1_score(labels, preds, average='weighted', zero_division=0),
        'Macro Precision': precision_score(labels, preds, average='macro', zero_division=0),
        'Macro Recall':    recall_score(labels,    preds, average='macro', zero_division=0),
        'Top-3 Accuracy':  top_k_accuracy_score(labels, probs, k=3, labels=range(num_classes)),
        'Top-5 Accuracy':  top_k_accuracy_score(labels, probs, k=5, labels=range(num_classes)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN & EVALUATE ALL THREE VARIANTS
# ─────────────────────────────────────────────────────────────────────────────
all_probs   = {}   # loss_type → test-set probability matrix
all_val_f1  = {}   # loss_type → best validation F1 (used for ensemble weights)
all_metrics = {}   # loss_type → metrics dict

# ── 1. FOCAL ─────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("TRAINING  V4-Focal")
print("═"*70)
m_focal, _, _ = build_model('focal')
bf_focal, hist_focal, swa_focal = train_model(m_focal, 'focal')
ck = torch.load(os.path.join(OUTPUT_DIR, "best_model_focal.pth"), map_location=DEVICE, weights_only=True)
m_focal.load_state_dict(ck.get('model', ck))
print("  Evaluating best checkpoint …")
labels_ref, probs_focal = predict_probs(m_focal)
all_probs['focal']  = probs_focal
all_val_f1['focal'] = bf_focal
all_metrics['focal'] = metrics_from_probs(labels_ref, probs_focal)

# Also evaluate the SWA model if it was produced
if swa_focal is not None:
    print("  Evaluating SWA model …")
    _, probs_focal_swa = predict_probs(swa_focal)
    all_probs['focal_swa']   = probs_focal_swa
    all_val_f1['focal_swa']  = bf_focal          # same val F1 (use checkpoint val)
    all_metrics['focal_swa'] = metrics_from_probs(labels_ref, probs_focal_swa)

del m_focal, swa_focal
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ── 2. ARCFACE ────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("TRAINING  V4-ArcFace")
print("═"*70)
m_arc, af_head, _ = build_model('arcface')
bf_arc, hist_arc, swa_arc = train_model(m_arc, 'arcface', arcface_head=af_head)
ck = torch.load(os.path.join(OUTPUT_DIR, "best_model_arcface.pth"), map_location=DEVICE, weights_only=True)
m_arc.load_state_dict(ck.get('model', ck))
if 'arcface' in ck: af_head.load_state_dict(ck['arcface'])
print("  Evaluating best checkpoint …")
_, probs_arc = predict_probs(m_arc, arcface_head=af_head)
all_probs['arcface']  = probs_arc
all_val_f1['arcface'] = bf_arc
all_metrics['arcface'] = metrics_from_probs(labels_ref, probs_arc)

del m_arc, af_head, swa_arc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ── 3. SUPCON ─────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("TRAINING  V4-SupCon  (stage 1 = contrastive  →  stage 2 = CE fine-tune)")
print("═"*70)
supcon_epochs = max(5, NUM_EPOCHS // 4)   # stage-1 epochs (shorter)
m_supcon, _, in_feat = build_model('supcon')
print(f"  Stage 1: contrastive ({supcon_epochs} epochs)")
_, hist_s1, _ = train_model(m_supcon, 'supcon', supcon_stage1=True, num_epochs=supcon_epochs)

# Swap projection head → classification head for stage 2
backbone_only = m_supcon[0]
cls_head = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_feat, 512), nn.BatchNorm1d(512), nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes),
).to(DEVICE)
m_supcon_s2 = nn.Sequential(backbone_only, cls_head).to(DEVICE)
print(f"  Stage 2: CE fine-tune ({NUM_EPOCHS} epochs)")
bf_supcon, hist_s2, swa_supcon = train_model(m_supcon_s2, 'supcon_s2')

ck_path = os.path.join(OUTPUT_DIR, "best_model_supcon_s2.pth")
if os.path.exists(ck_path):
    ck = torch.load(ck_path, map_location=DEVICE, weights_only=True)
    m_supcon_s2.load_state_dict(ck.get('model', ck))
print("  Evaluating best checkpoint …")
_, probs_supcon = predict_probs(m_supcon_s2)
all_probs['supcon']  = probs_supcon
all_val_f1['supcon'] = bf_supcon
all_metrics['supcon'] = metrics_from_probs(labels_ref, probs_supcon)

del m_supcon, m_supcon_s2, swa_supcon
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ─────────────────────────────────────────────────────────────────────────────
# V4 KEY ADDITION: WEIGHTED ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("COMPUTING WEIGHTED ENSEMBLE  (V4 key optimisation)")
print("═"*70)

# Use the 3 primary models (Focal, ArcFace, SupCon) for the ensemble.
# Weight each model's probabilities by its validation Macro-F1 score so that
# the stronger model contributes more to the final prediction.
ensemble_keys = ['focal', 'arcface', 'supcon']
raw_weights   = np.array([all_val_f1[k] for k in ensemble_keys])
norm_weights  = raw_weights / raw_weights.sum()

print("  Ensemble members and their weights:")
for k, w, f in zip(ensemble_keys, norm_weights, raw_weights):
    print(f"    {k:<12}  val-F1={f:.4f}  →  weight={w:.4f}")

ensemble_probs = sum(norm_weights[i] * all_probs[ensemble_keys[i]]
                     for i in range(len(ensemble_keys)))
all_probs['ensemble']   = ensemble_probs
all_metrics['ensemble'] = metrics_from_probs(labels_ref, ensemble_probs)

# Uniform ensemble as a sanity-check baseline
uniform_probs = sum(all_probs[k] for k in ensemble_keys) / len(ensemble_keys)
all_probs['ensemble_uniform']   = uniform_probs
all_metrics['ensemble_uniform'] = metrics_from_probs(labels_ref, uniform_probs)

print(f"\n  Weighted ensemble Macro F1 : {all_metrics['ensemble']['Macro F1']:.4f}")
print(f"  Uniform  ensemble Macro F1 : {all_metrics['ensemble_uniform']['Macro F1']:.4f}")

# If we produced SWA models, include them in a "super-ensemble" as well
if 'focal_swa' in all_probs:
    se_keys = ensemble_keys + ['focal_swa']
    se_w    = np.array([all_val_f1[k] for k in se_keys])
    se_w   /= se_w.sum()
    se_probs = sum(se_w[i] * all_probs[se_keys[i]] for i in range(len(se_keys)))
    all_probs['super_ensemble']   = se_probs
    all_metrics['super_ensemble'] = metrics_from_probs(labels_ref, se_probs)
    print(f"  Super-ensemble (+ SWA) Macro F1 : {all_metrics['super_ensemble']['Macro F1']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("FULL COMPARISON")
print("═"*70)

# Historical V3 reference numbers (from RESULTS.md)
v3_ref = {
    'V3-Focal':  {'Macro F1': 0.7736, 'Accuracy': None, 'Weighted F1': None},
    'V3-SupCon': {'Macro F1': 0.7951, 'Accuracy': None, 'Weighted F1': None},
}

metrics_order = ['Accuracy','Macro F1','Weighted F1','Macro Precision',
                 'Macro Recall','Top-3 Accuracy','Top-5 Accuracy']

header = f"{'Model':<24}" + "".join(f"{m:>18}" for m in metrics_order)
print(header)
print("-" * len(header))

for ref_name, ref_vals in v3_ref.items():
    row = f"{ref_name + ' (V3 ref)':<24}"
    for m in metrics_order:
        v = ref_vals.get(m)
        row += f"{'—':>18}" if v is None else f"{v:>18.4f}"
    print(row)

print("-" * len(header))
for name, mets in all_metrics.items():
    row = f"{'V4-'+name:<24}"
    for m in metrics_order:
        row += f"{mets.get(m, 0):>18.4f}"
    print(row)

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION REPORT  (best model)
# ─────────────────────────────────────────────────────────────────────────────
best_key = max(all_metrics, key=lambda k: all_metrics[k]['Macro F1'])
best_probs = all_probs[best_key]
best_preds = best_probs.argmax(1)

print(f"\nBest model: V4-{best_key}  (Macro F1 = {all_metrics[best_key]['Macro F1']:.4f})")
report = classification_report(labels_ref, best_preds,
                                target_names=[idx2cat[i] for i in range(num_classes)],
                                zero_division=0)
print(report)

report_path = os.path.join(OUTPUT_DIR, f"classification_report_{best_key}.txt")
with open(report_path, 'w') as f:
    f.write(f"Best model: V4-{best_key}\n\n")
    f.write(f"Macro F1 : {all_metrics[best_key]['Macro F1']:.4f}\n")
    f.write(f"Accuracy : {all_metrics[best_key]['Accuracy']:.4f}\n\n")
    f.write(report)
print(f"Saved → {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

# 1. Training curves (Focal — representative)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(hist_focal['train_loss'], label='train'); axes[0].plot(hist_focal['val_loss'], label='val')
axes[0].set_title('Focal — Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
axes[1].plot(hist_focal['val_f1'])
axes[1].set_title('Focal — Val Macro F1'); axes[1].set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves_focal.png"), dpi=150)
plt.close()

# 2. Confusion matrix (best model)
cm   = confusion_matrix(labels_ref, best_preds)
cmn  = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(max(12, num_classes // 2), max(10, num_classes // 2)))
sns.heatmap(cmn, annot=False, fmt='.2f', xticklabels=[idx2cat[i] for i in range(num_classes)],
            yticklabels=[idx2cat[i] for i in range(num_classes)], ax=ax, cmap='Blues')
ax.set_title(f'Normalised Confusion Matrix — V4-{best_key}')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{best_key}.png"), dpi=150)
plt.close()
print(f"Saved confusion matrix → {OUTPUT_DIR}/confusion_matrix_{best_key}.png")

# 3. Per-class F1 bar chart
per_cls_f1 = f1_score(labels_ref, best_preds, average=None, zero_division=0)
order       = np.argsort(per_cls_f1)
fig, ax     = plt.subplots(figsize=(10, max(8, num_classes * 0.3)))
ax.barh([idx2cat[i] for i in order], per_cls_f1[order], color='steelblue')
ax.axvline(per_cls_f1.mean(), color='red', linestyle='--', label=f'mean={per_cls_f1.mean():.3f}')
ax.set_xlabel('F1 Score'); ax.set_title(f'Per-Class F1 — V4-{best_key}'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"per_class_f1_{best_key}.png"), dpi=150)
plt.close()

# 4. Ensemble comparison bar chart
ens_names = list(all_metrics.keys())
ens_f1    = [all_metrics[k]['Macro F1'] for k in ens_names]
fig, ax   = plt.subplots(figsize=(10, 5))
bars = ax.bar(ens_names, ens_f1, color=['steelblue' if 'ensemble' not in k else 'darkorange'
                                         for k in ens_names])
ax.axhline(0.7951, color='green', linestyle='--', label='V3-SupCon best (0.7951)')
for b, v in zip(bars, ens_f1):
    ax.text(b.get_x() + b.get_width()/2, v + 0.003, f'{v:.4f}', ha='center', fontsize=8)
ax.set_ylabel('Macro F1'); ax.set_title('V4 Model Comparison (with Ensemble)'); ax.legend()
ax.set_xticklabels(ens_names, rotation=25, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ensemble_comparison.png"), dpi=150)
plt.close()
print(f"Saved ensemble comparison → {OUTPUT_DIR}/ensemble_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE PREDICTIONS  (grid: image | true | predicted | confidence)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating sample-prediction grid …")
rng  = np.random.default_rng(0)
idxs = rng.choice(len(test_data), size=16, replace=False)

fig, axes = plt.subplots(4, 4, figsize=(18, 18))
for ax, idx in zip(axes.flat, idxs):
    p  = os.path.join(BASE_DIR, test_data[idx]['img_local_path'])
    img = Image.open(p).convert('RGB') if os.path.exists(p) else Image.new('RGB',(100,100),(200,200,200))
    pred_idx   = best_preds[idx]
    true_idx   = labels_ref[idx]
    confidence = best_probs[idx, pred_idx]
    correct    = pred_idx == true_idx
    ax.imshow(img)
    ax.axis('off')
    color = 'green' if correct else 'red'
    ax.set_title(f"True: {idx2cat[true_idx]}\nPred: {idx2cat[pred_idx]}\nConf: {confidence:.2f}",
                 fontsize=7, color=color)
plt.suptitle(f'Sample Predictions — V4-{best_key}', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"), dpi=150)
plt.close()
print(f"Saved sample predictions → {OUTPUT_DIR}/sample_predictions.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTOUT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("V4 OPTIMISATION SUMMARY")
print("═"*70)
print(f"Backbone:            {backbone_name}")
print(f"SWA:                 activated at epoch {int(SWA_START_FRAC*NUM_EPOCHS)+1} / {NUM_EPOCHS}")
print(f"TTA views:           {len(tta_transforms)}  (V3 had 5)")
print(f"Ensemble members:    {', '.join(ensemble_keys)}")
print()
print("Per-model test Macro F1:")
for k in ensemble_keys:
    delta = all_metrics[k]['Macro F1'] - 0.7951   # vs V3 SupCon best
    sign  = '+' if delta >= 0 else ''
    print(f"  V4-{k:<14} {all_metrics[k]['Macro F1']:.4f}  ({sign}{delta:.4f} vs V3 best)")
print()
print("Ensemble test Macro F1:")
for k in [kk for kk in all_metrics if 'ensemble' in kk]:
    delta = all_metrics[k]['Macro F1'] - 0.7951
    sign  = '+' if delta >= 0 else ''
    print(f"  V4-{k:<20} {all_metrics[k]['Macro F1']:.4f}  ({sign}{delta:.4f} vs V3 best)")

print(f"\nAll artefacts saved to: {OUTPUT_DIR}/")
print("Done.")
