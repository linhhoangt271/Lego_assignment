"""V3 training script — auto-generated from Models_Summary.ipynb"""


# ── cell 4 ──────────────────────────────────────────

import copy
import json
import math
import os
import random
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

BASE_DIR = Path.cwd()
JSON_PATH = BASE_DIR / 'minifigs.json'
IMG_DIR = BASE_DIR / 'images'

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    print(f'Using device: {DEVICE}')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)


# ── cell 6 ──────────────────────────────────────────

def load_minifig_records(json_path=JSON_PATH, base_dir=BASE_DIR):
    with open(json_path, encoding='utf-8') as f:
        raw = json.load(f)

    valid = []
    for row in raw:
        rel_path = row.get('img_local_path')
        if not rel_path:
            continue
        if (base_dir / rel_path).exists():
            valid.append(row)

    print(f'Total records: {len(raw):,}')
    print(f'Valid image records: {len(valid):,}')
    return valid

records = load_minifig_records()


# ── cell 8 ──────────────────────────────────────────

def split_town(subcategory):
    sub = str(subcategory).lower()
    if 'police' in sub:
        return 'Town - Police'
    if 'fire' in sub:
        return 'Town - Fire'
    if 'airport' in sub:
        return 'Town - Airport'
    if 'hospital' in sub or 'rescue' in sub:
        return 'Town - Rescue'
    if 'space' in sub:
        return 'Town - Space'
    if 'race' in sub or 'stuntz' in sub:
        return 'Town - Racing'
    if 'coast guard' in sub:
        return 'Town - Coast Guard'
    if 'construction' in sub:
        return 'Town - Construction'
    if any(x in sub for x in ['arctic', 'jungle', 'volcano', 'ocean', 'deep sea', 'exploration']):
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
    'Dimensions': 'Collectible & Special', 'Vidiyo': 'Collectible & Special', 'Games': 'Collectible & Special',
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

def add_labels(records, merge=False, town_split=False):
    labelled = copy.deepcopy(records)
    for row in labelled:
        category = row['category']
        if town_split and category == 'Town':
            label = split_town(row.get('subcategory', ''))
        elif merge:
            label = MERGE_MAP.get(category, category)
        else:
            label = category
        row['model_label'] = label
    return labelled


# ── cell 10 ──────────────────────────────────────────

def make_splits(labelled_records, label_key='model_label', test_size=0.15, val_size=0.15):
    labels_text = [row[label_key] for row in labelled_records]
    label_names = sorted(set(labels_text))
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}
    idx_to_label = {idx: name for name, idx in label_to_idx.items()}
    labels = [label_to_idx[name] for name in labels_text]

    counts = Counter(labels)
    small_idx = [i for i, label in enumerate(labels) if counts[label] < 7]
    big_idx = [i for i, label in enumerate(labels) if counts[label] >= 7]

    big_records = [labelled_records[i] for i in big_idx]
    big_labels = [labels[i] for i in big_idx]

    train_records, temp_records, train_labels, temp_labels = train_test_split(
        big_records,
        big_labels,
        test_size=test_size + val_size,
        random_state=SEED,
        stratify=big_labels,
    )
    relative_test = test_size / (test_size + val_size)
    val_records, test_records, val_labels, test_labels = train_test_split(
        temp_records,
        temp_labels,
        test_size=relative_test,
        random_state=SEED,
        stratify=temp_labels,
    )

    train_records += [labelled_records[i] for i in small_idx]
    train_labels += [labels[i] for i in small_idx]

    print(f'Classes: {len(label_names)}')
    print(f'Train: {len(train_records):,} | Val: {len(val_records):,} | Test: {len(test_records):,}')
    return {
        'train_records': train_records,
        'val_records': val_records,
        'test_records': test_records,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'num_classes': len(label_names),
    }


# ── cell 12 ──────────────────────────────────────────

class PadToSquare:
    def __call__(self, img):
        max_side = max(img.size)
        return ImageOps.pad(img, (max_side, max_side), color=(255, 255, 255))

class MinifigDataset(Dataset):
    def __init__(self, records, labels, transform=None, fallback_size=224):
        self.records = records
        self.labels = labels
        self.transform = transform
        self.fallback_size = fallback_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path = BASE_DIR / self.records[idx]['img_local_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (self.fallback_size, self.fallback_size), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def build_transforms(img_size, preserve_aspect=False, strong=False):
    resize_prefix = [PadToSquare()] if preserve_aspect else []
    train_ops = resize_prefix + [transforms.Resize((img_size, img_size))]
    if strong:
        train_ops += [
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        ]
    else:
        train_ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    eval_ops = resize_prefix + [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)

def make_loaders(split, img_size, batch_size, num_workers, preserve_aspect=False, strong_aug=False):
    train_tf, eval_tf = build_transforms(img_size, preserve_aspect, strong_aug)
    train_ds = MinifigDataset(split['train_records'], split['train_labels'], train_tf, img_size)
    val_ds = MinifigDataset(split['val_records'], split['val_labels'], eval_tf, img_size)
    test_ds = MinifigDataset(split['test_records'], split['test_labels'], eval_tf, img_size)

    label_counts = Counter(split['train_labels'])
    weights = [1.0 / label_counts[label] for label in split['train_labels']]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


# ── cell 14 ──────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        ce = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return ((1 - pt).pow(self.gamma) * ce).mean()

class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1)
        return (one_hot * phi + (1 - one_hot) * cosine) * self.scale

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        logits_mask = 1 - torch.eye(features.shape[0], device=features.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        denom = mask.sum(1).clamp(min=1)
        return -(mask * log_prob).sum(1).div(denom).mean()

def apply_mixup(images, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam

def class_weights_from_labels(labels, num_classes, sqrt=True):
    counts = Counter(labels)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for idx in range(num_classes):
        count = max(counts.get(idx, 1), 1)
        weights[idx] = 1.0 / (math.sqrt(count) if sqrt else count)
    weights = weights / weights.sum() * num_classes
    return weights.to(DEVICE)


# ── cell 16 ──────────────────────────────────────────

def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ('model', 'state_dict', 'model_state_dict'):
            if key in checkpoint and hasattr(checkpoint[key], 'keys'):
                return checkpoint[key]
    return checkpoint

def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        normalized[key] = value
    return normalized

def load_checkpoint_if_available(model, checkpoint_path, strict=True):
    if not checkpoint_path:
        return False
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = BASE_DIR / checkpoint_path
    if not checkpoint_path.exists():
        print(f'No checkpoint found at {checkpoint_path}. Training will run.')
        return False

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = normalize_state_dict_keys(checkpoint_state_dict(checkpoint))
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        print(f'Checkpoint is not compatible with this model: {checkpoint_path}')
        print(exc)
        print('Training will run instead.')
        return False

    print(f'Loaded checkpoint: {checkpoint_path}')
    return True

def train_classifier(model, loaders, criterion, optimizer, scheduler=None, epochs=10, output_dir='results', use_mixup=False):
    output_dir = BASE_DIR / output_dir
    output_dir.mkdir(exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')
    best_val_f1 = -1.0
    history = []

    for epoch in range(epochs):
        start = time.time()
        model.train()
        train_loss = 0.0
        train_preds = []
        train_true = []

        for images, labels in loaders['train']:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
                arcface_model = hasattr(model, 'arcface') and model.arcface is not None
                if use_mixup:
                    images, y_a, y_b, lam = apply_mixup(images, labels)
                    outputs_a = model(images, y_a) if arcface_model else model(images)
                    outputs_b = model(images, y_b) if arcface_model else outputs_a
                    loss = lam * criterion(outputs_a, y_a) + (1 - lam) * criterion(outputs_b, y_b)
                    outputs = outputs_a
                else:
                    outputs = model(images, labels) if arcface_model else model(images)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            train_preds.extend(outputs.argmax(1).detach().cpu().numpy())
            train_true.extend(labels.detach().cpu().numpy())

        val_metrics, _, _, _ = evaluate_classifier(model, loaders['val'], loaders.get('num_classes'))
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['Macro F1'])
            else:
                scheduler.step()

        train_f1 = f1_score(train_true, train_preds, average='macro', zero_division=0)
        epoch_row = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(loaders['train'].dataset),
            'train_macro_f1': train_f1,
            'val_macro_f1': val_metrics['Macro F1'],
            'val_accuracy': val_metrics['Accuracy'],
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"loss {epoch_row['train_loss']:.4f} | "
            f"train F1 {train_f1:.4f} | "
            f"val F1 {val_metrics['Macro F1']:.4f} | "
            f"val acc {val_metrics['Accuracy']:.4f} | "
            f"{time.time() - start:.0f}s"
        )

        if val_metrics['Macro F1'] > best_val_f1:
            best_val_f1 = val_metrics['Macro F1']
            torch.save(model.state_dict(), output_dir / 'best_model.pth')

    return history

def evaluate_classifier(model, loader, num_classes=None, idx_to_label=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    if num_classes is None:
        num_classes = all_probs.shape[1]

    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Macro F1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'Weighted F1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'Macro Precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'Macro Recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
    }
    if num_classes >= 3:
        metrics['Top-3 Accuracy'] = top_k_accuracy_score(all_labels, all_probs, k=3, labels=range(num_classes))
    if num_classes >= 5:
        metrics['Top-5 Accuracy'] = top_k_accuracy_score(all_labels, all_probs, k=5, labels=range(num_classes))

    for name, value in metrics.items():
        print(f'{name}: {value:.4f}')

    if idx_to_label is not None:
        labels = sorted(set(all_labels) | set(all_preds))
        target_names = [idx_to_label[idx] for idx in labels]
        print(classification_report(all_labels, all_preds, labels=labels, target_names=target_names, zero_division=0))

    return metrics, all_preds, all_labels, all_probs

def load_best_and_evaluate(model, split, loaders, output_dir):
    model.load_state_dict(torch.load(BASE_DIR / output_dir / 'best_model.pth', map_location=DEVICE))
    return evaluate_classifier(model, loaders['test'], split['num_classes'], split['idx_to_label'])


# ── cell 45 ──────────────────────────────────────────

v3_cfg = {
    'img_size': 380,
    'batch_size': 16,
    'epochs': 20,
    'lr': 3e-4,
    'num_workers': 4 if DEVICE.type == 'cuda' else 0,
    'output_dir': 'optionB_v3_results',
    'checkpoint_path': 'optionB_v3_results/best_model.pth',
    'label_smoothing': 0.1,
    'focal_gamma': 2.0,
    'variant': 'focal',  # choose: focal, arcface, supcon
}


# ── cell 47 ──────────────────────────────────────────

v3_records = add_labels(records, merge=True, town_split=True)
v3_split = make_splits(v3_records)


# ── cell 48 ──────────────────────────────────────────

# Save label mapping for the web app (app.py loads this file)
_v3_label_dir = Path(v3_cfg['output_dir'])
_v3_label_dir.mkdir(exist_ok=True)
with open(_v3_label_dir / 'labels.json', 'w', encoding='utf-8') as _f:
    json.dump({str(k): v for k, v in v3_split['idx_to_label'].items()}, _f, ensure_ascii=False, indent=2)
print(f"Saved {v3_split['num_classes']} class labels to {_v3_label_dir / 'labels.json'}")


# ── cell 50 ──────────────────────────────────────────

v3_loaders = make_loaders(
    v3_split,
    img_size=v3_cfg['img_size'],
    batch_size=v3_cfg['batch_size'],
    num_workers=v3_cfg['num_workers'],
    preserve_aspect=True,
    strong_aug=True,
)
v3_loaders['num_classes'] = v3_split['num_classes']


# ── cell 52 ──────────────────────────────────────────

class EmbeddingClassifier(nn.Module):
    def __init__(self, backbone, in_features, num_classes, embedding_dim=512, use_arcface=False):
        super().__init__()
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.arcface = ArcFaceHead(embedding_dim, num_classes) if use_arcface else None

    def forward(self, x, labels=None, return_embedding=False):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        if self.arcface is not None:
            logits = self.arcface(embeddings, labels)
        else:
            logits = self.classifier(embeddings)
        if return_embedding:
            return logits, F.normalize(embeddings, dim=1)
        return logits

def build_v3_model(num_classes, variant='focal'):
    base = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    in_features = base.classifier[1].in_features
    backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten())
    return EmbeddingClassifier(backbone, in_features, num_classes, use_arcface=(variant == 'arcface')).to(DEVICE)

v3_model = build_v3_model(v3_split['num_classes'], v3_cfg['variant'])
v3_weights = class_weights_from_labels(v3_split['train_labels'], v3_split['num_classes'], sqrt=True)
if v3_cfg['variant'] == 'focal':
    v3_criterion = FocalLoss(weight=v3_weights, gamma=v3_cfg['focal_gamma'], label_smoothing=v3_cfg['label_smoothing'])
elif v3_cfg['variant'] == 'arcface':
    v3_criterion = nn.CrossEntropyLoss(weight=v3_weights, label_smoothing=v3_cfg['label_smoothing'])
else:
    v3_criterion = nn.CrossEntropyLoss(weight=v3_weights, label_smoothing=v3_cfg['label_smoothing'])
    v3_supcon = SupConLoss()

v3_optimizer = torch.optim.AdamW(v3_model.parameters(), lr=v3_cfg['lr'], weight_decay=1e-4)
v3_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(v3_optimizer, T_max=v3_cfg['epochs'])


# ── cell 54 ──────────────────────────────────────────

v3_loaded = load_checkpoint_if_available(v3_model, v3_cfg['checkpoint_path'])
if v3_loaded:
    v3_history = []
    print('Skipped V3 training because checkpoint is loaded.')
else:
    # For focal and arcface variants, use the shared trainer.
    # For SupCon, this cell trains CE + supervised contrastive loss together.
    if v3_cfg['variant'] != 'supcon':
        v3_history = train_classifier(
            v3_model,
            v3_loaders,
            v3_criterion,
            v3_optimizer,
            scheduler=v3_scheduler,
            epochs=v3_cfg['epochs'],
            output_dir=v3_cfg['output_dir'],
            use_mixup=True,
        )
    else:
        output_dir = BASE_DIR / v3_cfg['output_dir']
        output_dir.mkdir(exist_ok=True)
        scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')
        best_f1 = -1.0
        v3_history = []
        for epoch in range(v3_cfg['epochs']):
            v3_model.train()
            for images, labels in v3_loaders['train']:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                v3_optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
                    logits, embeddings = v3_model(images, return_embedding=True)
                    loss = v3_criterion(logits, labels) + 0.2 * v3_supcon(embeddings, labels)
                scaler.scale(loss).backward()
                scaler.step(v3_optimizer)
                scaler.update()
            v3_scheduler.step()
            metrics, _, _, _ = evaluate_classifier(v3_model, v3_loaders['val'], v3_split['num_classes'])
            v3_history.append({'epoch': epoch + 1, **metrics})
            if metrics['Macro F1'] > best_f1:
                best_f1 = metrics['Macro F1']
                torch.save(v3_model.state_dict(), output_dir / 'best_model.pth')
