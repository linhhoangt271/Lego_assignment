# Architecture & Technical Details

## System Overview

```
Raw Images (minifigs.json metadata)
    ↓
Data Pipeline (stratified split, augmentation)
    ↓
Model Architecture (EfficientNet backbone)
    ↓
Loss Functions (Focal, ArcFace, SupCon)
    ↓
Training Loop (with early stopping, cosine annealing)
    ↓
Hyperparameter Tuning (Optuna + 3-Fold CV)
    ↓
Evaluation (F1, precision, recall, confusion matrix)
```

## Model Architecture

### EfficientNet Backbone Selection

| Version | Backbone | Input | Parameters | Memory | Accuracy | Selection |
|---------|----------|-------|-----------|--------|----------|-----------|
| Baseline | B0 | 128×128 | 5.3M | ~500MB | Baseline | Quick baseline |
| Option B | B0 | 128×128 | 5.3M | ~500MB | 68%+ | Initial improvement |
| V2 | B0 | 128×128 | 5.3M | ~500MB | ~70% | Enhanced training |
| **V3** | **B4** | **380×380** | **19M** | **~4GB** | **79.5%** | Production |

### Model Head Design

```python
EfficientNet-B4 (pretrained ImageNet)
    ↓ [1792 frozen features from avg_pool]
    ↓
Classification Head:
├── Linear (1792 → 512)      # Embedding layer
├── Dropout (0.3)
├── BatchNorm
└── Linear (512 → num_classes)
```

**Feature extraction**: Only use frozen convolutional layers, train only head for transfer learning stability.

## Loss Functions

### 1. Focal Loss
**Purpose**: Handle extreme class imbalance
```
FL(p_t) = -α_t(1-p_t)^γ log(p_t)
```
- **α_t**: Class weighting (0.25-0.75)
- **γ**: Focusing parameter (default: 2.0)
- **Effect**: Down-weights easy examples, focuses on hard negatives

**When to use**: Binary/few-class problems with extreme imbalance
**V3 Results**: F1=0.7163 (most stable across folds)

### 2. ArcFace Loss
**Purpose**: Learn discriminative embeddings via angular margin
```
L = log(exp(s(cos(θ + m))) / (exp(s(cos(θ + m))) + Σ exp(s*cos(θ_j))))
```
- **s**: Temperature scale (30)
- **m**: Angular margin (0.5)
- **Effect**: Maximizes angular distance between classes

**When to use**: High-dimensional classification, metric learning
**V3 Results**: F1=0.7136 (high variance across folds, 0.7300 best)

### 3. Supervised Contrastive Loss (SupCon)
**Purpose**: Learn semantically similar representations via contrastive pairs
```
L_i = -log(exp(sim(z_i, z_i^+) / τ) / Σ_p exp(sim(z_i, z_p) / τ))
```
- **τ**: Temperature (0.07)
- **z_i, z_i^+**: Anchor and positive pair embeddings
- **Effect**: Pulls same-class samples together, pushes different classes apart

**Combined with**: Cross-entropy loss for classification
**V3 Results**: F1=0.7207 ⭐ **BEST**, F1=0.7432 single best fold

**Recommendation**: Use SupCon for this task due to best validation performance.

## Data Pipeline

### 1. Input Data Structure
```python
minifigs.json = [
    {
        "filename": "image_001.jpg",
        "category": "Star Wars",
        "town": "...subcategory..."
    },
    ...
]

images/
├── image_001.jpg
├── image_002.jpg
└── ... (millions of images)
```

### 2. Data Loading & Splitting

```python
# Stratified split (maintain class distribution)
train_data (70%) → augmentation + weighted sampling
val_data (15%)   → minimal augmentation
test_data (15%)  → no augmentation (only TTA at inference)

# Special handling: Town categories
# Split Town by subcategory to avoid severe imbalance
```

### 3. Augmentation Pipeline

#### Training Augmentation (V3)
```
Input Image → [Geometric]  → [Pixel]    → [Mixing]     → [Output]
           → Rotate       → Brightness → CutMix       → 380×380
           → Flip         → Saturation → MixUp        
           → Affine       → Hue        → RandomErase
           → Crop         → Contrast   
```

**Parameters**:
- **CutMix**: probability=0.3, α=1.0 (creates diverse patches)
- **MixUp**: probability=0.3, α=0.2 (soft label blending)
- **RandAugment**: N=2, M=8 (random augmentation chains)
- **RandomErasing**: probability=0.2, scale=(0.02, 0.33)

#### Validation/Test
```
Minimal: Normalization only
Test-Time Augmentation (TTA): 5 transforms, average predictions
```

### 4. Handling Class Imbalance

**Method 1: Weighted Random Sampling**
```python
# Calculate sqrt of class frequencies
weights = np.sqrt(class_counts)  
sampler = WeightedRandomSampler(weights)
```

**Method 2: Synthetic Minority Generation**
```python
# For classes < MIN_SAMPLES_TARGET (50):
1. Use YOLO to detect minifigures
2. Crop center region
3. Randomly transform (rotate, flip, warp)
4. Generate synthetic samples
5. Add to training set
```

**Method 3: Label Smoothing**
```python
# Smooth hard targets: [0,1,0] → [0.05, 0.9, 0.05]
smooth_label = label * (1 - α) + α / num_classes
```

## Training Loop Details

### Hyperparameter Configuration (V3 Defaults)

```python
# Architecture
IMG_SIZE = 380
BATCH_SIZE = 16
NUM_WORKERS = 4

# Optimization
LEARNING_RATE = 3e-4
OPTIMIZER = AdamW(betas=(0.9, 0.999), weight_decay=1e-4)
SCHEDULER = CosineAnnealingWarmRestarts(T_0=5, T_mult=2)

# Regularization
LABEL_SMOOTHING = 0.1
DROPOUT = 0.3
PATIENCE = 5  # Early stopping

# Loss functions
FOCAL_GAMMA = 2.0
ARCFACE_MARGIN = 0.5
SUPCON_TEMPERATURE = 0.07

# Data
MIN_SAMPLES_TARGET = 50  # Synthetic augmentation threshold
TTA_TRANSFORMS = 5  # Test-time augmentation variants
```

### Training Process

```python
for epoch in range(num_epochs):
    # Training phase
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        embeddings = model.get_embeddings(images)
        
        # Loss computation
        loss_ce = cross_entropy(logits, labels)
        loss_focal = focal_loss(logits, labels)
        loss_contrastive = supcon_loss(embeddings, labels)
        
        total_loss = loss_ce + loss_focal + loss_contrastive
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation phase
    val_metrics = evaluate(model, val_loader)
    
    # Scheduler step
    scheduler.step()
    
    # Early stopping
    if val_loss > best_val_loss + patience_threshold:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            break
```

## Hyperparameter Tuning with Optuna

### Search Space (V3 HP Optimization)

```python
# Tuned parameters (per loss variant)
learning_rate = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
scheduler_t0 = trial.suggest_int('t0', 2, 10)
warmup_epochs = trial.suggest_int('warmup', 0, 3)

# Fixed parameters (not tuned)
batch_size = 16
weight_decay = 1e-4
label_smoothing = 0.1
```

### Optimization Strategy

**Algorithm**: Bayesian Optimization (TPE Sampler)
- Starts with random sampling
- Builds probabilistic model of objective
- Suggests promising regions for next trials
- Pruning: Stops unpromising trials early (median-based)

**Results** (30 trials × 3 loss variants):

| Loss | Best Trials | Best LR | Best t0 | Best Val F1 |
|------|------------|---------|---------|------------|
| Focal | 2/10 | 2.90e-4 | 4 | 0.6833 |
| ArcFace | 1/10 | 4.31e-4 | 3 | 0.6729 |
| SupCon | 5/10 | 3.71e-4 | 6 | 0.6984 |

### 3-Fold Cross-Validation

```python
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(data, labels)):
    # Train on fold
    model = train_with_best_hp(train_data[train_idx], epochs=20)
    
    # Evaluate on fold
    metrics[fold] = evaluate(model, val_data[val_idx])
    
    # Save best model for this fold
    torch.save(model.state_dict(), f'best_model_fold{fold}.pth')

# Report mean ± std
print(f"3-Fold CV: {metrics.mean()} ± {metrics.std()}")
```

**Results Summary**:
- **SupCon**: Mean F1=0.7207 ± 0.0167 (best)
- **Focal**: Mean F1=0.7163 ± 0.0056 (most stable)
- **ArcFace**: Mean F1=0.7136 ± 0.0202 (high variance)

## Evaluation Metrics

### Primary Metric: F1-Score (Weighted)
```python
F1 = 2 * (precision * recall) / (precision + recall)
# Weighted by class support to handle imbalance
```

### Secondary Metrics

| Metric | Formula | Usage |
|--------|---------|-------|
| Precision | TP / (TP + FP) | False positive cost |
| Recall | TP / (TP + FN) | False negative cost |
| Accuracy | (TP + TN) / Total | Overall correctness |
| Top-K Accuracy | % correct in top-K predictions | Ranking quality |
| Confusion Matrix | TP/FP/TN/FN per class pair | Confusion patterns |

### Confusion Analysis

```python
# Identify most confused class pairs
confusion = confusion_matrix(y_true, y_pred)
top_confusions = np.argsort(confusion.flatten())[-20:]
# These suggest candidates for merging (Option B approach)
```

## Performance Optimization

### GPU Acceleration
- Transfer to GPU: `model.to('cuda')`
- Mixed precision: `torch.cuda.amp.autocast()`
- Gradient checkpointing: Save memory for large batch sizes

### Memory Optimization
- Reduce batch size if OOM
- Use gradient accumulation for effective larger batches
- Pin memory: `DataLoader(pin_memory=True, num_workers=4)`

### Speed Optimization
- Multi-worker data loading: `num_workers=4`
- Prefetch batches: `persistent_workers=True`
- Compiled model (PyTorch 2.0): `model = torch.compile(model)`

## Reproducibility

### Fixed Seeds
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
```

### Logging
```python
# Save all configs
config = {
    'model': 'EfficientNet-B4',
    'loss': 'SupCon',
    'learning_rate': 3e-4,
    'epochs': 20,
    'batch_size': 16,
}
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

---

**Last Updated**: 2026-04-21
