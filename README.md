# LEGO Minifigure Classification Pipeline

A comprehensive deep learning pipeline for classifying LEGO minifigures by category using transfer learning, advanced data augmentation, and hyperparameter optimization.

## 🎯 Project Overview

This project implements multiple progressive training approaches for LEGO minifigure classification:

- **Baseline**: EfficientNet-B0 with 122 raw categories
- **Option B**: Category merging based on domain knowledge + confusion analysis
- **Option B V2**: Enhanced augmentation and optimization
- **Option B V3**: Advanced techniques (YOLO cropping, EfficientNet-B4, multiple loss functions)
- **V3 HP-Tuned**: Hyperparameter optimization + 3-fold cross-validation (best: SupCon Loss, F1=0.7207)

## 📊 Best Results

| Model | F1 Score | Architecture | Key Techniques |
|-------|----------|--------------|-----------------|
| **V3 HP-Tuned (SupCon)** | **0.7207** | EfficientNet-B4 | SupCon Loss, Optuna Tuning, 3-Fold CV |
| V3 Baseline | 0.7951 | EfficientNet-B4 | Focal/ArcFace/SupCon Loss variants |
| Option B | 0.68+ | EfficientNet-B0 | Category merging |
| Baseline | 0.62+ | EfficientNet-B0 | Raw categories |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd big_data_assignment_2

# Install dependencies
pip install -r requirements.txt

# Optional: YOLO for minifigure cropping
pip install ultralytics
```

### 2. Prepare Data

```bash
# Expected structure:
# - images/          (2.1G minifigure images)
# - minifigs.json    (11M metadata: filename, category, town)
```

### 3. Run Training Pipeline

```bash
# Option A: Quick baseline test
python src/training/baseline_train.py

# Option B: Full V3 with HP tuning (29 hours)
python src/training/optionB_v3_hpopt.py

# Option B V3 without HP tuning (2-3 hours)
python src/training/optionB_v3.py

# Compare all models
python src/training/compare_all_models.py
```

## 📁 Directory Structure

```
big_data_assignment_2/
├── src/
│   ├── models/              # Model definitions
│   │   ├── efficientnet.py
│   │   ├── losses.py        # Focal, ArcFace, SupCon
│   │   └── augmentation.py
│   ├── training/            # Training scripts
│   │   ├── baseline_train.py
│   │   ├── optionB_train.py
│   │   ├── optionB_v2.py
│   │   ├── optionB_v3.py
│   │   ├── optionB_v3_hpopt.py  # HP tuning + CV
│   │   └── evaluate_all.py
│   ├── data/                # Data loading & preprocessing
│   │   └── dataset.py
│   └── utils/               # Utilities
│       ├── metrics.py
│       └── visualization.py
├── notebooks/
│   ├── training_dashboard.ipynb
│   ├── all_results_dashboard.ipynb
│   └── demo.ipynb
├── docs/
│   ├── ARCHITECTURE.md      # Detailed architecture
│   ├── TRAINING_GUIDE.md    # Training instructions
│   └── RESULTS.md           # Experiment results
├── data/
│   ├── images/              # (NOT in git, ~2.1G)
│   └── minifigs.json        # (NOT in git, ~11M)
├── results/                 # (NOT in git, model outputs)
└── README.md
```

## 🔬 Key Features

### 1. **Three Loss Functions**
- **Focal Loss**: Handles class imbalance by focusing on hard negatives
- **ArcFace Loss**: Metric learning for discriminative embeddings
- **Supervised Contrastive (SupCon)**: Contrastive learning with classification

### 2. **Data Augmentation**
- CutMix, MixUp, RandomErasing
- RandAugment, AutoAugment
- Synthetic minority sample generation via YOLO-based cropping

### 3. **Training Techniques**
- Cosine annealing with warm restarts
- Early stopping with patience
- Label smoothing (0.1)
- Weighted random sampling for class imbalance
- Test-time augmentation (TTA)

### 4. **Hyperparameter Optimization**
- Optuna with TPE sampler
- Median-based pruning
- 30 trials × 3 loss variants
- 3-fold cross-validation

## 📈 Training Pipeline Details

### Phase 1: Baseline (baseline_train.py)
- EfficientNet-B0, 122 classes, 10 epochs
- Output: confusion matrix, classification report

### Phase 2: Category Merging (optionB_train.py)
- Domain-guided merging (NINJAGO, Friends, etc.)
- Reduces class count, improves per-class balance
- EfficientNet-B0, 15 epochs

### Phase 3: Advanced V2/V3 (optionB_v3.py)
- YOLO-based minifigure detection + cropping
- EfficientNet-B4, 380px input
- 3 loss variants, 20 epochs
- Synthetic augmentation for minorities

### Phase 4: HP Tuning (optionB_v3_hpopt.py)
- **Stage 1**: Optuna searches 30 hyperparameter configs
- **Stage 2**: 3-fold CV validates best configs
- **Runtime**: ~29 hours, validates findings

## 🎮 Running Specific Scenarios

### Quick Test (5 min)
```bash
python src/training/baseline_train.py  # 10 epochs
```

### Full Baseline Evaluation (30 min)
```bash
python src/training/optionB_v3.py --epochs 10
```

### Production Training (2-3 hours)
```bash
python src/training/optionB_v3.py --epochs 20
```

### Research HP Tuning (29 hours)
```bash
python src/training/optionB_v3_hpopt.py  # 30 trials + 3-fold CV
```

### Compare All Models
```bash
python src/training/compare_all_models.py
```

## 📊 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir=runs/
```

### Interactive Dashboard
```bash
jupyter notebook notebooks/training_dashboard.ipynb
```

## 🔧 Customization

Edit these in the script headers:

```python
IMG_SIZE = 380              # Input image size
BATCH_SIZE = 16             # Larger = faster but more memory
NUM_EPOCHS = 20             # Training epochs
LR = 3e-4                   # Learning rate
LABEL_SMOOTHING = 0.1       # Smoothing strength
MIN_SAMPLES_TARGET = 50     # Min samples per class (synthetic)
```

## 📋 Output Files

After training, check these directories:

```
optionB_v3_results/
├── best_model.pth          # Best checkpoint
├── classification_report.txt
├── confusion_matrix.png
└── training_curves.png

optionB_v3_hpopt_results/
├── hp_search_results.txt
├── cv_results.txt
├── best_model_focal_fold*.pth
└── final_comparison.txt
```

## 🤝 Contributing

When continuing this work:

1. **Create a feature branch**: `git checkout -b feature/your-feature`
2. **Document changes**: Update TRAINING_STATUS.md with results
3. **Save models**: Use consistent naming (e.g., `best_model_v4.pth`)
4. **Commit often**: `git commit -m "Clear, descriptive message"`
5. **Update results**: Add findings to docs/RESULTS.md

## 📚 Key References

- **EfficientNet**: Tan & Le (2019)
- **Focal Loss**: Lin et al. (2017)
- **ArcFace**: Deng et al. (2018)
- **Supervised Contrastive**: Khosla et al. (2020)
- **Optuna**: Akiba et al. (2019)

## ⚠️ Important Notes

- **Data not in git**: Images (2.1G) and minifigs.json (11M) must be fetched separately
- **Large checkpoints**: Model weights (72MB each) are not committed
- **GPU recommended**: Training V3 on CPU takes significantly longer
- **Memory**: B4 model + batch size 16 requires ~8GB VRAM

## 📞 Questions?

Refer to `docs/TRAINING_GUIDE.md` for detailed instructions or check `TRAINING_STATUS.md` for current progress.

---

**Last Updated**: 2026-04-21  
**Status**: ✅ Complete (V3 HP-Tuned)
