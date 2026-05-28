# LEGO Minifigure Classification

This repository keeps the model workflow in one GPU-ready notebook for easier GitHub management:

- `Models_Summary.ipynb` - consolidated notebook for training and evaluation
- `requirements.txt` - Python dependencies
- `docs/` - supporting project notes and results summaries

Local data, images, checkpoints, and generated result folders are intentionally ignored by Git.

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook Models_Summary.ipynb
```

Expected local files:

```text
big_data_assignment_2/
├── Models_Summary.ipynb
├── minifigs.json      # ignored by Git
└── images/            # ignored by Git
```

## Models In The Notebook

| Model | Classes | Backbone | Purpose |
|---|---:|---|---|
| Baseline | 122 | EfficientNet-B0 | Raw-category reference baseline |
| Option B V2 | 37 | EfficientNet-B2 | Town split, stronger augmentation, TTA |
| Option B V3 | 28/37 | EfficientNet-B4 | YOLO cropping, synthetic minority augmentation, Focal/ArcFace/SupCon variants |
| V3 HP Tuning | 28/37 | EfficientNet-B4 | Optuna search plus 3-fold cross-validation |
| Option B V4 | 28 | ConvNeXt-Small or EfficientNet-B4 | Latest optimized model with SWA and weighted ensemble |
| ResNet50 Top-20 | 20 | ResNet50 | Small reference baseline on common classes |
| Evaluation Utilities | Multiple | Saved checkpoints | Compare saved model outputs |

## GitHub Cleanup Policy

Do not commit generated files:

- `images/`
- `minifigs.json`
- `*_results/`
- `*.pth` and `*.pt`
- training logs and cache folders

These are covered by `.gitignore` so GitHub stays focused on the notebook, documentation, and dependency file.
