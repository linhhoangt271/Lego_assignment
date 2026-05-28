# LEGO Minifigure Classification

This repository keeps the model workflow in one GPU-ready notebook for easier GitHub management:

- `Models_Summary.ipynb` - step-based notebook for shared data loading, training, and evaluation
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
| Option B V3 | 37 | EfficientNet-B4 | Larger images, merged labels, Focal/ArcFace/SupCon variants |
| Option B V4 | 37 | ConvNeXt-Small or EfficientNet-B4 | V3-style labels with SWA-ready ConvNeXt/EfficientNet training |

## Reusing Local Checkpoints

The notebook checks each model config for `checkpoint_path` before training. If a compatible local `.pth` exists, the training cell loads it and skips retraining. Checkpoints stay ignored by Git, so keep them locally in the matching result folder.

## GitHub Cleanup Policy

Do not commit generated files:

- `images/`
- `minifigs.json`
- `*_results/`
- `*.pth` and `*.pt`
- training logs and cache folders

These are covered by `.gitignore` so GitHub stays focused on the notebook, documentation, and dependency file.
