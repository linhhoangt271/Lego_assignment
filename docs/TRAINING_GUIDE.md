# Training Guide

All model training code now lives in `Models_Summary.ipynb`. Run the shared setup/data cells first, then run one model section at a time:

- Baseline: EfficientNet-B0 on raw categories
- Option B V2: EfficientNet-B2 with merged labels, Town split, stronger augmentation, and TTA-ready evaluation
- Option B V3: EfficientNet-B4 with Focal/ArcFace/SupCon variants
- Option B V4: ConvNeXt-Small or EfficientNet-B4 with SWA-ready training

Generated checkpoints and result folders should stay local and untracked.
