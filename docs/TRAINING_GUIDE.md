# Training Guide & Workflows

## Prerequisites

```bash
# 1. Clone and setup
git clone <repo-url>
cd big_data_assignment_2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}')"

# 4. Get data (NOT in git)
# - Download images/ folder (2.1G)
# - Download minifigs.json (11M)
# Place in project root directory
```

## Quick Start (5 minutes)

### Run baseline on subset for testing
```bash
python src/training/baseline_train.py
```

**What it does:**
- Loads 122 LEGO minifigure categories
- Trains EfficientNet-B0 for 10 epochs
- Outputs: confusion_matrix.png, classification_report.txt
- Expected time: 5-10 minutes (CPU), 1-2 minutes (GPU)

## Full Training Workflows

### Workflow 1: Quick Benchmark (1-2 hours)

```bash
# Train V3 with fewer epochs to test pipeline
python src/training/optionB_v3.py \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 3e-4
```

**Expected output:**
- baseline_results/classification_report.txt (F1 ≈ 0.70-0.75)
- Training curves + confusion matrix
- Total time: ~1-2 hours

### Workflow 2: Production Training (2-3 hours)

```bash
# Full V3 training with all techniques
python src/training/optionB_v3.py \
  --epochs 20 \
  --batch_size 16 \
  --learning_rate 3e-4 \
  --use_synthetic_augmentation \
  --use_tta
```

**Expected output:**
- optionB_v3_results/best_model.pth (checkpoint)
- Classification report with F1 ≈ 0.75-0.80
- Confusion matrix + training curves
- Total time: ~2-3 hours (GPU)

**To monitor progress:**
```bash
# In another terminal, watch directory
watch -n 5 'tail -20 optionB_v3_results/training.log'

# Or use TensorBoard
tensorboard --logdir=optionB_v3_results/
```

### Workflow 3: Research HP Tuning (29 hours)

**⚠️ Long-running task. Run on background/cluster.**

```bash
# Full hyperparameter optimization + 3-fold CV
nohup python src/training/optionB_v3_hpopt.py > hpopt.log 2>&1 &

# Check progress
tail -f hpopt.log

# Monitor output directory
watch -n 30 'du -sh optionB_v3_hpopt_results/'
```

**Pipeline:**
1. **Phase 1** (~20 hours): 30 Optuna trials (10 per loss variant)
2. **Phase 2** (~6.5 hours): 3-fold CV with best hyperparameters
3. **Phase 3** (~2 min): Final reporting

**Expected output:**
- optionB_v3_hpopt_results/hp_search_results.txt (best HP per variant)
- optionB_v3_hpopt_results/cv_results.txt (3-fold metrics)
- optionB_v3_hpopt_results/best_model_focal_fold*.pth (saved checkpoints)
- Final F1 ≈ 0.72 (SupCon best)

### Workflow 4: Compare All Approaches (1 hour)

```bash
# Run all models and create comparison report
python src/training/compare_all_models.py

# Outputs:
# - evaluation_results/comparison_table.txt
# - evaluation_results/model_comparison.png
# - evaluation_results/convergence_curves.png
```

## Parameter Tuning Guide

### When to adjust learning rate

| Symptom | Likely Issue | Fix |
|---------|-------------|-----|
| Training stuck at high loss | LR too low | Increase to 1e-3 |
| Loss oscillates wildly | LR too high | Decrease to 1e-5 |
| Validation improves then plateaus | LR too high | Decrease to 5e-4 |
| Slow convergence | LR too low | Increase to 5e-4 |

**Default ranges:**
- Conservative: 1e-5 to 1e-4 (slow but stable)
- Standard: 3e-4 to 1e-3 (typical)
- Aggressive: 5e-3 to 1e-2 (risky, for small batches)

### When to adjust batch size

| Symptom | Likely Issue | Fix |
|---------|-------------|-----|
| CUDA OOM error | Batch too large | Reduce to 8 or 4 |
| Slow training | Batch too small | Increase to 32 |
| Noisy gradients | Batch too small | Increase to 32 |
| Poor generalization | Batch too large | Reduce to 8 |

**Memory usage (EfficientNet-B4):**
- Batch size 4: ~2GB
- Batch size 8: ~4GB
- Batch size 16: ~8GB
- Batch size 32: ~16GB

### When to adjust augmentation

| Symptom | Likely Issue | Fix |
|---------|-------------|-----|
| Overfitting (val >> train) | Augmentation too weak | Increase RandAugment M |
| Underfitting (train ≈ val ≈ low) | Augmentation too strong | Decrease CutMix α or MixUp α |
| Unstable training | Augmentation too aggressive | Reduce RandomErasing strength |

## Monitoring Training

### 1. Log Files

```bash
# View latest logs
tail -100 optionB_v3_results/training.log

# Search for errors
grep -i "error\|warning" optionB_v3_results/training.log

# Watch real-time
tail -f optionB_v3_results/training.log
```

### 2. TensorBoard Visualization

```bash
# Start TensorBoard
tensorboard --logdir=optionB_v3_results/

# Open in browser: http://localhost:6006
# View:
# - Scalar: losses, metrics over time
# - Histograms: weight distributions
# - Hparams: hyperparameter sweep results
```

### 3. Custom Dashboard

```bash
# Interactive Jupyter notebook
jupyter notebook notebooks/training_dashboard.ipynb

# Shows:
# - Training/validation curves
# - Learning rate schedule
# - Class distribution
# - Sample predictions
```

## Troubleshooting

### Issue: CUDA Out of Memory (OOM)

```python
# Solution 1: Reduce batch size
python optionB_v3.py --batch_size 8

# Solution 2: Use gradient accumulation
python optionB_v3.py --batch_size 4 --accumulation_steps 4

# Solution 3: Enable mixed precision
# (already enabled in most scripts)

# Solution 4: Reduce image size
python optionB_v3.py --img_size 256
```

### Issue: Training very slow

```python
# Check GPU usage
nvidia-smi -l 1  # Update every 1 sec

# Increase workers
python optionB_v3.py --num_workers 8

# Enable compilation (PyTorch 2.0+)
# model = torch.compile(model)
```

### Issue: Poor validation performance

```python
# 1. Check data augmentation isn't too strong
python src/training/baseline_train.py  # Use minimal augmentation

# 2. Verify class distribution
python -c "import json; d=json.load(open('minifigs.json')); \
from collections import Counter; \
print(Counter(x['category'] for x in d).most_common(10))"

# 3. Try longer training
python optionB_v3.py --epochs 30

# 4. Reduce learning rate
python optionB_v3.py --learning_rate 1e-4
```

### Issue: NaN or Inf losses

```python
# 1. Reduce learning rate (prevents explosion)
python optionB_v3.py --learning_rate 1e-4

# 2. Check for label issues
python -c "import json; d=json.load(open('minifigs.json')); \
assert all('category' in x for x in d), 'Missing categories!'"

# 3. Enable gradient clipping (done by default)
# torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

## Advanced: Custom Experiments

### Experiment 1: Test Loss Functions

```bash
# Compare loss functions on same data
for loss in focal arcface supcon; do
    python optionB_v3.py --loss $loss --output results/loss_$loss
done

# Analyze results
python src/training/compare_all_models.py --focus loss_comparison
```

### Experiment 2: Ablation Study

```bash
# Without synthetic augmentation
python optionB_v3.py --no_synthetic_augmentation

# Without TTA
python optionB_v3.py --no_tta

# Without label smoothing
python optionB_v3.py --label_smoothing 0.0

# Compare outputs
python src/training/compare_all_models.py
```

### Experiment 3: Data Sampling Strategy

```bash
# Standard weighted sampling
python optionB_v3.py --sampling_strategy weighted

# Balanced sampling (equal per class)
python optionB_v3.py --sampling_strategy balanced

# Upsampling minority classes
python optionB_v3.py --sampling_strategy upsample_minority
```

## Performance Benchmarks

### Hardware Specs
- **GPU**: NVIDIA A100 (baseline)
- **CPU**: 16-core AMD EPYC
- **RAM**: 256GB

### Training Times (V3, 20 epochs)

| Hardware | Batch 16 | Batch 8 |
|----------|----------|---------|
| A100 GPU | ~2-3 hours | ~3-4 hours |
| V100 GPU | ~3-4 hours | ~4-5 hours |
| RTX 3090 | ~4-5 hours | ~5-7 hours |
| CPU only | ~20-30 hours | ~25-35 hours |

### HP Tuning Timeline

| Phase | Hardware | Time |
|-------|----------|------|
| Optuna (30 trials) | A100 | ~20 hours |
| CV (3 folds × 3 losses × 20 epochs) | A100 | ~6.5 hours |
| **Total** | A100 | ~26-29 hours |

## Best Practices

### ✅ Do's
- ✓ Save every model checkpoint (enable early stopping)
- ✓ Log all hyperparameters in config.json
- ✓ Validate on held-out test set (don't tune on it)
- ✓ Use deterministic seeds for reproducibility
- ✓ Monitor GPU/CPU usage during training
- ✓ Commit successful experiments to git

### ❌ Don'ts
- ✗ Don't train on test set (data leakage)
- ✗ Don't change augmentation mid-training
- ✗ Don't manually tune hyperparameters randomly
- ✗ Don't ignore early stopping signals
- ✗ Don't forget to normalize inputs (ImageNet stats)
- ✗ Don't commit large model files (~100MB+)

## Next Steps

After training:
1. **Analyze results**: Check confusion_matrix.png
2. **Document findings**: Update docs/RESULTS.md
3. **Version model**: Commit metadata + small test dataset
4. **Share results**: Create pull request with summary

---

**Last Updated**: 2026-04-21  
**Tested on**: Python 3.9+, PyTorch 2.0+, CUDA 12.0+
