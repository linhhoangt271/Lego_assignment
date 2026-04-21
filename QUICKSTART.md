# Quick Start Reference Card

## 🚀 Get Running in 5 Minutes

```bash
# 1. Clone (or already have it)
cd big_data_assignment_2

# 2. Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run
python src/training/baseline_train.py

# Done! Check results in baseline_results/
```

---

## 📚 Key Documentation

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [README.md](README.md) | Overview, results, features | 5 min |
| [docs/SETUP.md](docs/SETUP.md) | Environment setup | 10 min |
| [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | How to train models | 10 min |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical details | 15 min |
| [docs/RESULTS.md](docs/RESULTS.md) | Experiment results & findings | 20 min |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to collaborate with Git | 10 min |

---

## 🎯 Training Workflows

### Quick Test (5 min)
```bash
python src/training/baseline_train.py
```

### Full Training (2-3 hours)
```bash
python src/training/optionB_v3.py --epochs 20
```

### Research Mode (29 hours)
```bash
python src/training/optionB_v3_hpopt.py  # HP tuning + 3-fold CV
```

### Compare All Models (1 hour)
```bash
python src/training/compare_all_models.py
```

---

## 📊 Best Model

```
Architecture:     EfficientNet-B4 (380×380 pixels)
Loss Function:    Supervised Contrastive (SupCon)
F1 Score:         0.7207 (3-fold CV average)
Best Fold:        0.7432
Training Time:    ~2-3 hours per fold
```

---

## 🛠️ Common Tasks

### Check GPU
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Monitor Training
```bash
tensorboard --logdir optionB_v3_results/
# Open http://localhost:6006
```

### Change Hyperparameters
Edit lines in script header:
```python
BATCH_SIZE = 16      # Reduce to 8 if OOM
NUM_EPOCHS = 20      # Increase for better accuracy
LEARNING_RATE = 3e-4 # Tune if not converging
```

### View Results
```bash
cat optionB_v3_results/classification_report.txt
```

---

## 🔄 Git Workflow

### First Time
```bash
git config user.email "your@email.com"
git config user.name "Your Name"
```

### Make Changes
```bash
git checkout -b feature/my-improvement
# ... edit files ...
git add <files>
git commit -m "Description of change"
git push -u origin feature/my-improvement
# Create Pull Request on GitHub
```

### Update from Others
```bash
git fetch origin
git rebase origin/master
```

---

## 📈 Expected Performance

| Model | F1 Score | Time |
|-------|----------|------|
| Baseline (B0, 122 cats) | 0.62 | 5 min |
| Option B (B0, merged) | 0.68 | 30 min |
| Option B V3 (B4) | 0.77 | 3 hours |
| V3 HP-Tuned (best) | 0.7207 | 29 hours |

---

## ⚠️ Common Issues

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | `--batch_size 8` |
| Training very slow | `--num_workers 8` |
| No GPU detected | Check `nvidia-smi`, reinstall PyTorch |
| Data not found | Download images/ and minifigs.json |
| Import errors | `pip install -r requirements.txt` |

---

## 📁 Project Structure

```
big_data_assignment_2/
├── src/training/          ← Main training scripts
├── src/models/            ← Model definitions
├── docs/                  ← Documentation
├── notebooks/             ← Jupyter analysis
├── requirements.txt       ← Dependencies
└── README.md              ← Start here!
```

---

## 🔗 Useful Links

- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **EfficientNet Paper**: https://arxiv.org/abs/1905.11946
- **Optuna Docs**: https://optuna.readthedocs.io/
- **TensorBoard Guide**: https://www.tensorflow.org/tensorboard

---

## 💡 Tips

✅ **DO:**
- Read README.md first
- Run quick test before big training
- Save important results
- Commit frequently with clear messages
- Update docs when making changes

❌ **DON'T:**
- Train models directly on master branch
- Commit large .pth files
- Ignore errors, investigate first
- Force push to master
- Forget to save hyperparameters

---

## 📞 Help

1. Check `docs/TRAINING_GUIDE.md` for troubleshooting
2. Read `docs/ARCHITECTURE.md` for technical questions
3. See `CONTRIBUTING.md` for git help
4. Create GitHub issue for bugs

---

**Last Updated**: 2026-04-21  
**Status**: ✅ Ready for team collaboration
