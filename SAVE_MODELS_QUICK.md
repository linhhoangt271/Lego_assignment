# Save Models to Git - Quick Reference

## 🚀 Quickest Option: GitHub Releases (5 min, Free)

**Best for**: Release versions, archiving, easy sharing

### Step 1: Train & Save Model
```bash
python src/training/optionB_v3.py --epochs 20
# Creates: optionB_v3_results/best_model.pth
```

### Step 2: Create Tag
```bash
git tag v1.0-supcon -m "V3 SupCon model - F1=0.7207"
git push origin v1.0-supcon
```

### Step 3: Create Release on GitHub
1. Go to your repo → Releases → Create release
2. Select tag: v1.0-supcon
3. Upload file: `optionB_v3_results/best_model.pth`
4. Click "Publish Release"

### Step 4: Team Downloads
```bash
# From release page, or via wget:
wget https://github.com/YOUR-USERNAME/big-data-lego-classification/releases/download/v1.0-supcon/best_model.pth
```

✅ **Done!**

---

## 🔥 Professional Option: Git LFS (5 min, Recommended)

**Best for**: Active development, storing many models

### Step 1: Install Git LFS
```bash
# macOS
brew install git-lfs

# Ubuntu
sudo apt-get install git-lfs

# Then initialize
git lfs install
```

### Step 2: Configure Project
```bash
# Track .pth files with LFS
git lfs track "*.pth"
git lfs track "*.pt"

# Commit configuration
git add .gitattributes
git commit -m "Configure Git LFS for model files"
git push origin main
```

### Step 3: Save Model
```bash
# Train your model
python src/training/optionB_v3.py

# Add to git (LFS handles it automatically)
git add optionB_v3_results/best_model.pth
git add optionB_v3_results/model_metadata.json

git commit -m "Save V3 model: F1=0.7207 (SupCon loss)"
git push origin main
```

### Step 4: Team Clones
```bash
git clone <repo>
# best_model.pth automatically downloads (72 MB)
ls best_model.pth  # Ready to use!
```

✅ **Done!**

---

## 📝 Always Create Metadata File

Save alongside model:

```bash
cat > optionB_v3_results/model_metadata.json << 'EOF'
{
  "model_name": "V3 SupCon",
  "architecture": "EfficientNet-B4",
  "loss_function": "Supervised Contrastive",
  "f1_score": 0.7207,
  "validation_f1": 0.7207,
  "training_date": "2026-04-21",
  "epochs": 20,
  "batch_size": 16,
  "learning_rate": 3e-4,
  "notes": "Best overall model - recommended for production"
}
EOF
```

Add to git:
```bash
git add model_metadata.json
git commit -m "Add metadata for V3 model"
```

---

## 🎯 Comparison: Which to Use?

| Feature | GitHub Releases | Git LFS |
|---------|-----------------|---------|
| **Setup** | None | 5 min |
| **Cost** | Free | Free (1GB/month) |
| **Best for** | Release versions | Active development |
| **Team access** | Download link | Auto with `git clone` |
| **Max files** | Unlimited | Unlimited |
| **Learn more** | See MODEL_VERSIONING.md | See MODEL_VERSIONING.md |

---

## ❌ What NOT to Do

```bash
# ❌ DON'T: Commit large files without LFS
git add best_model.pth
git commit -m "Add model"
# GitHub will reject (files > 100MB)

# ✅ DO: Use LFS or GitHub Releases instead
```

---

## 📞 Help

- **Detailed guide**: See `docs/MODEL_VERSIONING.md`
- **Git LFS docs**: https://git-lfs.github.com
- **GitHub Releases**: https://docs.github.com/en/repositories/releasing-projects-on-github

---

**Choose one approach above and go! 🚀**
