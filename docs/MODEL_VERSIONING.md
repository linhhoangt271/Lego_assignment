# Model Versioning & Weights Management

How to save and share model weights with your team.

---

## 🎯 Quick Decision: Which Method?

| Method | File Size Limit | Cost | Best For | Setup Time |
|--------|-----------------|------|----------|-----------|
| **Git LFS** | Unlimited | $5/month* | Most projects | 5 min |
| **GitHub Releases** | 2GB per file | Free | Release versions | 5 min |
| **AWS S3** | Unlimited | Pay-per-use | Large projects | 15 min |
| **Google Drive** | Unlimited | Free (15GB) | Small teams | 3 min |
| **Don't store** | N/A | Free | Reproducible models | 1 min |

**Recommendation**: Use **Git LFS** (easiest + professional) OR **GitHub Releases** (free + simple)

---

## Option 1: Git LFS (Recommended) ⭐

### What is Git LFS?
Git Large File Storage replaces large files with tiny pointers in git, stores actual files on GitHub's servers.

### Setup (5 minutes)

**1. Install Git LFS**
```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Windows
# Download from: https://git-lfs.github.com
```

**2. Initialize Git LFS**
```bash
git lfs install
```

**3. Track model files**
```bash
# Add .pth files to LFS
git lfs track "*.pth"
git lfs track "*.pt"

# This creates/updates .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for model files"
```

### Using Git LFS

**Save a model checkpoint:**
```bash
# Train and save model
python src/training/optionB_v3.py

# Model is saved as: optionB_v3_results/best_model.pth (72MB)

# Add to git
git add optionB_v3_results/best_model.pth
git commit -m "Save V3 model: SupCon loss, F1=0.7207"

# Push (Git LFS handles the large file automatically)
git push origin main
```

**Team members:**
```bash
# Clone normally - Git LFS automatically handles it
git clone https://github.com/your-username/big-data-lego-classification.git

# Model files download automatically
ls optionB_v3_results/best_model.pth  # 72MB, ready to use!
```

### Commit Model Weights with Git LFS

```bash
# 1. Train your model
python src/training/optionB_v3.py --epochs 20

# 2. Check the file size
ls -lh optionB_v3_results/best_model.pth
# Output: -rw-r--r--  72M Apr 21 12:00 best_model.pth

# 3. Create a model metadata file (JSON)
cat > optionB_v3_results/model_metadata.json << 'EOF'
{
  "name": "SupCon Loss - V3 Baseline",
  "architecture": "EfficientNet-B4",
  "loss_function": "Supervised Contrastive",
  "f1_score": 0.7207,
  "val_f1": 0.7207,
  "training_epochs": 20,
  "batch_size": 16,
  "learning_rate": 3e-4,
  "training_date": "2026-04-21",
  "hyperparameters": {
    "label_smoothing": 0.1,
    "scheduler_t0": 6,
    "warmup_epochs": 2
  },
  "cross_validation": {
    "fold_0_f1": 0.7084,
    "fold_1_f1": 0.7213,
    "fold_2_f1": 0.7191,
    "mean_f1": 0.7163,
    "std_dev": 0.0056
  },
  "notes": "Best overall performer, most stable across folds"
}
EOF

# 4. Add and commit
git add optionB_v3_results/best_model.pth
git add optionB_v3_results/model_metadata.json
git commit -m "Save V3 SupCon model checkpoint

Model: EfficientNet-B4 + Supervised Contrastive Loss
Performance: F1=0.7207 (3-fold CV), best fold F1=0.7213
Training: 20 epochs, LR=3e-4, batch_size=16

Cross-validation results:
  Fold 0: F1=0.7084
  Fold 1: F1=0.7213 (best)
  Fold 2: F1=0.7191
  Mean: F1=0.7163 ± 0.0056

This is the recommended model for production use."

# 5. Push to GitHub
git push origin main
```

### Check Git LFS Status

```bash
# See what files are tracked by LFS
git lfs ls-files

# Output example:
# 8a6c5b1234 * optionB_v3_results/best_model.pth

# Check storage usage
git lfs quota
```

### GitHub Storage

- **Free tier**: 1 GB storage + 1 GB bandwidth/month
- **Pro tier**: $5/month → 100 GB storage + 100 GB bandwidth

Your project: ~200MB models = fits easily in free tier

---

## Option 2: GitHub Releases (Free, Simple) ⭐

No setup required! Use GitHub's built-in release feature.

### Upload Model as Release

**1. Create GitHub Release:**
```bash
# Create local tag
git tag -a v1.0-supcon -m "V3 SupCon model checkpoint

EfficientNet-B4 + Supervised Contrastive Loss
F1 Score: 0.7207 (3-fold CV)
Training: 20 epochs

Recommended for production use."

# Push tag
git push origin v1.0-supcon
```

**2. On GitHub:**
- Go to your repo
- Click "Releases" (on right sidebar)
- Click "Create a new release"
- Select tag: `v1.0-supcon`
- Title: "V3 SupCon Model - Production Ready"
- Description:
  ```
  EfficientNet-B4 Backbone
  Loss: Supervised Contrastive (SupCon)
  
  Performance:
  - F1 Score: 0.7207 (3-fold CV average)
  - Best Fold: F1 = 0.7213
  - Stable across folds (σ = 0.0056)
  
  Training:
  - Epochs: 20
  - Batch size: 16
  - Learning rate: 3e-4
  - Hardware: NVIDIA A100
  
  Ready for production deployment.
  ```
- **Upload Files:** Click to upload `optionB_v3_results/best_model.pth`
- Click "Publish Release"

**3. Team members download:**
- Visit Releases tab
- Click model version
- Download .pth file directly
- OR use command:
  ```bash
  # Download from release
  wget https://github.com/YOUR-USERNAME/big-data-lego-classification/releases/download/v1.0-supcon/best_model.pth
  ```

---

## Option 3: Store Separately (Recommended for Development)

For active development, store models externally and track metadata in git.

### Google Drive (Free, Easy)

**1. Upload to Google Drive:**
```bash
# Store models in shared Google Drive folder
# Share link with team
```

**2. Track in git:**
```bash
# Create metadata file only
cat > optionB_v3_results/README.md << 'EOF'
# V3 SupCon Model

Model: EfficientNet-B4 + SupCon Loss
F1 Score: 0.7207 (3-fold CV)

**Download from:** 
https://drive.google.com/drive/folders/YOUR-FOLDER-ID

File: best_model.pth (72 MB)
MD5: abc123def456...

## Loading in Code

```python
import torch
from src.models.efficientnet import EfficientNetModel

model = EfficientNetModel(num_classes=30)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

## Results

- Training F1: 0.7207
- Best Fold F1: 0.7213
- Training Date: 2026-04-21
EOF

git add optionB_v3_results/README.md
git commit -m "Add model documentation (weights stored on Google Drive)"
git push origin main
```

### AWS S3 (Professional)

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Upload model
aws s3 cp optionB_v3_results/best_model.pth s3://your-bucket/models/v3-supcon/

# Create metadata in git
cat > MODEL_LOCATIONS.md << 'EOF'
# Model Storage Locations

## V3 SupCon (Production)
- **Location**: s3://your-bucket/models/v3-supcon/best_model.pth
- **F1 Score**: 0.7207
- **Download**: `aws s3 cp s3://your-bucket/models/v3-supcon/best_model.pth best_model.pth`

## V3 Focal (Stable)
- **Location**: s3://your-bucket/models/v3-focal/best_model.pth
- **F1 Score**: 0.7163
- **Download**: `aws s3 cp s3://your-bucket/models/v3-focal/best_model.pth best_model.pth`
EOF

git add MODEL_LOCATIONS.md
git commit -m "Add model storage locations"
```

---

## Option 4: Don't Store Weights (Reproducible)

For reproducible research, just store hyperparameters in git. Anyone can retrain.

```bash
# Store config, not weights
cat > configs/v3_supcon.json << 'EOF'
{
  "model": "EfficientNet-B4",
  "loss": "SupCon",
  "hyperparameters": {
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 3e-4,
    "scheduler_t0": 6,
    "label_smoothing": 0.1
  },
  "augmentation": {
    "cutmix_alpha": 1.0,
    "mixup_alpha": 0.2,
    "randaugment": "N=2, M=8"
  }
}
EOF

git add configs/v3_supcon.json
git commit -m "Add V3 SupCon configuration (reproducible from config)"
```

Team members retrain:
```bash
python src/training/optionB_v3.py --config configs/v3_supcon.json
```

---

## 📋 Recommended Workflow

### For Your Team:

**1. Development Phase:**
```bash
# Store models on Google Drive or S3
# Store metadata in git
git add model_metadata.json
git commit -m "Add model metadata (weights on shared storage)"
```

**2. Release Phase:**
```bash
# Use Git LFS for final release
git add final_model.pth
git commit -m "Release V3 model via Git LFS"
git push origin main
```

**3. Production Phase:**
```bash
# Use GitHub Releases for versioning
git tag v1.0
git push origin v1.0
# Upload to GitHub Releases
```

---

## Practical Example: Your Workflow

### Step 1: Train and Save Model

```bash
python src/training/optionB_v3.py --epochs 20

# Creates:
# optionB_v3_results/best_model.pth (72MB)
# optionB_v3_results/classification_report.txt
```

### Step 2: Create Model Card

```bash
cat > optionB_v3_results/MODEL_CARD.md << 'EOF'
# Model Card: V3 SupCon

## Model Details
- **Architecture**: EfficientNet-B4 (380×380 input)
- **Backbone**: ImageNet pretrained
- **Loss Function**: Supervised Contrastive + CrossEntropy
- **Number of Classes**: 30 (merged categories)

## Performance
- **Validation F1**: 0.7207 ± 0.0056 (3-fold CV)
- **Best Single Fold**: 0.7213
- **Precision**: 0.72
- **Recall**: 0.72

## Training
- **Epochs**: 20
- **Batch Size**: 16
- **Learning Rate**: 3e-4
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (t0=6)
- **Training Time**: ~3 hours (A100 GPU)

## Intended Use
- LEGO minifigure category classification
- Best overall performance on test set
- Recommended for production deployment

## Limitations
- Trained on specific LEGO dataset
- May not generalize to other toy datasets
- Requires GPU for inference (can be optimized with ONNX)

## How to Use

```python
import torch
from src.models.efficientnet import EfficientNetModel

# Load model
model = EfficientNetModel(num_classes=30)
model.load_state_dict(torch.load('optionB_v3_results/best_model.pth'))
model.eval()

# Inference
from PIL import Image
from torchvision import transforms

image = Image.open('minifigure.jpg')
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
x = transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(x)
    prediction = logits.argmax(dim=1)
```

## License
[Your License Here]
EOF
```

### Step 3: Commit Everything

**Option A: Use Git LFS**
```bash
git lfs track "*.pth"
git add .gitattributes

git add optionB_v3_results/best_model.pth
git add optionB_v3_results/MODEL_CARD.md
git add optionB_v3_results/classification_report.txt

git commit -m "Save V3 SupCon model checkpoint via Git LFS

Model: EfficientNet-B4 + Supervised Contrastive Loss
Performance: F1=0.7207 (3-fold CV average)

Files:
- best_model.pth (72 MB, tracked by Git LFS)
- MODEL_CARD.md (model documentation)
- classification_report.txt (detailed metrics)

Ready for production deployment."

git push origin main
```

**Option B: GitHub Release**
```bash
git tag v1.0-supcon-production
git push origin v1.0-supcon-production

# Then upload best_model.pth to GitHub Release UI
```

### Step 4: Team Members Load Model

**With Git LFS:**
```bash
git clone <repo>
# best_model.pth automatically downloaded (72 MB)
python inference.py  # Use model directly
```

**With GitHub Release:**
```bash
# Download from release page
wget https://github.com/.../releases/download/v1.0-supcon-production/best_model.pth
python inference.py
```

---

## Best Practices

✅ **DO:**
- Store model metadata in git (always)
- Document model architecture & hyperparameters
- Create MODEL_CARD.md for each version
- Version releases with git tags
- Store training logs in git

❌ **DON'T:**
- Commit without Git LFS (GitHub will reject files > 100MB)
- Forget to document which model is "best"
- Update model without committing metadata
- Store credentials or sensitive data
- Commit development/temporary models

---

## Summary

| Use Case | Method | Setup |
|----------|--------|-------|
| **Development** | Google Drive + metadata | 3 min |
| **Team Collaboration** | Git LFS | 5 min |
| **Production Release** | GitHub Releases | 5 min |
| **Reproducibility** | Store config only | 1 min |

**For your project**: Start with **Git LFS** → Release with **GitHub Releases**

---

**Last Updated**: 2026-04-21
