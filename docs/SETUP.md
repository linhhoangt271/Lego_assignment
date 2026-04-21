# Environment Setup

## System Requirements

### Minimum
- **GPU**: 6GB VRAM (for batch size 8) — e.g., RTX 3060
- **CPU**: 8 cores, 2.0+ GHz
- **RAM**: 32GB system memory
- **Storage**: 2.5GB free (images ~2.1GB, code ~400MB)
- **Python**: 3.9, 3.10, or 3.11

### Recommended
- **GPU**: 12GB+ VRAM (for batch size 16) — e.g., RTX 4090, A100
- **CPU**: 16+ cores
- **RAM**: 64GB+
- **Storage**: SSD with 3GB free
- **Python**: 3.11 (latest stable)

### Tested Configurations
- ✅ NVIDIA A100 + CUDA 12.0 + cuDNN 8.6 + PyTorch 2.1
- ✅ NVIDIA V100 + CUDA 11.8 + PyTorch 2.0
- ✅ RTX 3090 + CUDA 12.0 + PyTorch 2.1
- ✅ Mac M1/M2 (MPS) + PyTorch 2.0+
- ⚠️ CPU-only (very slow, ~20-30 hours per training run)

---

## Installation Steps

### 1. Clone Repository

```bash
git clone <repo-url> big_data_assignment_2
cd big_data_assignment_2
```

### 2. Create Virtual Environment

#### Option A: Conda (Recommended)

```bash
# Create environment
conda create -n lego_classification python=3.11

# Activate
conda activate lego_classification

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Option B: Python venv

```bash
# Create environment
python3.11 -m venv venv

# Activate
source venv/bin/activate    # Linux/Mac
# OR
venv\Scripts\activate       # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch

#### GPU (NVIDIA CUDA 12.0)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu120
```

#### GPU (NVIDIA CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Mac (Apple Silicon M1/M2/M3)

```bash
pip install torch torchvision torchaudio
# PyTorch auto-detects MPS backend
```

#### CPU-only

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```

### 4. Install Project Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- PyTorch 2.0+ (already done above)
- scikit-learn, numpy, scipy
- Pillow, matplotlib, seaborn
- Optuna (hyperparameter tuning)
- Jupyter (notebooks)

**Installation time:** 5-10 minutes

### 5. Optional: Install YOLO (for synthetic augmentation)

```bash
pip install ultralytics
```

**Download YOLOv8 model:**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Creates ~/.cache/ultralytics/ with model weights
```

---

## Data Setup

### Download Data

The following files are NOT in git (too large):

1. **images/** (2.1 GB)
   - Contains minifigure JPEG images
   - Directory structure: `images/image_001.jpg`, etc.

2. **minifigs.json** (11 MB)
   - Metadata: filename, category, town subcategory
   - JSON array with objects: `{filename, category, town}`

**Get data:**
```bash
# Option 1: Download from shared storage/S3
aws s3 cp s3://dataset-bucket/minifigs.json .
aws s3 sync s3://dataset-bucket/images/ images/

# Option 2: From instructor/shared drive
# Copy images/ folder to project root
# Copy minifigs.json to project root

# Verify
ls -lh images/ | head -5
head -c 200 minifigs.json
```

### Verify Data

```bash
# Check structure
python -c "
import json, os
with open('minifigs.json') as f:
    data = json.load(f)
print(f'Total images: {len(data)}')
print(f'Sample: {data[0]}')
print(f'Categories: {len(set(x[\"category\"] for x in data))}')
print(f'Files exist: {sum(1 for x in data if os.path.exists(f\"images/{x[\"filename\"]}\"))}')
"

# Output should show:
# Total images: 2,500,000+
# Categories: 122 (or 30 if merged)
# Files exist: should match total
```

---

## Development Setup

### IDE Configuration

#### VS Code

```bash
# Install extensions
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsservices.ms-vscode-vs-collab-pack

# Create .vscode/settings.json
cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.pylintEnabled": true,
    "python.linting.pylintPath": "${workspaceFolder}/venv/bin/pylint",
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "ms-python.python"
    },
    "jupyter.kernels.filter": ["python3"]
}
EOF
```

#### PyCharm

```bash
# Set interpreter: Settings > Project > Python Interpreter > Add > Existing Environment
# Point to: venv/bin/python

# Enable Jupyter support
# Settings > Plugins > Search "Jupyter" > Install JetBrains Jupyter
```

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
EOF

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

---

## Running Your First Training

### Quick Verification (5 minutes)

```bash
# Ensure you're in the virtual environment
which python  # should show venv path

# Test GPU availability
python -c "import torch; print('GPU:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Run baseline on small subset
python src/training/baseline_train.py

# Expected output:
# ✓ Using device: cuda (or cpu/mps)
# ✓ Loading data...
# ✓ Building model...
# ✓ Training epoch 1/10
# ...
# ✓ Test F1: 0.61-0.65
```

### Next: Full Training

```bash
# Train V3 model (2-3 hours on GPU)
python src/training/optionB_v3.py --epochs 20 --batch_size 16

# Monitor with TensorBoard
tensorboard --logdir optionB_v3_results/
```

---

## Troubleshooting Setup

### Issue: ModuleNotFoundError

```bash
# Problem: import torch fails
# Solution:
pip list | grep torch
# If empty, reinstall PyTorch (step 3 above)

# Problem: import ultralytics fails
# Solution (optional, only if using YOLO):
pip install ultralytics
```

### Issue: CUDA/Device Errors

```bash
# Problem: RuntimeError: No CUDA devices found
# Solution 1: Verify NVIDIA drivers
nvidia-smi  # Should show GPU info

# Solution 2: Reinstall CUDA-compatible PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu120

# Problem: Out of memory
# Solution: Reduce batch size
python optionB_v3.py --batch_size 8
```

### Issue: Data Loading Fails

```bash
# Problem: FileNotFoundError: images/...jpg
# Solution:
ls -lh images/ | wc -l  # Should show millions of files

# If empty, download data (see Data Setup section)

# Problem: minifigs.json not found
ls -la minifigs.json
# If missing, download from source
```

### Issue: Slow Training

```bash
# Check GPU utilization
nvidia-smi -l 1  # Update every 1 second

# If GPU < 90%:
# - Increase batch size: --batch_size 32
# - Increase workers: --num_workers 8
# - Check disk I/O: iostat 1 1

# If GPU 100%, CPU < 50%:
# - Increase workers to match CPU cores
```

---

## Environment Variables

### CUDA Configuration (Optional)

```bash
# Reduce GPU memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use TensorFloat32 for speed (slightly less precision)
export CUDA_LAUNCH_BLOCKING=0

# Disable cuDNN benchmarking (if training unstable)
export CUDNN_BENCHMARK=0
```

### Python Configuration

```bash
# Set matplotlib backend for headless systems
export MPLBACKEND=Agg

# Enable TensorBoard event file writing
export TENSORBOARD_LOGS=./runs/
```

### Full Setup Script

```bash
#!/bin/bash
# save as setup_env.sh

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export MPLBACKEND=Agg

source venv/bin/activate
echo "✓ Environment activated"
echo "✓ Python: $(python --version)"
echo "✓ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "✓ GPU: $(python -c 'import torch; print(torch.cuda.is_available())')"
```

**Usage:**
```bash
chmod +x setup_env.sh
source setup_env.sh
```

---

## Docker Setup (Optional)

For reproducible environments across machines:

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install scikit-learn optuna matplotlib seaborn tensorboard jupyter pandas

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . .

# Default command
CMD ["bash"]
```

**Build & Run:**
```bash
docker build -t lego-classification .
docker run --gpus all -it -v $(pwd):/workspace lego-classification
```

---

## Cleanup

### Deactivate Environment
```bash
deactivate
```

### Remove Virtual Environment
```bash
rm -rf venv/
# or
conda remove -n lego_classification --all
```

### Clear Cache & Checkpoints
```bash
# Clear PyTorch cache
rm -rf ~/.cache/torch

# Clear Optuna database
rm -rf optionB_v3_hpopt_results/*.db

# Clear large model files
rm -f *.pth optionB_*_results/*.pth
```

---

## System Check Checklist

- [ ] Python 3.9+ installed: `python --version`
- [ ] Virtual environment created & activated
- [ ] PyTorch installed & GPU available: `torch.cuda.is_available()`
- [ ] Dependencies installed: `pip list | grep torch scikit optuna`
- [ ] Data downloaded: `ls images/ | wc -l` (should be millions)
- [ ] minifigs.json exists: `wc -l minifigs.json`
- [ ] Code accessible: `ls src/training/*.py`
- [ ] Jupyter working: `jupyter notebook --version`
- [ ] TensorBoard ready: `tensorboard --help`

---

**Last Updated**: 2026-04-21  
**Tested on**: Python 3.11, PyTorch 2.1, CUDA 12.0
