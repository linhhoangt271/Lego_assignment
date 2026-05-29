"""Flask web app for LEGO Minifigure Classification — Model V3 (EfficientNet-B4)."""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
LABELS_PATH  = PROJECT_DIR / 'optionB_v3_results' / 'labels.json'
CKPT_PATH    = PROJECT_DIR / 'optionB_v3_results' / 'best_model.pth'

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)

# ── Model (mirrors notebook exactly) ──────────────────────────────────
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)
        self.mm    = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale
        sine  = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0, max=1.0))
        phi   = cosine * self.cos_m - sine * self.sin_m
        phi   = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1)
        return (one_hot * phi + (1 - one_hot) * cosine) * self.scale


class EmbeddingClassifier(nn.Module):
    def __init__(self, backbone, in_features, num_classes, embedding_dim=512, use_arcface=False):
        super().__init__()
        self.backbone   = backbone
        self.embedding  = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.arcface    = ArcFaceHead(embedding_dim, num_classes) if use_arcface else None

    def forward(self, x, labels=None, return_embedding=False):
        features   = self.backbone(x)
        embeddings = self.embedding(features)
        logits     = self.arcface(embeddings, labels) if self.arcface is not None else self.classifier(embeddings)
        if return_embedding:
            return logits, F.normalize(embeddings, dim=1)
        return logits


def build_v3_model(num_classes, variant='focal'):
    base        = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    in_features = base.classifier[1].in_features
    backbone    = nn.Sequential(base.features, base.avgpool, nn.Flatten())
    return EmbeddingClassifier(
        backbone, in_features, num_classes, use_arcface=(variant == 'arcface')
    ).to(DEVICE)


# ── Transform (mirrors notebook eval_transform) ────────────────────────
class PadToSquare:
    def __call__(self, img):
        max_side = max(img.size)
        return ImageOps.pad(img, (max_side, max_side), color=(255, 255, 255))


eval_transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Startup: load model once ───────────────────────────────────────────
app = Flask(__name__)
_model        = None
_idx_to_label = None
_init_error   = None


def _load_model():
    global _model, _idx_to_label, _init_error

    if not LABELS_PATH.exists():
        _init_error = (
            f'Label mapping not found at {LABELS_PATH.relative_to(PROJECT_DIR)}. '
            'Run the V3 section of Models_Summary.ipynb — Step 2 saves labels.json automatically.'
        )
        return

    if not CKPT_PATH.exists():
        _init_error = (
            f'Checkpoint not found at {CKPT_PATH.relative_to(PROJECT_DIR)}. '
            'Complete V3 training in Models_Summary.ipynb first.'
        )
        return

    try:
        with open(LABELS_PATH, encoding='utf-8') as f:
            raw = json.load(f)
        _idx_to_label = {int(k): v for k, v in raw.items()}
        num_classes   = len(_idx_to_label)

        m = build_v3_model(num_classes, variant='focal')
        state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        _model = m
        print(f'[app] Model V3 loaded — {num_classes} classes on {DEVICE}')
    except Exception as exc:
        _init_error = str(exc)
        print(f'[app] Load error: {exc}')


_load_model()

# ── Routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', model_ready=(_model is not None), error=_init_error)


@app.route('/status')
def status():
    return jsonify({
        'model_ready': _model is not None,
        'error':       _init_error,
        'device':      str(DEVICE),
        'num_classes': len(_idx_to_label) if _idx_to_label else 0,
    })


@app.route('/predict', methods=['POST'])
def predict():
    if _model is None:
        return jsonify({'error': _init_error or 'Model not loaded'}), 503

    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'error': 'No image provided'}), 400

    try:
        img = Image.open(request.files['image'].stream).convert('RGB')
    except Exception:
        return jsonify({'error': 'Could not read image file'}), 400

    tensor = eval_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(_model(tensor), dim=1)[0]

    k      = min(5, len(_idx_to_label))
    top_k  = torch.topk(probs, k)
    preds  = [
        {
            'rank':       i + 1,
            'class':      _idx_to_label[idx.item()],
            'confidence': round(prob.item() * 100, 2),
        }
        for i, (prob, idx) in enumerate(zip(top_k.values, top_k.indices))
    ]

    return jsonify({'predictions': preds, 'num_classes': len(_idx_to_label), 'device': str(DEVICE)})


if __name__ == '__main__':
    print(f'Starting LEGO Classifier on {DEVICE}')
    app.run(debug=True, port=5000)
