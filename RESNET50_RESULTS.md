# ResNet50 Simple Fine-tuning — Top 20 Classes

## 🎯 Overview
A clean, simple baseline using ResNet50 (pretrained on ImageNet) fine-tuned on the top 20 LEGO minifigure classes, which comprise 78.9% of the dataset (13,706 samples).

## 📊 Final Results

### Overall Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | 72.46% |
| **Macro F1** | 69.52% |
| **Weighted F1** | 74.01% |
| **Training Time** | 22.4 minutes |

### Training Strategy
- **Phase 1** (5 epochs): Frozen backbone, train head only
  - Learns task-specific classification on top of pretrained features
  - Best F1: 0.4605
  
- **Phase 2** (15 epochs): Full fine-tuning with lower learning rate (1e-5)
  - Updates entire network for better task adaptation
  - Best F1: 0.6952

### Per-Class Performance

#### Excellent (F1 > 0.80)
- **DUPLO**: 0.9550 (197 samples) — Distinctive look, easy to identify
- **Friends**: 0.9538 (177 samples) — Well-separated from other categories
- **Minecraft**: 0.9423 (51 samples) — Very distinctive style
- **Super Mario**: 0.8842 (46 samples) — Clear character designs
- **Star Wars**: 0.8466 (299 samples) — Recognizable costumes
- **NINJAGO**: 0.8451 (189 samples) — Consistent visual style

#### Good (F1 = 0.70-0.80)
- Pirates: 0.8315
- Monkie Kid: 0.7945
- Disney: 0.7703
- Castle: 0.7647
- Harry Potter: 0.7323
- Town: 0.7055
- Super Heroes: 0.7100

#### Challenging (F1 < 0.60)
- Collectible Minifigures: 0.5253 — Diverse random designs
- Sports: 0.5290 — Generic sports figures
- Space: 0.6140 — Mix of themes
- Holiday & Event: 0.4264 — Seasonal variations
- LEGO Ideas: 0.3597 — Fan designs, varied styles
- BrickLink Designer Program: 0.3759 — Custom/diverse designs
- LEGO Brand: 0.3387 — Generic promotional figures

## 🔍 Key Observations

### Why This Works
1. **Large dataset**: 13.7K samples provide strong training signal
2. **ImageNet pretraining**: ResNet50 has already learned general visual features
3. **Simple approach**: Less prone to overfitting than complex architectures
4. **Two-stage training**: Frozen backbone helps prevent catastrophic forgetting

### Why Some Classes Struggle
- **LEGO Brand, Ideas, BrickLink Designer**: No consistent visual patterns; highly diverse sets
- **Holiday & Event, Sports**: Generic/interchangeable figures
- **Collectible Minifigures**: By definition, random assortment of unrelated designs

### Confusion Patterns (from confusion matrix)
- **Town** class is the largest (704 samples) but only 58% recall — hard to distinguish from other generic figures
- **Holiday & Event** confused with generic Town figures
- **LEGO Brand** frequently misclassified as promotional categories
- **Sports** often confused with generic Town/Preschool figures

## 🚀 Potential Improvements

### Quick Wins (1-2% F1)
1. **Test-Time Augmentation (TTA)**: Average predictions from 8-10 augmented versions
2. **Increase input resolution**: 256×256 or 384×384 (vs current 224×224)
3. **Add regularization**: Increase dropout or weight decay

### Medium Effort (2-3% F1)
1. **Better backbone**: Vision Transformer (ViT) or EfficientNet (better for fine-grained classification)
2. **Curriculum learning**: Train on easy classes first, hard classes last
3. **Ensemble**: Train 3-5 models with different seeds, ensemble predictions

### Advanced (3-5% F1)
1. **Metric learning**: Add contrastive/triplet loss alongside cross-entropy
2. **Hard example mining**: Oversample misclassified samples
3. **Multi-task learning**: Add auxiliary task (e.g., predict set year, theme family)
4. **Data augmentation tuning**: Use Optuna to find best augmentation parameters

## 📁 Outputs

```
resnet50_top20_results/
├── best_model.pth                 (91 MB - ResNet50 fine-tuned weights)
├── classification_report.txt       (Per-class precision/recall/F1)
├── confusion_matrix.png            (20×20 confusion matrix heatmap)
├── per_class_f1.png               (Bar chart of F1 scores)
└── results_summary.txt            (This report)
```

## 🎓 Why This Baseline Matters

1. **Simple & interpretable**: No complex techniques; easy to understand what's happening
2. **Fast**: Trains in ~22 minutes on GPU (vs hours for large models)
3. **Good starting point**: 72% accuracy on 20 classes is solid foundation
4. **Clear failure modes**: We can see exactly which classes are hard and why
5. **Reproducible**: Standard ResNet50, standard PyTorch, no custom code

This baseline demonstrates that for a well-balanced subset of data with clear visual distinctions, transfer learning is highly effective. Further improvements would require addressing the ambiguity in "generic" LEGO figures.

---

**Created**: 2026-05-15  
**Model**: ResNet50 (pretrained ImageNet)  
**Framework**: PyTorch  
**Data**: 13,706 images across 20 classes
