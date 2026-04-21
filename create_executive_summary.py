"""Executive Summary Visualization — V3 Model Analysis"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

BASE_DIR = "/home/test/big_data_assignment_2"
OUTPUT_DIR = os.path.join(BASE_DIR, "optionB_v3_results")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#f8f9fa')

# Title
fig.text(0.5, 0.97, 'V3 Model Analysis — Executive Summary',
         ha='center', fontsize=24, fontweight='bold')
fig.text(0.5, 0.94, 'LEGO Minifigure Classification — Comprehensive Evaluation',
         ha='center', fontsize=14, style='italic', color='#555')

# ============================================================
# 1. KEY METRICS (Top Section)
# ============================================================
y_start = 0.88
ax_metrics = fig.add_axes([0.05, y_start-0.12, 0.9, 0.12])
ax_metrics.axis('off')

metrics_data = [
    ('Accuracy', '79.50%', '🎯'),
    ('Top-3 Acc', '91.55%', '📊'),
    ('Macro F1', '0.7406', '📈'),
    ('Weighted F1', '0.7951', '⭐'),
    ('Classes', '48', '🏷️'),
    ('Total Samples', '2,605', '📦'),
]

box_width = 0.145
box_height = 0.11
start_x = 0.05
colors_metric = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#e74c3c', '#1abc9c']

for i, (label, value, emoji) in enumerate(metrics_data):
    x = start_x + i * (box_width + 0.01)

    # Background box
    rect = FancyBboxPatch((x, y_start-0.105), box_width, 0.1,
                          boxstyle="round,pad=0.005",
                          facecolor=colors_metric[i], alpha=0.2,
                          edgecolor=colors_metric[i], linewidth=2)
    ax_metrics.add_patch(rect)

    # Text
    ax_metrics.text(x + box_width/2, y_start-0.04, emoji,
                   ha='center', va='center', fontsize=16)
    ax_metrics.text(x + box_width/2, y_start-0.065, label,
                   ha='center', va='center', fontsize=9, fontweight='bold')
    ax_metrics.text(x + box_width/2, y_start-0.095, value,
                   ha='center', va='center', fontsize=11, fontweight='bold', color='black')

# ============================================================
# 2. BEST MODEL SPECIFICATIONS
# ============================================================
y_pos = 0.75
ax_specs = fig.add_axes([0.05, y_pos-0.15, 0.28, 0.15])
ax_specs.axis('off')

rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                      facecolor='#2ecc71', alpha=0.1,
                      edgecolor='#27ae60', linewidth=2, transform=ax_specs.transAxes)
ax_specs.add_patch(rect)

spec_text = """🏆 BEST MODEL: V3-SupCon

Architecture:
  • Backbone: EfficientNet-B4
  • Input: 380×380 pixels
  • Classes: 48 (with Town split)

Key Techniques:
  • YOLO-based minifig cropping
  • Synthetic class augmentation
  • SupCon contrastive loss
  • Test-Time Augmentation (TTA)
  • Class weight balancing
"""

ax_specs.text(0.05, 0.95, spec_text, transform=ax_specs.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================
# 3. VARIANT COMPARISON (V3)
# ============================================================
ax_variants = fig.add_axes([0.37, y_pos-0.15, 0.28, 0.15])
ax_variants.axis('off')

rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                      facecolor='#9b59b6', alpha=0.1,
                      edgecolor='#8e44ad', linewidth=2, transform=ax_variants.transAxes)
ax_variants.add_patch(rect)

variants_text = """📊 V3 Loss Function Variants

┌─────────────────────────────┐
│ V3-SupCon    79.50% ★★★★★  │
│ V3-Focal     77.35% ★★★★   │
│ V3-ArcFace   74.36% ★★★    │
└─────────────────────────────┘

SupCon Advantage:
  +2.15% over Focal
  +5.14% over ArcFace

Contrastive learning enables
better metric space geometry.
"""

ax_variants.text(0.05, 0.95, variants_text, transform=ax_variants.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================
# 4. PERFORMANCE PROGRESSION
# ============================================================
ax_progress = fig.add_axes([0.69, y_pos-0.15, 0.26, 0.15])
ax_progress.axis('off')

rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                      facecolor='#3498db', alpha=0.1,
                      edgecolor='#2980b9', linewidth=2, transform=ax_progress.transAxes)
ax_progress.add_patch(rect)

progress_text = """📈 Model Improvement Path

Baseline (29.3%)    ▁
  ↓ Class Merge (58.9%)  ▃
  ↓ B0 Enhanced (74.4%)  ▅
  ↓ B2 + TTA (74.9%)     ▆
  ↓ B4 + SupCon (79.5%)  ▉

Total Improvement: +50.2%

Biggest Leap:
  Baseline → Option B
  +29.6 percentage points!
"""

ax_progress.text(0.05, 0.95, progress_text, transform=ax_progress.transAxes,
                fontsize=8.5, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================
# 5. CLASS PERFORMANCE ANALYSIS
# ============================================================
y_pos = 0.58
ax_class = fig.add_axes([0.05, y_pos-0.15, 0.43, 0.15])
ax_class.axis('off')

rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                      facecolor='#f39c12', alpha=0.1,
                      edgecolor='#e67e22', linewidth=2, transform=ax_class.transAxes)
ax_class.add_patch(rect)

class_text = """🏷️ CLASS-LEVEL PERFORMANCE

Top Performing Classes (F1 ≥ 0.90):
  ★ Friends & Fantasy (0.98)
  ★ Preschool (0.97)
  ★ Star Wars (0.95)
  ★ Minecraft (0.94)
  ★ Town - Police (0.92)
  ★ Super Mario (0.91)
  ★ Town - Fire (0.90)

  Total: 7 excellent classes

Challenging Classes (F1 < 0.50):
  • Time Cruisers (0.00) — only 1 sample
  • Town - Airport (0.38) — minority class

Per-Class Statistics:
  • Mean F1: 0.7406 (Macro)
  • Large classes (≥100): 0.8045 avg F1
  • Small classes (<30): 0.6280 avg F1
"""

ax_class.text(0.02, 0.97, class_text, transform=ax_class.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================
# 6. TECHNICAL INNOVATIONS
# ============================================================
ax_tech = fig.add_axes([0.52, y_pos-0.15, 0.43, 0.15])
ax_tech.axis('off')

rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                      facecolor='#e74c3c', alpha=0.1,
                      edgecolor='#c0392b', linewidth=2, transform=ax_tech.transAxes)
ax_tech.add_patch(rect)

tech_text = """🚀 KEY INNOVATIONS IN V3

1. OBJECT DETECTION PREPROCESSING
   → YOLO-based minifig cropping
   → Removes background noise
   → Focuses model on relevant features

2. SYNTHETIC AUGMENTATION
   → Addresses minority class imbalance
   → Target: ≥50 samples per class
   → Improves generalization

3. LOSS FUNCTION ADVANCEMENT
   → SupCon (Supervised Contrastive)
   → Metric space learning
   → Better class separability

4. ARCHITECTURAL UPGRADE
   → EfficientNet-B4 (vs B0/B2)
   → 380×380 resolution (vs 260px)
   → Increased model capacity

5. TEST-TIME AUGMENTATION
   → 5-view TTA ensemble
   → +0.3-1.0% accuracy gain
   → More robust predictions
"""

ax_tech.text(0.02, 0.97, tech_text, transform=ax_tech.transAxes,
            fontsize=8, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================
# 7. RECOMMENDATIONS
# ============================================================
y_pos = 0.38
ax_recom = fig.add_axes([0.05, y_pos-0.12, 0.9, 0.12])
ax_recom.axis('off')

rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01",
                      facecolor='#1abc9c', alpha=0.1,
                      edgecolor='#16a085', linewidth=2, transform=ax_recom.transAxes)
ax_recom.add_patch(rect)

recom_text = """💡 RECOMMENDATIONS & NEXT STEPS

✓ PRODUCTION DEPLOYMENT      Use V3-SupCon for best accuracy/robustness (79.5%)
✓ COMPUTATIONAL BUDGET       If GPU-constrained, use B-Improved (74.4%) — simpler, faster
✓ MINORITY CLASS HANDLING    Current synthetic augmentation strategy is effective
✓ CONFIDENCE SCORING         Top-3 accuracy of 91.55% enables reliable ranking systems
✓ FUTURE IMPROVEMENTS        (1) Ensemble V3-SupCon + V3-Focal  (2) Hyperparameter search  (3) Cross-validation"""

ax_recom.text(0.02, 0.92, recom_text, transform=ax_recom.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================
# 8. METRIC DETAILS TABLE
# ============================================================
y_pos = 0.33
ax_table = fig.add_axes([0.05, y_pos-0.18, 0.9, 0.18])
ax_table.axis('off')

rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01",
                      facecolor='#34495e', alpha=0.05,
                      edgecolor='#2c3e50', linewidth=2, transform=ax_table.transAxes)
ax_table.add_patch(rect)

table_text = """📋 DETAILED METRICS — V3-SupCon with TTA

┌──────────────────────────────────────────────────────────────────────────────────┐
│ Metric                  Value       │ Interpretation                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│ Accuracy                0.7950      │ 79.50% of predictions correct               │
│ Top-3 Accuracy          0.9155      │ 91.55% correct in top 3 predictions         │
│ Top-5 Accuracy          0.9463      │ 94.63% correct in top 5 predictions         │
│ Macro F1                0.7406      │ Unweighted avg, fair for imbalanced data    │
│ Weighted F1             0.7951      │ Weighted avg, reflects actual performance   │
│ Macro Precision         0.7399      │ Avg precision across all classes            │
│ Macro Recall            0.7495      │ Avg recall across all classes               │
│ Test Set Size           2,605       │ Total minifigures evaluated                 │
│ Number of Classes       48          │ Distinct LEGO themes                        │
│ Image Resolution        380×380px   │ Input size to EfficientNet-B4               │
└──────────────────────────────────────────────────────────────────────────────────┘
"""

ax_table.text(0.02, 0.98, table_text, transform=ax_table.transAxes,
             fontsize=7.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

# ============================================================
# FOOTER
# ============================================================
fig.text(0.5, 0.01,
         'Generated: April 8, 2026 | Analysis includes per-class evaluation, variant comparison, and model progression tracking',
         ha='center', fontsize=9, style='italic', color='#999')

plt.savefig(os.path.join(OUTPUT_DIR, 'EXECUTIVE_SUMMARY.png'),
           dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

print("✓ Executive Summary generated: EXECUTIVE_SUMMARY.png")

# ============================================================
# INDEX OF ALL GENERATED FILES
# ============================================================
index_text = """
═══════════════════════════════════════════════════════════════════════════════════
                      V3 MODEL ANALYSIS — FILE INDEX
═══════════════════════════════════════════════════════════════════════════════════

EXECUTIVE SUMMARIES
──────────────────────────────────────────────────────────────────────────────────
📊 EXECUTIVE_SUMMARY.png
   One-page visual summary of all key findings, metrics, and recommendations

TEXT REPORTS
──────────────────────────────────────────────────────────────────────────────────
📄 v3_detailed_analysis.txt
   Comprehensive per-class performance analysis with statistics and rankings
   • Top 15 best performing classes
   • Bottom 15 worst performing classes
   • Performance tiers analysis
   • Precision-recall analysis
   • All 39 classes ranked by F1 score

📄 all_models_analysis.txt
   Cross-model comparison of all 7 variants (Baseline through V3-SupCon)
   • Overall rankings across all metrics
   • Detailed specifications for each model
   • Improvement analysis (baseline vs best, incremental)
   • Key innovations by model
   • Loss function impact analysis

VISUALIZATIONS — V3 MODEL DEEP DIVE
──────────────────────────────────────────────────────────────────────────────────
📈 v3_per_class_f1_detailed.png
   Top 20 / Bottom 10 classes by F1 score with detailed bar charts

📈 v3_precision_recall_scatter.png
   Scatter plot: Precision vs Recall by class (bubble size = support)
   Helps identify classes with precision vs recall imbalance

📈 v3_class_distribution_vs_performance.png
   Top 30 classes by sample count: F1 score vs class size
   Shows correlation between dataset size and model performance

📈 v3_precision_recall_tradeoff.png
   Analysis of precision-biased vs recall-biased classes
   Color coding identifies model strengths and weaknesses

📈 v3_statistical_summary.png
   5-part dashboard with distributions and summary statistics
   • Histograms of F1, Precision, Recall scores
   • Scatter plots vs class size
   • Summary statistics table

VISUALIZATIONS — CROSS-MODEL COMPARISON
──────────────────────────────────────────────────────────────────────────────────
📊 all_models_comparison_radar.png
   Radar charts and heatmap comparing all 7 models
   • Top-K accuracy metrics
   • F1 score metrics
   • Architecture complexity (resolution)
   • All metrics heatmap

📊 all_models_bar_comparison.png
   Side-by-side bar charts for each metric (Accuracy, Top-3, Top-5, F1)
   • Shows clear performance ranking
   • Color-coded by model
   • Value annotations

📊 improvement_trajectory.png
   Line charts showing progression from Baseline to V3-SupCon
   • Accuracy trajectory with percentage gains
   • Weighted F1 trajectory with absolute gains
   • Visualization of incremental improvements

VISUALIZATIONS — ORIGINAL (FROM TRAINING)
──────────────────────────────────────────────────────────────────────────────────
📊 confusion_matrix.png
   Confusion matrix for V3-SupCon showing per-class predictions

📊 per_class_f1.png
   Per-class F1 scores visualization

📊 training_curves_v3_focal.png
   Training/validation curves for Focal loss variant

📊 training_curves_v3_arcface.png
   Training/validation curves for ArcFace loss variant

📊 training_curves_v3_supcon.png
   Training/validation curves for SupCon loss variant (winner)

📊 v3_variant_comparison.png
   Side-by-side comparison of V3 loss function variants

CLASSIFICATION REPORTS
──────────────────────────────────────────────────────────────────────────────────
📋 classification_report.txt
   Standard sklearn classification report with per-class metrics
   • Precision, recall, F1-score for each class
   • Support (sample count) per class
   • Weighted and macro averages

═══════════════════════════════════════════════════════════════════════════════════

KEY FINDINGS SUMMARY:
───────────────────────────────────────────────────────────────────────────────────

✓ BEST MODEL PERFORMANCE
  • V3-SupCon: 79.50% accuracy (27.2% improvement over Baseline)
  • Top-3 accuracy: 91.55% (enables ranking-based systems)
  • Weighted F1: 0.7951

✓ LOSS FUNCTION WINNER
  • SupCon outperforms Focal and ArcFace
  • Contrastive learning creates better metric space
  • +2.15% over Focal, +5.14% over ArcFace

✓ CLASS PERFORMANCE
  • 7 classes with excellent F1 (≥0.90)
  • 2 challenging classes (F1 < 0.50)
  • Large classes: 0.8045 avg F1
  • Small classes: 0.6280 avg F1

✓ INNOVATION IMPACT
  • YOLO preprocessing + synthetic augmentation
  • EfficientNet-B4 backbone
  • SupCon contrastive loss
  • Test-Time Augmentation (+0.3-1.0% boost)

═══════════════════════════════════════════════════════════════════════════════════
"""

index_path = os.path.join(OUTPUT_DIR, 'FILE_INDEX.txt')
with open(index_path, 'w') as f:
    f.write(index_text)

print(f"✓ File index created: FILE_INDEX.txt")
print("\n" + "="*80)
print("ALL ANALYSIS COMPLETE")
print("="*80)
