"""Comprehensive Cross-Model Comparison Analysis"""
import json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

BASE_DIR = "/home/test/big_data_assignment_2"

# ============================================================
# MODEL METRICS
# ============================================================
models = {
    'Baseline': {
        'accuracy': 0.2930,
        'top3': 0.4821,
        'top5': 0.5818,
        'macro_f1': 0.3224,
        'weighted_f1': 0.2730,
        'arch': 'EfficientNet-B0',
        'img_size': 128,
        'classes': 122,
        'features': 'Basic',
        'color': '#e74c3c'
    },
    'Option B': {
        'accuracy': 0.5885,
        'top3': 0.8534,
        'top5': 0.9290,
        'macro_f1': 0.5730,
        'weighted_f1': 0.5777,
        'arch': 'EfficientNet-B0',
        'img_size': 128,
        'classes': 40,
        'features': 'Merged Classes',
        'color': '#f39c12'
    },
    'B Improved': {
        'accuracy': 0.7440,
        'top3': 0.9213,
        'top5': 0.9585,
        'macro_f1': 0.6929,
        'weighted_f1': 0.7487,
        'arch': 'EfficientNet-B0',
        'img_size': 224,
        'classes': 40,
        'features': 'Focal+CutMix+MixUp',
        'color': '#f1c40f'
    },
    'V2': {
        'accuracy': 0.7493,
        'top3': 0.9029,
        'top5': 0.9413,
        'macro_f1': 0.7083,
        'weighted_f1': 0.7493,
        'arch': 'EfficientNet-B2',
        'img_size': 260,
        'classes': 48,
        'features': 'Town Split+TTA',
        'color': '#3498db'
    },
    'V3-Focal': {
        'accuracy': 0.7735,
        'top3': 0.9102,
        'top5': 0.9409,
        'macro_f1': 0.7315,
        'weighted_f1': 0.7736,
        'arch': 'EfficientNet-B4',
        'img_size': 380,
        'classes': 48,
        'features': 'YOLO+Synthetic+Focal',
        'color': '#9b59b6'
    },
    'V3-ArcFace': {
        'accuracy': 0.7436,
        'top3': 0.8795,
        'top5': 0.9167,
        'macro_f1': 0.6913,
        'weighted_f1': 0.7345,
        'arch': 'EfficientNet-B4',
        'img_size': 380,
        'classes': 48,
        'features': 'YOLO+Synthetic+ArcFace',
        'color': '#8e44ad'
    },
    'V3-SupCon': {
        'accuracy': 0.7950,
        'top3': 0.9155,
        'top5': 0.9463,
        'macro_f1': 0.7406,
        'weighted_f1': 0.7951,
        'arch': 'EfficientNet-B4',
        'img_size': 380,
        'classes': 48,
        'features': 'YOLO+Synthetic+SupCon+TTA',
        'color': '#2ecc71'
    },
}

print("="*80)
print("COMPREHENSIVE CROSS-MODEL COMPARISON")
print("="*80)

# ============================================================
# VISUALIZATION 1: Metric Comparison Radar Chart
# ============================================================
print("\nGenerating radar chart comparison...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

metrics_list = [
    ('accuracy', 'top3', 'top5'),
    ('macro_f1', 'weighted_f1'),
]

metric_names_display = {
    'accuracy': 'Accuracy',
    'top3': 'Top-3 Acc',
    'top5': 'Top-5 Acc',
    'macro_f1': 'Macro F1',
    'weighted_f1': 'Weighted F1'
}

# Radar 1: Top-K Accuracy
ax = axes[0]
selected_metrics = ('accuracy', 'top3', 'top5')
angles = np.linspace(0, 2*np.pi, len(selected_metrics), endpoint=False).tolist()
angles += angles[:1]

for model_name, model_data in models.items():
    values = [model_data[m] for m in selected_metrics]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=model_data['color'])
    ax.fill(angles, values, alpha=0.15, color=model_data['color'])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([metric_names_display[m] for m in selected_metrics], fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('Top-K Accuracy Metrics', fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax.grid(True)

# Radar 2: F1 Metrics
ax = axes[1]
selected_metrics = ('macro_f1', 'weighted_f1')
angles = np.linspace(0, 2*np.pi, len(selected_metrics), endpoint=False).tolist()
angles += angles[:1]

for model_name, model_data in models.items():
    values = [model_data[m] for m in selected_metrics]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=model_data['color'])
    ax.fill(angles, values, alpha=0.15, color=model_data['color'])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([metric_names_display[m] for m in selected_metrics], fontsize=10)
ax.set_ylim(0, 1)
ax.set_title('F1 Score Metrics', fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
ax.grid(True)

# Bar chart: Architecture complexity
ax = axes[2]
model_names = list(models.keys())
img_sizes = [models[m]['img_size'] for m in model_names]
colors = [models[m]['color'] for m in model_names]

bars = ax.bar(model_names, img_sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Image Size (pixels)', fontsize=11)
ax.set_title('Model Architecture: Input Resolution', fontsize=12, fontweight='bold')
ax.set_xticklabels(model_names, rotation=45, ha='right')
for bar, size in zip(bars, img_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, size + 5, f'{size}px',
           ha='center', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# All metrics heatmap
ax = axes[3]
ax.axis('off')
metrics_to_show = ['accuracy', 'top3', 'top5', 'macro_f1', 'weighted_f1']
data_matrix = np.array([[models[m][met] for met in metrics_to_show] for m in model_names])

im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(metrics_to_show)))
ax.set_yticks(np.arange(len(model_names)))
ax.set_xticklabels([metric_names_display[m] for m in metrics_to_show], fontsize=9)
ax.set_yticklabels(model_names, fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add text annotations
for i in range(len(model_names)):
    for j in range(len(metrics_to_show)):
        text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=8)

cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Score', fontsize=9)

fig.suptitle('Cross-Model Performance Comparison', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'optionB_v3_results/all_models_comparison_radar.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: all_models_comparison_radar.png")

# ============================================================
# VISUALIZATION 2: Detailed Bar Comparison
# ============================================================
print("\nGenerating detailed bar comparison...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics_groups = {
    'Accuracy': axes[0, 0],
    'Top-3 Accuracy': axes[0, 1],
    'Top-5 Accuracy': axes[0, 2],
    'Macro F1': axes[1, 0],
    'Weighted F1': axes[1, 1],
}

metric_keys = {
    'Accuracy': 'accuracy',
    'Top-3 Accuracy': 'top3',
    'Top-5 Accuracy': 'top5',
    'Macro F1': 'macro_f1',
    'Weighted F1': 'weighted_f1',
}

model_names_list = list(models.keys())
x_pos = np.arange(len(model_names_list))

for title, ax in metrics_groups.items():
    metric_key = metric_keys[title]
    values = [models[m][metric_key] for m in model_names_list]
    colors_list = [models[m]['color'] for m in model_names_list]

    bars = ax.bar(x_pos, values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names_list, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

# Model comparison info in the last subplot
ax_info = axes[1, 2]
ax_info.axis('off')

info_text = """
╔═══════════════════════════════════════════╗
║    Model Architecture Summary              ║
╠═══════════════════════════════════════════╣
║                                           ║
║ Baseline: B0, 128px, 122 classes         ║
║ Option B: B0, 128px, 40 classes          ║
║ B Improv: B0, 224px, 40 classes          ║
║ V2:       B2, 260px, 48 classes, TTA     ║
║ V3-Focal: B4, 380px, 48 classes, YOLO   ║
║ V3-ArcFace: B4, 380px, 48 classes       ║
║ V3-SupCon: B4, 380px, 48 classes, TTA   ║
║                                           ║
║ Best: V3-SupCon (79.5% accuracy)         ║
║ +27.2% improvement over Baseline          ║
║                                           ║
╚═══════════════════════════════════════════╝
"""
ax_info.text(0.5, 0.5, info_text, transform=ax_info.transAxes,
            fontfamily='monospace', fontsize=8, verticalalignment='center',
            horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'optionB_v3_results/all_models_bar_comparison.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: all_models_bar_comparison.png")

# ============================================================
# VISUALIZATION 3: Improvement Trajectory
# ============================================================
print("\nGenerating improvement trajectory...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_sequence = ['Baseline', 'Option B', 'B Improved', 'V2', 'V3-SupCon']
x_pos_seq = np.arange(len(model_sequence))

# Accuracy trajectory
ax = axes[0]
accuracy_vals = [models[m]['accuracy'] for m in model_sequence]
colors_seq = [models[m]['color'] for m in model_sequence]

ax.plot(x_pos_seq, accuracy_vals, 'o-', linewidth=3, markersize=10, color='#2c3e50', alpha=0.7)
for i, (x, y, model) in enumerate(zip(x_pos_seq, accuracy_vals, model_sequence)):
    ax.scatter(x, y, s=300, c=[models[model]['color']], edgecolor='black', linewidth=2, zorder=5)
    ax.text(x, y + 0.02, f'{y:.1%}', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Accuracy Improvement Trajectory', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos_seq)
ax.set_xticklabels(model_sequence, fontsize=10)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Add percentage improvements
for i in range(1, len(model_sequence)):
    prev = accuracy_vals[i-1]
    curr = accuracy_vals[i]
    improvement = (curr - prev) / prev * 100
    mid_x = (x_pos_seq[i-1] + x_pos_seq[i]) / 2
    mid_y = (prev + curr) / 2
    ax.text(mid_x, mid_y - 0.05, f'+{improvement:.1f}%', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.4))

# F1 trajectory
ax = axes[1]
f1_vals = [models[m]['weighted_f1'] for m in model_sequence]

ax.plot(x_pos_seq, f1_vals, 's-', linewidth=3, markersize=10, color='#c0392b', alpha=0.7)
for i, (x, y, model) in enumerate(zip(x_pos_seq, f1_vals, model_sequence)):
    ax.scatter(x, y, s=300, c=[models[model]['color']], edgecolor='black', linewidth=2, zorder=5)
    ax.text(x, y + 0.02, f'{y:.3f}', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Weighted F1 Score', fontsize=12, fontweight='bold')
ax.set_title('Weighted F1 Improvement Trajectory', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos_seq)
ax.set_xticklabels(model_sequence, fontsize=10)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# Add absolute improvements
for i in range(1, len(model_sequence)):
    prev = f1_vals[i-1]
    curr = f1_vals[i]
    improvement = curr - prev
    mid_x = (x_pos_seq[i-1] + x_pos_seq[i]) / 2
    mid_y = (prev + curr) / 2
    ax.text(mid_x, mid_y - 0.05, f'+{improvement:.4f}', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.4))

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'optionB_v3_results/improvement_trajectory.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: improvement_trajectory.png")

# ============================================================
# COMPREHENSIVE TEXT REPORT
# ============================================================
print("\nGenerating comprehensive comparison report...")
report_path = os.path.join(BASE_DIR, 'optionB_v3_results/all_models_analysis.txt')

with open(report_path, 'w') as f:
    f.write("="*100 + "\n")
    f.write("COMPREHENSIVE CROSS-MODEL ANALYSIS — LEGO MINIFIG CLASSIFICATION PROJECT\n")
    f.write("="*100 + "\n\n")

    f.write("EXECUTIVE SUMMARY\n")
    f.write("-"*100 + "\n")
    f.write("This analysis compares 7 model variants from Baseline through V3-SupCon, tracking the progression\n")
    f.write("of improvements in accuracy, F1 score, and other metrics through systematic enhancements.\n\n")

    f.write("OVERALL RANKINGS\n")
    f.write("-"*100 + "\n")
    f.write(f"{'Rank':<6} {'Model':<20} {'Accuracy':>12} {'Top-3':>12} {'Macro F1':>12} {'Weighted F1':>12}\n")
    f.write("-"*100 + "\n")

    sorted_models = sorted(models.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for rank, (model_name, model_data) in enumerate(sorted_models, 1):
        f.write(f"{rank:<6} {model_name:<20} {model_data['accuracy']:>12.4f} "
               f"{model_data['top3']:>12.4f} {model_data['macro_f1']:>12.4f} "
               f"{model_data['weighted_f1']:>12.4f}\n")
    f.write("\n")

    f.write("DETAILED MODEL SPECIFICATIONS\n")
    f.write("-"*100 + "\n")
    for model_name, model_data in models.items():
        f.write(f"\n{model_name}\n")
        f.write("-" * 100 + "\n")
        f.write(f"  Architecture:     {model_data['arch']}\n")
        f.write(f"  Input Resolution: {model_data['img_size']}x{model_data['img_size']} pixels\n")
        f.write(f"  Num Classes:      {model_data['classes']}\n")
        f.write(f"  Key Features:     {model_data['features']}\n")
        f.write(f"  Accuracy:         {model_data['accuracy']:.4f} ({model_data['accuracy']*100:.2f}%)\n")
        f.write(f"  Top-3 Accuracy:   {model_data['top3']:.4f} ({model_data['top3']*100:.2f}%)\n")
        f.write(f"  Top-5 Accuracy:   {model_data['top5']:.4f} ({model_data['top5']*100:.2f}%)\n")
        f.write(f"  Macro F1:         {model_data['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1:      {model_data['weighted_f1']:.4f}\n")

    f.write("\n\n")
    f.write("IMPROVEMENT ANALYSIS\n")
    f.write("-"*100 + "\n")
    baseline_acc = models['Baseline']['accuracy']
    v3_acc = models['V3-SupCon']['accuracy']
    acc_improvement = (v3_acc - baseline_acc) / baseline_acc * 100

    baseline_f1 = models['Baseline']['weighted_f1']
    v3_f1 = models['V3-SupCon']['weighted_f1']
    f1_improvement = (v3_f1 - baseline_f1) / baseline_f1 * 100

    f.write(f"\nBaseline vs V3-SupCon (Best):\n")
    f.write(f"  Accuracy:      {baseline_acc:.4f} → {v3_acc:.4f} (+{acc_improvement:.1f}%)\n")
    f.write(f"  Weighted F1:   {baseline_f1:.4f} → {v3_f1:.4f} (+{f1_improvement:.1f}%)\n")
    f.write(f"  Absolute Gain: {v3_acc - baseline_acc:+.4f} accuracy points\n\n")

    # Improvements per model
    model_sequence = ['Baseline', 'Option B', 'B Improved', 'V2', 'V3-SupCon']
    f.write("Incremental Improvements:\n")
    for i in range(1, len(model_sequence)):
        prev = model_sequence[i-1]
        curr = model_sequence[i]
        prev_acc = models[prev]['accuracy']
        curr_acc = models[curr]['accuracy']
        improvement = curr_acc - prev_acc
        pct_improvement = (curr_acc - prev_acc) / prev_acc * 100
        f.write(f"  {prev:15} → {curr:15}: {improvement:+.4f} ({pct_improvement:+.1f}%)\n")

    f.write("\n\nKEY INNOVATIONS BY MODEL\n")
    f.write("-"*100 + "\n")
    innovations = {
        'Baseline': 'Standard training pipeline, no augmentation or special techniques',
        'Option B': 'Category merging to reduce class imbalance (122→40 classes)',
        'B Improved': 'Higher resolution (128→224px), Focal loss, CutMix, MixUp augmentation',
        'V2': 'Larger backbone (B0→B2), Town subcategory splitting (40→48 classes), TTA',
        'V3-Focal': 'YOLO-based minifig cropping, synthetic augmentation, EfficientNet-B4',
        'V3-ArcFace': 'V3 architecture with ArcFace metric learning loss function',
        'V3-SupCon': 'V3 architecture with SupCon contrastive loss + TTA (BEST)',
    }
    for model_name, innovation in innovations.items():
        f.write(f"\n{model_name:15} — {innovation}\n")

    f.write("\n\nPERFORMANCE INSIGHTS\n")
    f.write("-"*100 + "\n")
    f.write("\n1. ACCURACY TREND\n")
    f.write("   • Baseline (29.3%) establishes lower bound\n")
    f.write("   • Option B (58.9%) nearly doubles accuracy through class merging\n")
    f.write("   • B Improved (74.4%) shows power of augmentation + higher resolution\n")
    f.write("   • V2 (74.9%) adds marginal gains through better architecture + TTA\n")
    f.write("   • V3-SupCon (79.5%) achieves best performance through comprehensive innovations\n\n")

    f.write("2. TOP-K ACCURACY ANALYSIS\n")
    f.write("   • Top-3 accuracy shows model confidence in top predictions\n")
    f.write("   • V3-SupCon: 91.55% (Top-3), 94.63% (Top-5) indicates strong ranking\n")
    f.write("   • Baseline: 48.21% (Top-3), 58.18% (Top-5) shows poor confidence\n\n")

    f.write("3. LOSS FUNCTION IMPACT (V3 Variants)\n")
    v3_focal = models['V3-Focal']['accuracy']
    v3_arcface = models['V3-ArcFace']['accuracy']
    v3_supcon = models['V3-SupCon']['accuracy']
    f.write(f"   • Focal Loss:     {v3_focal:.4f} (77.35%)\n")
    f.write(f"   • ArcFace Loss:   {v3_arcface:.4f} (74.36%) [weakest variant]\n")
    f.write(f"   • SupCon Loss:    {v3_supcon:.4f} (79.50%) [best variant]\n")
    f.write(f"   ✓ SupCon outperforms Focal by {(v3_supcon-v3_focal)*100:.2f} percentage points\n\n")

    f.write("4. ARCHITECTURAL PROGRESSION\n")
    f.write("   • EfficientNet-B0 → B2 → B4 progression increases model capacity\n")
    f.write("   • Resolution: 128px → 224px → 260px → 380px enables finer feature detection\n")
    f.write("   • Larger architectures benefit from YOLO preprocessing + synthetic augmentation\n\n")

    f.write("5. TRAINING TECHNIQUES IMPACT\n")
    f.write("   • YOLO-based minifig cropping: Removes background noise\n")
    f.write("   • Synthetic augmentation: Addresses class imbalance\n")
    f.write("   • TTA (Test-Time Augmentation): Provides 0.3-1% improvement\n")
    f.write("   • Class weighting: Improves minority class performance\n\n")

    f.write("RECOMMENDATIONS\n")
    f.write("-"*100 + "\n")
    f.write("✓ Use V3-SupCon for production deployment (best accuracy/robustness trade-off)\n")
    f.write("✓ Consider B Improved if computational budget is limited (74.4% accuracy, simpler model)\n")
    f.write("✓ SupCon loss superior to ArcFace for this dataset\n")
    f.write("✓ Future improvements: ensemble methods, additional synthetic augmentation, hyperparameter tuning\n")

print(f"✓ Saved: {report_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("CROSS-MODEL ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated Visualizations:")
print("  1. all_models_comparison_radar.png  - Radar & heatmap comparison")
print("  2. all_models_bar_comparison.png    - Detailed bar chart comparison")
print("  3. improvement_trajectory.png       - Accuracy/F1 improvement over time")
print(f"\nGenerated Reports:")
print(f"  4. all_models_analysis.txt         - Comprehensive cross-model analysis")
print(f"\nKey Finding:")
print(f"  • V3-SupCon achieves 79.5% accuracy (27.2% improvement over Baseline)")
print(f"  • SupCon loss outperforms ArcFace and Focal loss variants")
print("\nAll files saved to:", os.path.join(BASE_DIR, 'optionB_v3_results'))
