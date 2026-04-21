"""Comprehensive V3 Model Analysis and Visualization"""
import json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

BASE_DIR = "/home/test/big_data_assignment_2"
OUTPUT_DIR = os.path.join(BASE_DIR, "optionB_v3_results")

# ============================================================
# PARSING CLASSIFICATION REPORTS
# ============================================================
def parse_classification_report(filepath):
    """Parse sklearn classification_report text output."""
    with open(filepath) as f:
        lines = f.readlines()

    data = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('precision') or '====' in line:
            continue

        parts = line.split()
        if len(parts) >= 4:
            try:
                # Try to parse as class result
                precision = float(parts[-4])
                recall = float(parts[-3])
                f1 = float(parts[-2])
                support = int(parts[-1])

                class_name = ' '.join(parts[:-4])
                if class_name and not class_name.startswith('accuracy'):
                    data[class_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'support': support
                    }
            except (ValueError, IndexError):
                pass

    return data

print("Parsing V3-SupCon classification report...")
v3_report = parse_classification_report(os.path.join(OUTPUT_DIR, 'classification_report.txt'))
df_v3 = pd.DataFrame(v3_report).T
df_v3 = df_v3.sort_values('f1', ascending=False)

print(f"Found {len(df_v3)} classes")
print("\nTop 10 classes by F1 score:")
print(df_v3.head(10)[['f1', 'precision', 'recall', 'support']])

print("\nBottom 10 classes by F1 score:")
print(df_v3.tail(10)[['f1', 'precision', 'recall', 'support']])

# ============================================================
# VISUALIZATION 1: Per-Class F1 Score Heatmap
# ============================================================
print("\nGenerating per-class F1 score visualization...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

# Top 20 classes
top_20 = df_v3.head(20)[['f1', 'precision', 'recall']].sort_values('f1')
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_20)))
y_pos = np.arange(len(top_20))
ax1.barh(y_pos, top_20['f1'], color=colors, alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_20.index, fontsize=9)
ax1.set_xlabel('F1 Score', fontsize=10)
ax1.set_title('Top 20 Classes by F1 Score (V3-SupCon)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(top_20.iterrows()):
    ax1.text(row['f1'] + 0.01, i, f"{row['f1']:.3f}", va='center', fontsize=8)

# Bottom 10 classes
bottom_10 = df_v3.tail(10)[['f1', 'precision', 'recall']].sort_values('f1')
colors_bot = plt.cm.RdYlGn(np.linspace(0.2, 0.7, len(bottom_10)))
y_pos_bot = np.arange(len(bottom_10))
ax2.barh(y_pos_bot, bottom_10['f1'], color=colors_bot, alpha=0.8)
ax2.set_yticks(y_pos_bot)
ax2.set_yticklabels(bottom_10.index, fontsize=9)
ax2.set_xlabel('F1 Score', fontsize=10)
ax2.set_title('Bottom 10 Classes by F1 Score (V3-SupCon)', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (idx, row) in enumerate(bottom_10.iterrows()):
    ax2.text(row['f1'] + 0.01, i, f"{row['f1']:.3f}", va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'v3_per_class_f1_detailed.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: v3_per_class_f1_detailed.png")

# ============================================================
# VISUALIZATION 2: Precision-Recall-F1 Comparison by Class
# ============================================================
print("\nGenerating precision-recall-F1 scatter plot...")
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(df_v3['precision'], df_v3['recall'],
                    s=df_v3['support']*2,
                    c=df_v3['f1'],
                    cmap='RdYlGn',
                    alpha=0.6,
                    vmin=0, vmax=1)

ax.set_xlabel('Precision', fontsize=11)
ax.set_ylabel('Recall', fontsize=11)
ax.set_title('Precision vs Recall by Class (bubble size = support, color = F1 score)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('F1 Score', fontsize=10)

# Annotate outliers/interesting classes
interesting = pd.concat([df_v3.nsmallest(3, 'f1'), df_v3.nlargest(3, 'f1')])
for idx, row in interesting.iterrows():
    ax.annotate(idx, (row['precision'], row['recall']),
               fontsize=7, alpha=0.7,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.2))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'v3_precision_recall_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: v3_precision_recall_scatter.png")

# ============================================================
# VISUALIZATION 3: Class Distribution and Performance
# ============================================================
print("\nGenerating class distribution vs performance plot...")
fig, ax = plt.subplots(figsize=(14, 8))

df_sorted = df_v3.sort_values('support', ascending=False).head(30)
x = np.arange(len(df_sorted))
width = 0.35

bars1 = ax.bar(x - width/2, df_sorted['f1']*100, width, label='F1 Score (%)', alpha=0.8, color='#2ecc71')
bars2 = ax.bar(x + width/2, df_sorted['support']/df_sorted['support'].max()*100, width,
              label='Relative Support (%)', alpha=0.8, color='#3498db')

ax.set_ylabel('Score / Relative Support (%)', fontsize=11)
ax.set_title('Top 30 Classes by Support: F1 Score vs Class Size', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_sorted.index, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'v3_class_distribution_vs_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: v3_class_distribution_vs_performance.png")

# ============================================================
# VISUALIZATION 4: Precision-Recall Tradeoff Analysis
# ============================================================
print("\nGenerating precision-recall tradeoff analysis...")
fig, ax = plt.subplots(figsize=(12, 8))

# Color by difference between precision and recall
diff = df_v3['precision'] - df_v3['recall']
colors_diff = plt.cm.RdBu_r((diff + 1) / 2)  # Map [-1, 1] to [0, 1]

scatter = ax.scatter(df_v3['precision'], df_v3['recall'],
                    s=200, c=diff, cmap='RdBu_r',
                    alpha=0.6, vmin=-1, vmax=1)

ax.set_xlabel('Precision', fontsize=11)
ax.set_ylabel('Recall', fontsize=11)
ax.set_title('Precision vs Recall: Precision-Biased (blue) vs Recall-Biased (red)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect Balance')
ax.legend()

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Precision - Recall Difference', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'v3_precision_recall_tradeoff.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: v3_precision_recall_tradeoff.png")

# ============================================================
# VISUALIZATION 5: Statistical Summary Dashboard
# ============================================================
print("\nGenerating statistical summary dashboard...")
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Distribution of F1 scores
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df_v3['f1'], bins=20, color='#3498db', alpha=0.7, edgecolor='black')
ax1.axvline(df_v3['f1'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df_v3['f1'].mean():.3f}")
ax1.axvline(df_v3['f1'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {df_v3['f1'].median():.3f}")
ax1.set_xlabel('F1 Score')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of F1 Scores')
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# Distribution of Precision scores
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df_v3['precision'], bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
ax2.axvline(df_v3['precision'].mean(), color='red', linestyle='--', linewidth=2,
           label=f"Mean: {df_v3['precision'].mean():.3f}")
ax2.set_xlabel('Precision')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Precision Scores')
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# Distribution of Recall scores
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df_v3['recall'], bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
ax3.axvline(df_v3['recall'].mean(), color='blue', linestyle='--', linewidth=2,
           label=f"Mean: {df_v3['recall'].mean():.3f}")
ax3.set_xlabel('Recall')
ax3.set_ylabel('Count')
ax3.set_title('Distribution of Recall Scores')
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# F1 vs Support
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(df_v3['support'], df_v3['f1'], alpha=0.6, s=100, color='#9b59b6')
ax4.set_xlabel('Support (samples)')
ax4.set_ylabel('F1 Score')
ax4.set_title('F1 Score vs Class Size')
ax4.grid(True, alpha=0.3)

# Precision vs Support
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(df_v3['support'], df_v3['precision'], alpha=0.6, s=100, color='#f39c12')
ax5.set_xlabel('Support (samples)')
ax5.set_ylabel('Precision')
ax5.set_title('Precision vs Class Size')
ax5.grid(True, alpha=0.3)

# Recall vs Support
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(df_v3['support'], df_v3['recall'], alpha=0.6, s=100, color='#1abc9c')
ax6.set_xlabel('Support (samples)')
ax6.set_ylabel('Recall')
ax6.set_title('Recall vs Class Size')
ax6.grid(True, alpha=0.3)

# Summary statistics table
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('off')

stats_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        V3-SupCon Model Summary Statistics                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Total Classes:     {len(df_v3):>3}        |  Total Samples:     {int(df_v3['support'].sum()):>5}      ║
║                                                                              ║
║  F1 Score:  Mean={df_v3['f1'].mean():.4f}  Median={df_v3['f1'].median():.4f}  Std={df_v3['f1'].std():.4f}      ║
║  Precision: Mean={df_v3['precision'].mean():.4f}  Median={df_v3['precision'].median():.4f}  Std={df_v3['precision'].std():.4f}      ║
║  Recall:    Mean={df_v3['recall'].mean():.4f}  Median={df_v3['recall'].median():.4f}  Std={df_v3['recall'].std():.4f}      ║
║                                                                              ║
║  Avg Support per Class: {df_v3['support'].mean():.1f}  (min={df_v3['support'].min()}, max={df_v3['support'].max()})  ║
║                                                                              ║
║  Classes with F1 >= 0.80:  {(df_v3['f1'] >= 0.80).sum():>2}   Classes with F1 < 0.50:  {(df_v3['f1'] < 0.50).sum():>2}      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
ax7.text(0.5, 0.5, stats_text, transform=ax7.transAxes,
        fontfamily='monospace', fontsize=9, verticalalignment='center',
        horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig(os.path.join(OUTPUT_DIR, 'v3_statistical_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: v3_statistical_summary.png")

# ============================================================
# DETAILED REPORT
# ============================================================
print("\nGenerating detailed text report...")
report_path = os.path.join(OUTPUT_DIR, 'v3_detailed_analysis.txt')
with open(report_path, 'w') as f:
    f.write("="*90 + "\n")
    f.write("V3-SupCon MODEL — COMPREHENSIVE ANALYSIS REPORT\n")
    f.write("="*90 + "\n\n")

    f.write("EXECUTIVE SUMMARY\n")
    f.write("-"*90 + "\n")
    f.write(f"Total Classes: {len(df_v3)}\n")
    f.write(f"Total Test Samples: {int(df_v3['support'].sum())}\n")
    f.write(f"Overall Accuracy: 79.50%\n")
    f.write(f"Overall Weighted F1: 0.7951\n")
    f.write(f"Overall Macro F1: 0.7406\n\n")

    f.write("PERFORMANCE METRICS SUMMARY\n")
    f.write("-"*90 + "\n")
    f.write(f"{'Metric':<20} {'Mean':>10} {'Median':>10} {'Std Dev':>10} {'Min':>10} {'Max':>10}\n")
    f.write("-"*90 + "\n")
    for metric in ['f1', 'precision', 'recall']:
        f.write(f"{metric.capitalize():<20} {df_v3[metric].mean():>10.4f} "
               f"{df_v3[metric].median():>10.4f} {df_v3[metric].std():>10.4f} "
               f"{df_v3[metric].min():>10.4f} {df_v3[metric].max():>10.4f}\n")
    f.write("\n")

    f.write("TOP 15 BEST PERFORMING CLASSES\n")
    f.write("-"*90 + "\n")
    f.write(f"{'Rank':<6} {'Class':<35} {'F1':>8} {'Precision':>10} {'Recall':>10} {'Support':>10}\n")
    f.write("-"*90 + "\n")
    for rank, (idx, row) in enumerate(df_v3.head(15).iterrows(), 1):
        f.write(f"{rank:<6} {idx:<35} {row['f1']:>8.4f} {row['precision']:>10.4f} "
               f"{row['recall']:>10.4f} {int(row['support']):>10}\n")
    f.write("\n")

    f.write("BOTTOM 15 WORST PERFORMING CLASSES\n")
    f.write("-"*90 + "\n")
    f.write(f"{'Rank':<6} {'Class':<35} {'F1':>8} {'Precision':>10} {'Recall':>10} {'Support':>10}\n")
    f.write("-"*90 + "\n")
    for rank, (idx, row) in enumerate(df_v3.tail(15).iloc[::-1].iterrows(), 1):
        f.write(f"{rank:<6} {idx:<35} {row['f1']:>8.4f} {row['precision']:>10.4f} "
               f"{row['recall']:>10.4f} {int(row['support']):>10}\n")
    f.write("\n")

    f.write("PERFORMANCE TIERS\n")
    f.write("-"*90 + "\n")
    f.write(f"Excellent (F1 >= 0.90): {(df_v3['f1'] >= 0.90).sum()} classes\n")
    f.write(f"Very Good (0.80 <= F1 < 0.90): {((df_v3['f1'] >= 0.80) & (df_v3['f1'] < 0.90)).sum()} classes\n")
    f.write(f"Good (0.70 <= F1 < 0.80): {((df_v3['f1'] >= 0.70) & (df_v3['f1'] < 0.80)).sum()} classes\n")
    f.write(f"Fair (0.50 <= F1 < 0.70): {((df_v3['f1'] >= 0.50) & (df_v3['f1'] < 0.70)).sum()} classes\n")
    f.write(f"Poor (F1 < 0.50): {(df_v3['f1'] < 0.50).sum()} classes\n")
    f.write("\n")

    f.write("PRECISION-RECALL ANALYSIS\n")
    f.write("-"*90 + "\n")
    high_precision = df_v3[df_v3['precision'] > 0.9]
    high_recall = df_v3[df_v3['recall'] > 0.9]
    high_both = df_v3[(df_v3['precision'] > 0.9) & (df_v3['recall'] > 0.9)]
    f.write(f"Classes with Precision > 0.90: {len(high_precision)}\n")
    f.write(f"Classes with Recall > 0.90: {len(high_recall)}\n")
    f.write(f"Classes with both > 0.90: {len(high_both)}\n\n")

    f.write("SAMPLE SIZE IMPACT\n")
    f.write("-"*90 + "\n")
    small_classes = df_v3[df_v3['support'] < 30]
    medium_classes = df_v3[(df_v3['support'] >= 30) & (df_v3['support'] < 100)]
    large_classes = df_v3[df_v3['support'] >= 100]

    f.write(f"Small classes (support < 30):  {len(small_classes):>3}  | Avg F1: {small_classes['f1'].mean():.4f}\n")
    f.write(f"Medium classes (30-100):       {len(medium_classes):>3}  | Avg F1: {medium_classes['f1'].mean():.4f}\n")
    f.write(f"Large classes (support >= 100): {len(large_classes):>3}  | Avg F1: {large_classes['f1'].mean():.4f}\n")
    f.write("\n")

    f.write("ALL CLASSES RANKED BY F1 SCORE\n")
    f.write("-"*90 + "\n")
    f.write(f"{'Rank':<6} {'Class':<35} {'F1':>8} {'Precision':>10} {'Recall':>10} {'Support':>10}\n")
    f.write("-"*90 + "\n")
    for rank, (idx, row) in enumerate(df_v3.iterrows(), 1):
        f.write(f"{rank:<6} {idx:<35} {row['f1']:>8.4f} {row['precision']:>10.4f} "
               f"{row['recall']:>10.4f} {int(row['support']):>10}\n")

print(f"✓ Saved: {report_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nGenerated Visualizations:")
print("  1. v3_per_class_f1_detailed.png        - Top 20 / Bottom 10 classes")
print("  2. v3_precision_recall_scatter.png     - Precision vs Recall scatter plot")
print("  3. v3_class_distribution_vs_performance.png - Class size vs F1 relationship")
print("  4. v3_precision_recall_tradeoff.png    - Precision-Recall tradeoff analysis")
print("  5. v3_statistical_summary.png          - Comprehensive dashboard with stats")
print(f"\nGenerated Reports:")
print(f"  6. v3_detailed_analysis.txt            - Full detailed analysis report")

print(f"\nKey Findings:")
print(f"  • {(df_v3['f1'] >= 0.90).sum()} classes with excellent F1 (>=0.90)")
print(f"  • {(df_v3['f1'] < 0.50).sum()} challenging classes with F1 < 0.50")
print(f"  • Larger classes (>=100 samples) avg F1: {large_classes['f1'].mean():.4f}")
print(f"  • Smaller classes (<30 samples) avg F1: {small_classes['f1'].mean():.4f}")
print("\nAll files saved to:", OUTPUT_DIR)
