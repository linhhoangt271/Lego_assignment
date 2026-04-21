"""
Live Training Dashboard — Monitors all model training and displays live metrics.
Auto-refreshes every 30 seconds. Close window to stop.
"""

import re, time, os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE_DIR = "/Users/linh/Downloads/KUL /11. Advanced Analytics/big_data_assignment_2"
TMP_DIR = "/private/tmp/claude-502/-Users-linh-Downloads/0c93552d-a449-4fb4-87ec-3a1394222342/tasks"

# ============================================================
# HARDCODED LOG PATHS (the correct ones for each model)
# ============================================================
LOG_FILES = {
    'Baseline':    os.path.join(TMP_DIR, 'bz13it5xr.output'),
    'Option B':    os.path.join(TMP_DIR, 'bs6t3bcpi.output'),
    'B Improved':  os.path.join(TMP_DIR, 'bonb4zybn.output'),
    'V2':          os.path.join(TMP_DIR, 'bidttxvy4.output'),
}

COLORS = {
    'Baseline':   '#e74c3c',
    'Option B':   '#f39c12',
    'B Improved': '#2ecc71',
    'V2':         '#3498db',
}

FINAL_TEST = {
    'Baseline':   {'Accuracy': 0.293, 'Macro F1': 0.322, 'Weighted F1': 0.273, 'Top-3': 0.482, 'Top-5': 0.582},
    'Option B':   {'Accuracy': 0.589, 'Macro F1': 0.573, 'Weighted F1': 0.578, 'Top-3': 0.853, 'Top-5': 0.929},
    'B Improved': {'Accuracy': 0.744, 'Macro F1': 0.693, 'Weighted F1': 0.749, 'Top-3': 0.921, 'Top-5': 0.959},
}

EPOCH_RE = re.compile(
    r'Epoch\s+(\d+)/(\d+)\s*\|\s*'
    r'Train Loss:\s*([\d.]+)\s+Acc:\s*([\d.]+)\s*\|\s*'
    r'Val Loss:\s*([\d.]+)\s+Acc:\s*([\d.]+)'
    r'(?:\s+F1:\s*([\d.]+))?'
)

def parse_log(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        text = f.read()
    results = []
    for m in EPOCH_RE.finditer(text):
        results.append({
            'epoch': int(m.group(1)), 'total': int(m.group(2)),
            'train_loss': float(m.group(3)), 'train_acc': float(m.group(4)),
            'val_loss': float(m.group(5)), 'val_acc': float(m.group(6)),
            'val_f1': float(m.group(7)) if m.group(7) else None,
        })
    return results

# ============================================================
# BUILD DASHBOARD
# ============================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor('#0f0f23')

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.40, wspace=0.30,
                       top=0.92, bottom=0.05, left=0.05, right=0.98)

ax_tl = fig.add_subplot(gs[0, 0])   # Train Loss
ax_vl = fig.add_subplot(gs[0, 1])   # Val Loss
ax_va = fig.add_subplot(gs[0, 2])   # Val Accuracy
ax_vf = fig.add_subplot(gs[0, 3])   # Val F1
ax_ta = fig.add_subplot(gs[1, 0])   # Train Accuracy
ax_gap = fig.add_subplot(gs[1, 1])  # Overfit gap
ax_bar = fig.add_subplot(gs[1, 2:]) # Final comparison bars
ax_st = fig.add_subplot(gs[2, :])   # Status

def style(ax, title, ylabel=''):
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch', color='#888', fontsize=8)
    ax.set_ylabel(ylabel, color='#888', fontsize=8)
    ax.tick_params(colors='#888', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333')
    ax.grid(True, alpha=0.12)

def draw():
    for ax in [ax_tl, ax_vl, ax_va, ax_vf, ax_ta, ax_gap, ax_bar, ax_st]:
        ax.clear()

    style(ax_tl, 'Train Loss', 'Loss')
    style(ax_vl, 'Validation Loss', 'Loss')
    style(ax_va, 'Validation Accuracy', 'Accuracy')
    style(ax_vf, 'Validation Macro F1', 'F1')
    style(ax_ta, 'Train Accuracy', 'Accuracy')
    style(ax_gap, 'Overfit Gap (Train - Val Acc)', 'Gap')
    style(ax_bar, 'Final Test Metrics', '')
    ax_st.set_facecolor('#1a1a2e')
    ax_st.axis('off')

    status_info = []

    for name in ['Baseline', 'Option B', 'B Improved', 'V2']:
        color = COLORS[name]
        epochs = parse_log(LOG_FILES.get(name, ''))

        if not epochs:
            status_info.append((name, color, 'No log found', '-'))
            continue

        ep = [e['epoch'] for e in epochs]
        kw = dict(color=color, linewidth=2, label=name, marker='o', markersize=3, alpha=0.9)

        ax_tl.plot(ep, [e['train_loss'] for e in epochs], **kw)
        ax_vl.plot(ep, [e['val_loss'] for e in epochs], **kw)
        ax_va.plot(ep, [e['val_acc'] for e in epochs], **kw)
        ax_ta.plot(ep, [e['train_acc'] for e in epochs], **kw)

        # Overfit gap
        gap = [e['train_acc'] - e['val_acc'] for e in epochs]
        ax_gap.plot(ep, gap, **kw)
        ax_gap.axhline(y=0, color='white', linewidth=0.5, alpha=0.3)

        # Val F1 (V2 only)
        if epochs[0]['val_f1'] is not None:
            ax_vf.plot(ep, [e['val_f1'] for e in epochs], **kw)

        # Status
        cur, tot = epochs[-1]['epoch'], epochs[-1]['total']
        best_acc = max(e['val_acc'] for e in epochs)
        best_f1 = max((e['val_f1'] for e in epochs if e['val_f1'] is not None), default=None)
        done = cur >= tot or any('Early stopping' in open(LOG_FILES[name]).read() for _ in [1] if os.path.exists(LOG_FILES[name]))

        if done:
            status_str = f'Completed ({cur} epochs)'
        else:
            status_str = f'Training... Epoch {cur}/{tot}'

        metrics_str = f'Best Val Acc: {best_acc:.1%}'
        if best_f1 is not None:
            metrics_str += f' | Best Val F1: {best_f1:.4f}'

        status_info.append((name, color, status_str, metrics_str))

    # Legends
    for ax in [ax_tl, ax_vl, ax_va, ax_ta, ax_gap]:
        leg = ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    if ax_vf.get_legend_handles_labels()[1]:
        ax_vf.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')

    # Final comparison bar chart
    models = list(FINAL_TEST.keys())
    # Add V2 if it has completed
    v2_epochs = parse_log(LOG_FILES.get('V2', ''))
    # Check if V2 finished (has test metrics in log)
    v2_log = ''
    if os.path.exists(LOG_FILES.get('V2', '')):
        with open(LOG_FILES['V2']) as f:
            v2_log = f.read()
    v2_acc_match = re.search(r'Accuracy\s+([\d.]+)', v2_log[v2_log.find('EVALUATING'):] if 'EVALUATING' in v2_log else '')
    if v2_acc_match:
        # Parse V2 test metrics from log
        eval_section = v2_log[v2_log.find('EVALUATING'):]
        v2_metrics = {}
        for line in eval_section.split('\n'):
            line = line.strip()
            for metric_name in ['Accuracy', 'Macro F1', 'Weighted F1', 'Top-3 Accuracy', 'Top-5 Accuracy']:
                if line.startswith(metric_name):
                    parts = line.split()
                    try:
                        v2_metrics[metric_name] = float(parts[-1])
                    except:
                        pass
        if v2_metrics:
            FINAL_TEST['V2'] = {
                'Accuracy': v2_metrics.get('Accuracy', 0),
                'Macro F1': v2_metrics.get('Macro F1', 0),
                'Weighted F1': v2_metrics.get('Weighted F1', 0),
                'Top-3': v2_metrics.get('Top-3 Accuracy', 0),
                'Top-5': v2_metrics.get('Top-5 Accuracy', 0),
            }
            models = list(FINAL_TEST.keys())

    if models:
        x = np.arange(len(models))
        w = 0.18
        metrics_to_plot = ['Accuracy', 'Macro F1', 'Weighted F1', 'Top-3', 'Top-5']
        bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for i, (metric, bc) in enumerate(zip(metrics_to_plot, bar_colors)):
            vals = [FINAL_TEST[m].get(metric, 0) for m in models]
            bars = ax_bar.bar(x + (i - 2) * w, vals, w, label=metric, color=bc, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax_bar.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                                    xytext=(0, 2), textcoords='offset points',
                                    ha='center', va='bottom', fontsize=5.5, color='white')

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels([COLORS.get(m, m) for m in models], fontsize=1)  # hide, use colors
        # Custom x labels with colors
        for i, m in enumerate(models):
            ax_bar.text(i, -0.06, m, ha='center', va='top', fontsize=8, color=COLORS.get(m, 'white'),
                       fontweight='bold', transform=ax_bar.get_xaxis_transform())
        ax_bar.set_ylim(0, 1.08)
        ax_bar.legend(fontsize=6, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white',
                     ncol=5, loc='upper left')

    # Status panel
    headers = [('Model', 0.02), ('Status', 0.22), ('Best Metrics', 0.52)]
    for label, xpos in headers:
        ax_st.text(xpos, 0.95, label, color='#666', fontsize=9, fontweight='bold',
                   transform=ax_st.transAxes, va='top', family='monospace')

    ax_st.axhline(y=0.88, xmin=0.01, xmax=0.99, color='#333', linewidth=0.5,
                  transform=ax_st.transAxes)

    for i, (name, color, status, metrics) in enumerate(status_info):
        y = 0.78 - i * 0.20
        is_running = 'Training...' in status
        icon = '⬤' if is_running else '✔'
        icon_color = '#f1c40f' if is_running else '#2ecc71'

        ax_st.text(0.02, y, f'{icon} {name}', color=color, fontsize=11, fontweight='bold',
                   transform=ax_st.transAxes, va='top')
        ax_st.text(0.22, y, status, color=icon_color if is_running else '#aaa', fontsize=10,
                   transform=ax_st.transAxes, va='top')
        ax_st.text(0.52, y, metrics, color='#ddd', fontsize=9,
                   transform=ax_st.transAxes, va='top', family='monospace')

    timestamp = time.strftime('%H:%M:%S')
    fig.suptitle(f'LEGO Minifigure Classification — Training Dashboard        Last refresh: {timestamp}',
                 color='white', fontsize=13, fontweight='bold')

# ============================================================
# REFRESH LOOP
# ============================================================
print("Dashboard running. Close the window to stop.")
print("Auto-refreshes every 30 seconds.\n")

draw()
plt.ion()
plt.show()

try:
    while plt.fignum_exists(fig.number):
        plt.pause(30)
        draw()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
except KeyboardInterrupt:
    pass

print("Dashboard closed.")
