#!/usr/bin/env python3
"""Generate paper figures from experiment results."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ============================================================
# Data from experiments
# ============================================================

categories = ['AR', 'ALG', 'WP', 'LOG']

# End-to-end task accuracy (%)
e2e_data = {
    'L* routing → V5':      [100, 100, 12, 100],
    'RPNI routing → V5':    [96, 96, 13, 92],
    'Tool-calling agent':   [95, 76, 5, 59],
}
e2e_overall = {
    'L* routing → V5': 78.0,
    'RPNI routing → V5': 74.3,
    'Tool-calling agent': 58.7,
}

# Efficiency data
efficiency = {
    'L* routing → V5':    {'latency': 6907, 'energy': 33.12},
    'RPNI routing → V5':  {'latency': 6567, 'energy': 30.24},
    'Tool-calling agent': {'latency': 16439, 'energy': 74.96},
}

# Classification accuracy (%)
classify_data = {
    'L* (SLM oracle)':    [100, 100, 100, 100],
    'L* (feature oracle)': [100, 100, 100, 100],
    'RPNI (passive)':     [96, 92, 64, 92],
    'Feature classifier': [100, 100, 100, 100],
}
classify_overall = {
    'L* (SLM oracle)': 100.0,
    'L* (feature oracle)': 100.0,
    'RPNI (passive)': 86.0,
    'Feature classifier': 100.0,
}

# V1-V5 development ablation
v1v5_systems = ['SLM\nbaseline', 'V1\n(+A5)', 'V2\n(+A4)', 'V3.1\n(+A3)', 'V5\n(30tok)', 'V5\n(512tok)']
v1v5_acc = [37.3, 53.0, 67.0, 70.7, 79.7, 99.0]

# DFA sizes
dfa_sizes = {
    'RPNI': {'AR': 1, 'ALG': 2, 'WP': 2, 'LOG': 1},
    'L* (feature)': {'AR': 9, 'ALG': 2, 'WP': 47, 'LOG': 7},
    'L* (SLM)': {'AR': 19, 'ALG': 2, 'WP': 28, 'LOG': 7},
}

# Colors
COLORS = {
    'L* routing → V5': '#2196F3',
    'RPNI routing → V5': '#FF9800',
    'Tool-calling agent': '#F44336',
}
COLORS_CLASSIFY = {
    'L* (SLM oracle)': '#2196F3',
    'L* (feature oracle)': '#64B5F6',
    'RPNI (passive)': '#FF9800',
    'Feature classifier': '#9E9E9E',
}

# ============================================================
# Figure 1: End-to-end task accuracy comparison (THE KEY FIGURE)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(categories))
width = 0.25
for i, (name, vals) in enumerate(e2e_data.items()):
    bars = ax.bar(x + i * width, vals, width, label=f"{name} ({e2e_overall[name]:.1f}%)",
                  color=COLORS[name], edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{v}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Task Accuracy (%)')
ax.set_title('End-to-End Task Accuracy by Category')
ax.set_xticks(x + width)
ax.set_xticklabels(categories)
ax.set_ylim(0, 115)
ax.legend(loc='upper right', fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig9_e2e_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated fig9_e2e_comparison.png")

# ============================================================
# Figure 2: Efficiency comparison (latency + energy side by side)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

systems = list(efficiency.keys())
colors = [COLORS[s] for s in systems]

# Latency
latencies = [efficiency[s]['latency'] for s in systems]
bars = ax1.barh(systems, latencies, color=colors, edgecolor='white')
for bar, v in zip(bars, latencies):
    ax1.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
             f'{v:,.0f} ms', va='center', fontsize=9)
ax1.set_xlabel('Mean Latency (ms)')
ax1.set_title('Latency')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, max(latencies) * 1.3)

# Energy
energies = [efficiency[s]['energy'] for s in systems]
bars = ax2.barh(systems, energies, color=colors, edgecolor='white')
for bar, v in zip(bars, energies):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{v:.1f} mWh', va='center', fontsize=9)
ax2.set_xlabel('Energy per Prompt (mWh)')
ax2.set_title('Energy')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(0, max(energies) * 1.3)

fig.suptitle('Efficiency: Learned Routing vs. Tool-Calling Agent', fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig10_efficiency.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated fig10_efficiency.png")

# ============================================================
# Figure 3: Classification accuracy comparison
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(categories))
width = 0.2
for i, (name, vals) in enumerate(classify_data.items()):
    bars = ax.bar(x + i * width, vals, width, label=f"{name} ({classify_overall[name]:.0f}%)",
                  color=list(COLORS_CLASSIFY.values())[i], edgecolor='white', linewidth=0.5)

ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('Routing Classification Accuracy on Held-Out T2')
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(categories)
ax.set_ylim(0, 115)
ax.legend(loc='lower right', fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig11_classification.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated fig11_classification.png")

# ============================================================
# Figure 4: V1-V5 system progression (updated version of fig1)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 3.5))
colors_prog = ['#F44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50', '#2196F3']
bars = ax.bar(v1v5_systems, v1v5_acc, color=colors_prog, edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, v1v5_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Overall Accuracy (%)')
ax.set_title('System Development Progression (V1–V5)')
ax.set_ylim(0, 110)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig1_system_progression.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated fig1_system_progression.png (updated)")

# ============================================================
# Figure 5: DFA sizes comparison
# ============================================================
fig, ax = plt.subplots(figsize=(6, 3.5))
x = np.arange(len(categories))
width = 0.25
dfa_colors = ['#FF9800', '#64B5F6', '#2196F3']
for i, (name, sizes) in enumerate(dfa_sizes.items()):
    vals = [sizes[c] for c in categories]
    bars = ax.bar(x + i * width, vals, width, label=name, color=dfa_colors[i],
                  edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(v), ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Number of DFA States')
ax.set_title('Learned DFA Sizes by Method')
ax.set_xticks(x + width)
ax.set_xticklabels(categories)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig12_dfa_sizes.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated fig12_dfa_sizes.png")

print("\nAll figures generated in", OUT_DIR)
