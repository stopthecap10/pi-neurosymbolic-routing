#!/usr/bin/env python3
"""
Regenerate charts for the arXiv paper WITHOUT internal "Figure N." captions.
LaTeX \caption{} handles numbering; embedded captions cause conflicts.
Outputs to paper/figures/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

OUT_DIR = 'paper/figures'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BLUE = '#2563EB'
RED = '#DC2626'
GREEN = '#16A34A'
ORANGE = '#EA580C'
PURPLE = '#9333EA'
GRAY = '#6B7280'
LIGHT_BLUE = '#93C5FD'
LIGHT_RED = '#FCA5A5'


def fig1_system_progression():
    versions = ['Baseline\n(C2)', 'V1\n+A5', 'V2\n+A4', 'V3\n+A3', 'V4\n+Cal', 'V5\n+A6', 'V5\n+CoT512']
    accuracy = [37.3, 53.0, 67.0, 70.7, 70.7, 79.7, 99.0]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(range(len(versions)), accuracy, 'o-', color=BLUE, linewidth=2.5,
            markersize=10, markerfacecolor='white', markeredgewidth=2.5, zorder=3)
    ax.fill_between(range(len(versions)), accuracy, alpha=0.08, color=BLUE)

    ann_config = {
        1: ('+A5 Arithmetic\nParser',   (-30, -70)),
        2: ('+A4 SymPy\nAlgebra',       (30, 55)),
        5: ('+A6 Logic\nEngine',        (-30, -70)),
        6: ('+512 token\nCoT for WP',   (30, 55)),
    }
    for idx, (text, xyoff) in ann_config.items():
        ax.annotate(text, (idx, accuracy[idx]),
                    textcoords='offset points', xytext=xyoff,
                    ha='center', fontsize=11, color=BLUE,
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5))

    for i, v in enumerate(accuracy):
        if i in ann_config:
            continue
        ax.annotate(f'{v:.1f}%', (i, v), textcoords='offset points',
                    xytext=(0, 14), ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions)
    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('System Progression: Baseline to V5')
    ax.set_ylim(5, 120)
    ax.axhline(y=37.3, color=GRAY, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig1_system_progression.png'))
    plt.close(fig)
    print('  fig1 saved')


def fig2_category_accuracy():
    categories = ['AR', 'ALG', 'LOG', 'WP']
    baseline = [48.0, 40.0, 53.3, 8.0]
    v5_512 = [100.0, 100.0, 100.0, 96.0]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline (C2 Grammar)',
                   color=LIGHT_RED, edgecolor=RED, linewidth=1.5)
    bars2 = ax.bar(x + width/2, v5_512, width, label='V5 Hybrid (512 tok)',
                   color=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.5)

    for bar, val in zip(bars1, baseline):
        ax.text(bar.get_x() + bar.get_width()/2, val / 2,
                f'{val:.0f}%', ha='center', fontsize=12, color=RED,
                fontweight='bold', va='center')
    for bar, val in zip(bars2, v5_512):
        ax.text(bar.get_x() + bar.get_width()/2, val / 2,
                f'{val:.0f}%', ha='center', fontsize=12, color=BLUE,
                fontweight='bold', va='center')
    for i in range(len(categories)):
        delta = v5_512[i] - baseline[i]
        mid_x = x[i]
        ax.annotate(f'+{delta:.0f}%', (mid_x, max(v5_512[i], baseline[i]) + 4),
                    ha='center', fontsize=13, fontweight='bold', color=GREEN)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Category Accuracy: Baseline vs V5')
    ax.set_xticks(x)
    ax.set_xticklabels(['Arithmetic\n(AR)', 'Algebra\n(ALG)', 'Logic\n(LOG)', 'Word Problems\n(WP)'])
    ax.set_ylim(0, 130)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=True, edgecolor=GRAY)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(OUT_DIR, 'fig2_category_accuracy.png'))
    plt.close(fig)
    print('  fig2 saved')


def fig3_latency():
    categories = ['AR', 'ALG', 'LOG', 'WP']
    baseline_lat = [6288, 5838, 2672, 25600]
    v5_lat = [1, 11, 2, 137234]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars1 = ax.bar(x - width/2, baseline_lat, width, label='Baseline (C2 Grammar)',
                   color=LIGHT_RED, edgecolor=RED, linewidth=1.5)
    bars2 = ax.bar(x + width/2, v5_lat, width, label='V5 Hybrid (512 tok)',
                   color=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.5)

    ax.set_yscale('log')
    ax.set_ylabel('Median Latency (ms) — log scale')
    ax.set_title('Per-Category Latency: Baseline vs V5')
    ax.set_xticks(x)
    ax.set_xticklabels(['Arithmetic\n(AR)', 'Algebra\n(ALG)', 'Logic\n(LOG)', 'Word Problems\n(WP)'])
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.3, 800000)

    for bar, val in zip(bars1, baseline_lat):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.8,
                f'{val:,}', ha='center', fontsize=11, color=RED)
    for bar, val in zip(bars2, v5_lat):
        ax.text(bar.get_x() + bar.get_width()/2, val * 3.0,
                f'{val:,}', ha='center', fontsize=11, color=BLUE)

    speedups = ['6,288\u00d7', '531\u00d7', '1,336\u00d7', '']
    for i, s in enumerate(speedups):
        if s:
            ax.text(x[i], baseline_lat[i] * 4.0,
                    f'{s}\nfaster', ha='center', fontsize=11,
                    fontweight='bold', color=GREEN)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=True, edgecolor=GRAY)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(os.path.join(OUT_DIR, 'fig3_latency.png'))
    plt.close(fig)
    print('  fig3 saved')


def fig4_wp_ablation():
    tokens = [30, 300, 512]
    phi_wp_acc = [18.7, 92.0, 96.0]
    qwen_wp_acc = [18.7, 64.0, 96.0]
    phi_energy = [30.6, 147.2, 211.1]
    qwen_energy = [30.6, 113.1, 147.6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(tokens, phi_wp_acc, '-', color=BLUE, linewidth=2.5, label='Phi-4-mini (3.8B)')
    ax1.plot(tokens, qwen_wp_acc, '-', color=ORANGE, linewidth=2.5, label='Qwen2.5-Math (1.5B)')
    ax1.plot(30, 18.7, 'D', color=GRAY, markersize=9, markeredgewidth=2, markerfacecolor='white', zorder=5)
    ax1.plot(512, 96, 'D', color=GRAY, markersize=9, markeredgewidth=2, markerfacecolor='white', zorder=5)
    ax1.plot(300, 92, 'o', color=BLUE, markersize=9, markeredgewidth=2, markerfacecolor='white', zorder=5)
    ax1.plot(300, 64, 's', color=ORANGE, markersize=9, markeredgewidth=2, markerfacecolor='white', zorder=5)

    ax1.annotate('Both 18.7%', (30, 18.7), textcoords='offset points',
                 xytext=(-40, 22), fontsize=11, color=GRAY, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))
    ax1.annotate('Both 96%', (512, 96), textcoords='offset points',
                 xytext=(-40, -25), fontsize=11, color=GRAY, fontweight='bold')
    ax1.annotate('92%', (300, 92), textcoords='offset points',
                 xytext=(12, 8), fontsize=11, color=BLUE)
    ax1.annotate('64%', (300, 64), textcoords='offset points',
                 xytext=(12, -10), fontsize=11, color=ORANGE)

    ax1.set_xlabel('Token Budget')
    ax1.set_ylabel('WP Accuracy (%)')
    ax1.set_title('Word Problem Accuracy')
    ax1.set_xticks(tokens)
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper left', fontsize=10, frameon=True, edgecolor=GRAY)
    ax1.grid(alpha=0.3)

    ax2.plot(tokens, phi_energy, '-', color=BLUE, linewidth=2.5, label='Phi-4-mini (3.8B)')
    ax2.plot(tokens, qwen_energy, '-', color=ORANGE, linewidth=2.5, label='Qwen2.5-Math (1.5B)')
    ax2.plot(30, 30.6, 'D', color=GRAY, markersize=9, markeredgewidth=2, markerfacecolor='white', zorder=5)
    ax2.plot(300, 147.2, 'o', color=BLUE, markersize=9, markeredgewidth=2, markerfacecolor='white', zorder=5)
    ax2.plot(300, 113.1, 's', color=ORANGE, markersize=9, markeredgewidth=2, markerfacecolor='white', zorder=5)
    ax2.plot(512, 211.1, 'o', color=BLUE, markersize=9, markeredgewidth=2, markerfacecolor='white', zorder=5)
    ax2.plot(512, 147.6, 's', color=ORANGE, markersize=9, markeredgewidth=2, markerfacecolor='white', zorder=5)

    ax2.annotate('Both 30.6', (30, 30.6), textcoords='offset points',
                 xytext=(-40, 22), fontsize=11, color=GRAY, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))
    ax2.annotate('211.1', (512, 211.1), textcoords='offset points',
                 xytext=(-40, 10), fontsize=11, color=BLUE)
    ax2.annotate('147.6', (512, 147.6), textcoords='offset points',
                 xytext=(-40, -20), fontsize=11, color=ORANGE)
    ax2.annotate('147.2', (300, 147.2), textcoords='offset points',
                 xytext=(12, 8), fontsize=11, color=BLUE)
    ax2.annotate('113.1', (300, 113.1), textcoords='offset points',
                 xytext=(12, -14), fontsize=11, color=ORANGE)

    ax2.set_xlabel('Token Budget')
    ax2.set_ylabel('Energy (mWh / prompt)')
    ax2.set_title('Energy Cost')
    ax2.set_xticks(tokens)
    ax2.set_ylim(0, 260)
    ax2.legend(loc='upper left', fontsize=10, frameon=True, edgecolor=GRAY)
    ax2.grid(alpha=0.3)

    plt.suptitle('WP Token Budget Ablation: Accuracy vs Energy', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig4_wp_ablation.png'))
    plt.close(fig)
    print('  fig4 saved')


def fig5_route_usage():
    labels = ['A5 Arithmetic\n(symbolic)', 'A2 Word Problems\n(LLM)',
              'A6 Logic\n(symbolic)', 'A4 Algebra\n(SymPy)']
    sizes = [25, 25, 25, 25]
    colors = [BLUE, ORANGE, GREEN, PURPLE]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                       autopct='%1.0f%%', startangle=90,
                                       textprops={'fontsize': 12},
                                       pctdistance=0.55)
    for at in autotexts:
        at.set_fontsize(14)
        at.set_fontweight('bold')
        at.set_color('white')

    ax.set_title('V5 Route Usage:\n75% Symbolic, 25% LLM', fontsize=14, fontweight='bold')

    ax.text(0.02, 0.02, 'Zero LLM tokens\nfor 75% of prompts',
            transform=ax.transAxes, fontsize=10, ha='left', va='bottom',
            style='italic', color=GREEN,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GREEN, alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig5_route_usage.png'))
    plt.close(fig)
    print('  fig5 saved')


def fig6_energy_configs():
    configs = ['Baseline\n(C2)', 'V5\n30 tok', 'V5\n300 tok', 'V5\n512 tok']
    energy = [75.0, 30.6, 147.2, 211.1]
    colors = [LIGHT_RED, '#BBF7D0', LIGHT_BLUE, LIGHT_BLUE]
    edge_colors = [RED, GREEN, BLUE, BLUE]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(configs, energy, color=colors, edgecolor=edge_colors, linewidth=2, width=0.55)

    for bar, val in zip(bars, energy):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                f'{val:.1f} mWh', ha='center', fontweight='bold', fontsize=12)

    ax.axhline(y=75.0, color=RED, linestyle='--', linewidth=1.5, alpha=0.6)
    ax.text(3.5, 77, 'Baseline: 75.0 mWh', fontsize=10, color=RED, ha='right', va='bottom')

    ax.text(0.5, 0.95, '75% of prompts (AR/ALG/LOG) use zero LLM tokens',
            transform=ax.transAxes, fontsize=10, ha='center', va='top',
            style='italic', color=GRAY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GRAY, alpha=0.8))

    ax.set_ylabel('Energy (mWh / prompt)')
    ax.set_title('Energy per Prompt: Baseline vs V5 Configurations')
    ax.set_ylim(0, 270)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig6_energy_configs.png'))
    plt.close(fig)
    print('  fig6 saved')


def fig7_progression_by_category():
    versions = ['Base', 'V1', 'V2', 'V3', 'V4', 'V5', 'V5\n512tok']
    ar  = [48.0, 100, 100, 100, 100, 100, 100]
    alg = [40.0,  40,  96, 100, 100, 100, 100]
    log = [53.3,  64,  64,  64,  64, 100, 100]
    wp  = [ 8.0,   8,   8, 18.7, 18.7, 18.7, 96]

    x = np.arange(len(versions))
    width = 0.19

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_ar  = ax.bar(x - 1.5*width, ar,  width, label='AR',  color='#BFDBFE', edgecolor=BLUE, linewidth=1.5)
    bars_alg = ax.bar(x - 0.5*width, alg, width, label='ALG', color='#BBF7D0', edgecolor=GREEN, linewidth=1.5)
    bars_log = ax.bar(x + 0.5*width, log, width, label='LOG', color='#FED7AA', edgecolor=ORANGE, linewidth=1.5)
    bars_wp  = ax.bar(x + 1.5*width, wp,  width, label='WP',  color='#E9D5FF', edgecolor=PURPLE, linewidth=1.5)

    for bars, color, vals in [(bars_ar, BLUE, ar), (bars_alg, GREEN, alg),
                               (bars_log, ORANGE, log), (bars_wp, PURPLE, wp)]:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                        f'{val:.0f}' if val == int(val) else f'{val:.1f}',
                        ha='center', fontsize=8, color=color, fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Category Accuracy Across System Versions')
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylim(0, 120)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4,
              frameon=True, edgecolor=GRAY)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(os.path.join(OUT_DIR, 'fig7_progression_by_category.png'))
    plt.close(fig)
    print('  fig7 saved')


def fig8_version_categories():
    """Heatmap of accuracy by version x category."""
    versions = [
        'Baseline', 'V1 (+A5)', 'V2 (+A4)', 'V3.1 (+A3)', 'V4 (+Cal)',
        'V5 Default', 'V5 (512tok)',
        'Tool-calling agent', 'RPNI routing', 'L* routing',
    ]
    categories = ['AR', 'ALG', 'LOG', 'WP']
    data = np.array([
        [48.0, 40.0, 53.3,  8.0],
        [100,  40.0, 64.0,  8.0],
        [100,  96.0, 64.0,  8.0],
        [100, 100,   64.0, 18.7],
        [100, 100,   64.0, 18.7],
        [100, 100,  100,   18.7],
        [100, 100,  100,   96.0],
        [95,   76,   59,    5.0],
        [96,   96,   92,   13.0],
        [100, 100,  100,   12.0],
    ])

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # Draw each cell as a colored rectangle with explicit borders
    from matplotlib.patches import Rectangle
    for i in range(len(versions)):
        for j in range(len(categories)):
            val = data[i, j]
            # Get color from colormap
            norm_val = val / 100.0
            cell_color = plt.cm.RdYlGn(norm_val)
            rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                              facecolor=cell_color, edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            text_color = 'white' if val < 40 else 'black'
            ax.text(j, i, f'{val:.0f}' if val == int(val) else f'{val:.1f}',
                    ha='center', va='center', fontsize=12, fontweight='bold', color=text_color)



    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_ylim(len(versions) - 0.5, -0.5)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_yticks(range(len(versions)))
    ax.set_yticklabels(versions)
    ax.set_title('Accuracy (%) by System Version and Category')

    # Add colorbar using a ScalarMappable
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy (%)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig8_version_categories.png'))
    plt.close(fig)
    print('  fig8 saved')


if __name__ == '__main__':
    print('Generating paper figures (no internal captions)...')
    fig1_system_progression()
    fig2_category_accuracy()
    fig3_latency()
    fig4_wp_ablation()
    fig5_route_usage()
    fig6_energy_configs()
    fig7_progression_by_category()
    fig8_version_categories()
    print(f'\nAll figures saved to {OUT_DIR}/')
