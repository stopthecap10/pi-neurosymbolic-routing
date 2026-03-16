#!/usr/bin/env python3
"""
Generate all poster charts for ISEF presentation.
Outputs high-res PNGs to outputs/official/poster/charts/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np
import os

OUT_DIR = 'outputs/official/poster/charts'
os.makedirs(OUT_DIR, exist_ok=True)

# Force Arial everywhere — 24pt minimum, larger for titles
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'font.size': 24,
    'axes.titlesize': 32,
    'axes.labelsize': 26,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Caption style
CAPTION_FONTSIZE = 30

def save_with_caption(fig, filename, caption):
    """Save figure with a caption below it."""
    fig.text(0.5, -0.04, caption, ha='center', va='top',
             fontsize=CAPTION_FONTSIZE, fontstyle='italic',
             family='Arial', transform=fig.transFigure)
    plt.savefig(os.path.join(OUT_DIR, filename),
                bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)

# Color palette
BLUE = '#2563EB'
RED = '#DC2626'
GREEN = '#16A34A'
ORANGE = '#EA580C'
PURPLE = '#9333EA'
GRAY = '#6B7280'
LIGHT_BLUE = '#93C5FD'
LIGHT_RED = '#FCA5A5'


# =====================================================================
# R5 — System Progression (Baseline → V5)
# =====================================================================
def chart_r5_progression():
    versions = ['Baseline\n(C2)', 'V1\n+A5', 'V2\n+A4', 'V3\n+A3', 'V4\n+Cal', 'V5\n+A6', 'V5\n+CoT512']
    accuracy = [37.3, 53.0, 67.0, 70.7, 70.7, 79.7, 99.0]

    fig, ax = plt.subplots(figsize=(18, 10))

    ax.plot(range(len(versions)), accuracy, 'o-', color=BLUE, linewidth=3,
            markersize=14, markerfacecolor='white', markeredgewidth=3, zorder=3)
    ax.fill_between(range(len(versions)), accuracy, alpha=0.08, color=BLUE)

    # Annotations — stagger above/below with long arrows, offset horizontally to avoid crowding
    ann_config = {
        1: ('+A5 Arithmetic\nParser',   (-40, -90)),   # below-left
        2: ('+A4 SymPy\nAlgebra',       (40, 70)),     # above-right
        5: ('+A6 Logic\nEngine',        (-40, -90)),   # below-left
        6: ('+512 token\nCoT for WP',   (40, 70)),     # above-right
    }
    for idx, (text, xyoff) in ann_config.items():
        ax.annotate(text, (idx, accuracy[idx]),
                    textcoords='offset points', xytext=xyoff,
                    ha='center', fontsize=24, color=BLUE,
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=2))

    # Value labels for points without annotations
    for i, v in enumerate(accuracy):
        if i in ann_config:
            continue
        ax.annotate(f'{v:.1f}%', (i, v), textcoords='offset points',
                    xytext=(0, 18), ha='center',
                    fontsize=24, fontweight='bold')

    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions)
    ax.set_ylabel('Overall Accuracy (%)')
    ax.set_title('System Progression: Baseline to V5')
    ax.set_ylim(5, 120)
    ax.axhline(y=37.3, color=GRAY, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(pad=2.0)
    caption = (r"$\bf{Figure\ 7.}$ System progression from baseline through V5, showing accuracy gains"
               "\nfrom added components. Created by author, 2026.")
    save_with_caption(fig, 'R5_system_progression.png', caption)
    print('  R5 saved')


# =====================================================================
# R1 — Main Results Summary (Baseline vs V5)
# =====================================================================
def chart_r1_headline():
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    labels = ['Baseline\n(C2)', 'V5 Hybrid\n(512 tok)']
    colors = [LIGHT_RED, LIGHT_BLUE]
    edge_colors = [RED, BLUE]

    # Accuracy
    vals = [37.3, 99.0]
    bars = axes[0].bar(labels, vals, color=colors, edgecolor=edge_colors, linewidth=2)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy')
    axes[0].set_ylim(0, 125)
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 3,
                     f'{val:.1f}%', ha='center', fontweight='bold', fontsize=26)

    # Median Latency
    vals = [8612, 6]
    bars = axes[1].bar(labels, vals, color=colors, edgecolor=edge_colors, linewidth=2)
    axes[1].set_ylabel('Median Latency (ms)')
    axes[1].set_title('Latency')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1, 30000)
    for bar, val in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, val * 2.0,
                     f'{val:,} ms', ha='center', fontweight='bold', fontsize=26)

    # Energy — clean layout, no annotation overlay
    vals = [75.0, 211.1]
    bars = axes[2].bar(labels, vals, color=colors, edgecolor=edge_colors, linewidth=2)
    axes[2].set_ylabel('Energy (mWh/prompt)')
    axes[2].set_title('Energy (WP=512 mode)')
    axes[2].set_ylim(0, 280)
    for bar, val in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 8,
                     f'{val:.1f}', ha='center', fontweight='bold', fontsize=26)

    plt.suptitle('Baseline vs V5 Hybrid: Overall Performance',
                 fontsize=32, fontweight='bold', y=1.03)
    plt.tight_layout(pad=2.0)
    caption = (r"$\bf{Figure\ 4.}$ Headline comparison: baseline vs V5 hybrid overall accuracy, median latency,"
               "\nand energy per prompt (WP=512 mode). Note: V5 uses more energy for 96% WP accuracy."
               "\nCreated by author, 2026.")
    save_with_caption(fig, 'R1_headline_summary.png', caption)
    print('  R1 saved')


# =====================================================================
# R2 — Accuracy by Category (grouped bars)
# =====================================================================
def chart_r2_category_accuracy():
    categories = ['AR', 'ALG', 'LOG', 'WP']
    baseline = [48.0, 40.0, 53.3, 8.0]
    v5_512 = [100.0, 100.0, 100.0, 96.0]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline (C2 Grammar)',
                   color=LIGHT_RED, edgecolor=RED, linewidth=1.5)
    bars2 = ax.bar(x + width/2, v5_512, width, label='V5 Hybrid (512 tok)',
                   color=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.5)

    # Baseline value labels — inside bars for cleaner look
    for bar, val in zip(bars1, baseline):
        ax.text(bar.get_x() + bar.get_width()/2, val / 2,
                f'{val:.0f}%', ha='center', fontsize=24, color=RED,
                fontweight='bold', va='center')

    # V5 value labels — inside bars
    for bar, val in zip(bars2, v5_512):
        ax.text(bar.get_x() + bar.get_width()/2, val / 2,
                f'{val:.0f}%', ha='center', fontsize=24, color=BLUE,
                fontweight='bold', va='center')

    # Delta annotations — above bars with clear spacing
    for i in range(len(categories)):
        delta = v5_512[i] - baseline[i]
        mid_x = x[i]
        ax.annotate(f'+{delta:.0f}%', (mid_x, max(v5_512[i], baseline[i]) + 5),
                    ha='center', fontsize=26, fontweight='bold', color=GREEN)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Category Accuracy: Baseline vs V5')
    ax.set_xticks(x)
    ax.set_xticklabels(['Arithmetic\n(AR)', 'Algebra\n(ALG)', 'Logic\n(LOG)', 'Word Problems\n(WP)'])
    ax.set_ylim(0, 135)
    # Legend below chart to avoid any data overlap
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=True, edgecolor=GRAY)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.22)
    caption = (r"$\bf{Figure\ 6.}$ Per-category accuracy comparison: baseline vs V5"
               "\n(512 tokens for WP). Created by author, 2026.")
    save_with_caption(fig, 'R2_category_accuracy.png', caption)
    print('  R2 saved')


# =====================================================================
# R3 — WP Token Budget Tradeoff
# =====================================================================
def chart_r3_wp_ablation():
    tokens = [30, 300, 512]

    phi_wp_acc = [18.7, 92.0, 96.0]
    qwen_wp_acc = [18.7, 64.0, 96.0]
    phi_energy = [30.6, 147.2, 211.1]
    qwen_energy = [30.6, 113.1, 147.6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot lines without markers, then add markers manually to avoid overlap at shared points
    ax1.plot(tokens, phi_wp_acc, '-', color=BLUE, linewidth=3, label='Phi-4-mini (3.8B)')
    ax1.plot(tokens, qwen_wp_acc, '-', color=ORANGE, linewidth=3, label='Qwen2.5-Math (1.5B)')
    # Shared points: single diamond marker
    ax1.plot(30, 18.7, 'D', color=GRAY, markersize=12, markeredgewidth=2, markerfacecolor='white', zorder=5)
    ax1.plot(512, 96, 'D', color=GRAY, markersize=12, markeredgewidth=2, markerfacecolor='white', zorder=5)
    # Separate points at 300
    ax1.plot(300, 92, 'o', color=BLUE, markersize=12, markeredgewidth=2.5, markerfacecolor='white', zorder=5)
    ax1.plot(300, 64, 's', color=ORANGE, markersize=12, markeredgewidth=2.5, markerfacecolor='white', zorder=5)

    ax1.annotate('Both 18.7%', (30, 18.7), textcoords='offset points',
                 xytext=(-60, 30), fontsize=24, color=GRAY, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))
    ax1.annotate('Both 96%', (512, 96), textcoords='offset points',
                 xytext=(-50, -35), fontsize=24, color=GRAY, fontweight='bold')
    ax1.annotate('92%', (300, 92), textcoords='offset points',
                 xytext=(18, 12), fontsize=24, color=BLUE)
    ax1.annotate('64%', (300, 64), textcoords='offset points',
                 xytext=(18, -15), fontsize=24, color=ORANGE)

    ax1.set_xlabel('Token Budget')
    ax1.set_ylabel('WP Accuracy (%)')
    ax1.set_title('Word Problem Accuracy')
    ax1.set_xticks(tokens)
    ax1.set_ylim(0, 120)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1,
               frameon=True, edgecolor=GRAY)
    ax1.grid(alpha=0.3)

    # Energy subplot — same approach
    ax2.plot(tokens, phi_energy, '-', color=BLUE, linewidth=3, label='Phi-4-mini (3.8B)')
    ax2.plot(tokens, qwen_energy, '-', color=ORANGE, linewidth=3, label='Qwen2.5-Math (1.5B)')
    # Shared point at 30
    ax2.plot(30, 30.6, 'D', color=GRAY, markersize=12, markeredgewidth=2, markerfacecolor='white', zorder=5)
    # Separate points
    ax2.plot(300, 147.2, 'o', color=BLUE, markersize=12, markeredgewidth=2.5, markerfacecolor='white', zorder=5)
    ax2.plot(300, 113.1, 's', color=ORANGE, markersize=12, markeredgewidth=2.5, markerfacecolor='white', zorder=5)
    ax2.plot(512, 211.1, 'o', color=BLUE, markersize=12, markeredgewidth=2.5, markerfacecolor='white', zorder=5)
    ax2.plot(512, 147.6, 's', color=ORANGE, markersize=12, markeredgewidth=2.5, markerfacecolor='white', zorder=5)

    ax2.annotate('Both 30.6', (30, 30.6), textcoords='offset points',
                 xytext=(-60, 30), fontsize=24, color=GRAY, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))
    ax2.annotate('147.2', (300, 147.2), textcoords='offset points',
                 xytext=(18, 12), fontsize=24, color=BLUE)
    ax2.annotate('113.1', (300, 113.1), textcoords='offset points',
                 xytext=(18, -18), fontsize=24, color=ORANGE)
    ax2.annotate('211.1', (512, 211.1), textcoords='offset points',
                 xytext=(-50, 15), fontsize=24, color=BLUE)
    ax2.annotate('147.6', (512, 147.6), textcoords='offset points',
                 xytext=(-50, -30), fontsize=24, color=ORANGE)

    ax2.set_xlabel('Token Budget')
    ax2.set_ylabel('Energy (mWh / prompt)')
    ax2.set_title('Energy Cost')
    ax2.set_xticks(tokens)
    ax2.set_ylim(0, 270)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1,
               frameon=True, edgecolor=GRAY)
    ax2.grid(alpha=0.3)

    plt.suptitle('WP Token Budget Ablation: Accuracy vs Energy Tradeoff',
                 fontsize=30, fontweight='bold', y=1.03)
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.22)
    caption = (r"$\bf{Figure\ 5.}$ Word-problem token budget ablation (30/300/512): accuracy vs energy"
               "\ntradeoff for two SLMs. Created by author, 2026.")
    save_with_caption(fig, 'R3_wp_token_ablation.png', caption)
    print('  R3 saved')


# =====================================================================
# R4 — Latency by Category (log scale)
# =====================================================================
def chart_r4_latency():
    categories = ['AR', 'ALG', 'LOG', 'WP']
    baseline_lat = [6288, 5838, 2672, 25600]
    v5_lat = [1, 11, 2, 137234]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 9))

    bars1 = ax.bar(x - width/2, baseline_lat, width, label='Baseline (C2 Grammar)',
                   color=LIGHT_RED, edgecolor=RED, linewidth=1.5)
    bars2 = ax.bar(x + width/2, v5_lat, width, label='V5 Hybrid (512 tok)',
                   color=LIGHT_BLUE, edgecolor=BLUE, linewidth=1.5)

    ax.set_yscale('log')
    ax.set_ylabel('Median Latency (ms)\nlog scale')
    ax.set_title('Per-Category Latency: Baseline vs V5')
    ax.set_xticks(x)
    ax.set_xticklabels(['Arithmetic\n(AR)', 'Algebra\n(ALG)', 'Logic\n(LOG)', 'Word Problems\n(WP)'])
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.3, 800000)

    # Value labels on top of bars
    for bar, val in zip(bars1, baseline_lat):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.8,
                f'{val:,}', ha='center', fontsize=24, color=RED)
    for bar, val in zip(bars2, v5_lat):
        ax.text(bar.get_x() + bar.get_width()/2, val * 3.0,
                f'{val:,}', ha='center', fontsize=24, color=BLUE)

    # Speedup annotations at the very bottom
    speedups = ['6,288x', '531x', '1,336x', '']
    for i, s in enumerate(speedups):
        if s:
            ax.annotate(f'{s}\nfaster', (x[i], 0.03),
                        ha='center', fontsize=24, fontweight='bold', color=GREEN,
                        xycoords=('data', 'axes fraction'))

    # Legend below chart to avoid covering any bars/labels
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=True, edgecolor=GRAY)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.20)
    caption = (r"$\bf{Figure\ 8.}$ Per-category median latency: baseline vs V5"
               "\n(log scale). Created by author, 2026.")
    save_with_caption(fig, 'R4_latency_by_category.png', caption)
    print('  R4 saved')


# =====================================================================
# R6 — Per-Category Accuracy Across Versions (V1–V5)
# =====================================================================
def chart_r6_version_categories():
    versions = ['Base', 'V1', 'V2', 'V3', 'V4', 'V5', 'V5\n+512']
    ar  = [48.0, 100, 100, 100, 100, 100, 100]
    alg = [40.0,  40,  96, 100, 100, 100, 100]
    log = [53.3,  64,  64,  64,  64, 100, 100]
    wp  = [ 8.0,   8,   8, 18.7, 18.7, 18.7, 96]

    x = np.arange(len(versions))
    width = 0.19

    fig, ax = plt.subplots(figsize=(20, 10))

    bars_ar  = ax.bar(x - 1.5*width, ar,  width, label='AR',  color='#BFDBFE', edgecolor=BLUE, linewidth=1.5)
    bars_alg = ax.bar(x - 0.5*width, alg, width, label='ALG', color='#BBF7D0', edgecolor=GREEN, linewidth=1.5)
    bars_log = ax.bar(x + 0.5*width, log, width, label='LOG', color='#FED7AA', edgecolor=ORANGE, linewidth=1.5)
    bars_wp  = ax.bar(x + 1.5*width, wp,  width, label='WP',  color='#E9D5FF', edgecolor=PURPLE, linewidth=1.5)

    # Value labels on top of each bar
    for bars, color in [(bars_ar, BLUE), (bars_alg, GREEN), (bars_log, ORANGE), (bars_wp, PURPLE)]:
        for bar, val in zip(bars, [ar, alg, log, wp][[bars_ar, bars_alg, bars_log, bars_wp].index(bars)]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                        f'{val:.0f}' if val == int(val) else f'{val:.1f}',
                        ha='center', fontsize=16, color=color, fontweight='bold')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Category Accuracy Across System Versions')
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylim(0, 125)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4,
              frameon=True, edgecolor=GRAY)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.18)
    caption = (r"$\bf{Figure\ 9.}$ Per-category accuracy from baseline through V5+512, showing"
               "\nwhich component improved each category. Created by author, 2026.")
    save_with_caption(fig, 'R6_version_categories.png', caption)
    print('  R6 saved')


# =====================================================================
# R7 — Energy by Configuration (Baseline vs V5 at different token budgets)
# =====================================================================
def chart_r7_energy_configs():
    configs = ['Baseline\n(C2)', 'V5\n30 tok', 'V5\n300 tok', 'V5\n512 tok']
    energy = [75.0, 30.6, 147.2, 211.1]
    colors = [LIGHT_RED, '#BBF7D0', LIGHT_BLUE, LIGHT_BLUE]
    edge_colors = [RED, GREEN, BLUE, BLUE]

    fig, ax = plt.subplots(figsize=(14, 9))

    bars = ax.bar(configs, energy, color=colors, edgecolor=edge_colors, linewidth=2, width=0.55)

    # Value labels
    for bar, val in zip(bars, energy):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5,
                f'{val:.1f} mWh', ha='center', fontweight='bold', fontsize=24)

    # Baseline reference line
    ax.axhline(y=75.0, color=RED, linestyle='--', linewidth=1.5, alpha=0.6)
    ax.text(3.6, 77, 'Baseline: 75.0 mWh', fontsize=20, color=RED, ha='right', va='bottom')

    # Note about 75% symbolic
    ax.text(0.5, 0.95, '75% of prompts (AR/ALG/LOG) use zero LLM tokens',
            transform=ax.transAxes, fontsize=20, ha='center', va='top',
            style='italic', color=GRAY,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GRAY, alpha=0.8))

    ax.set_ylabel('Energy (mWh / prompt)')
    ax.set_title('Energy per Prompt: Baseline vs V5 Configurations')
    ax.set_ylim(0, 280)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(pad=2.0)
    caption = (r"$\bf{Figure\ 10.}$ Energy per prompt across configurations. V5 at 30 tokens uses"
               "\n59% less energy than baseline while achieving 70.7% accuracy. Created by author, 2026.")
    save_with_caption(fig, 'R7_energy_configs.png', caption)
    print('  R7 saved')


# =====================================================================
# Run all
# =====================================================================
if __name__ == '__main__':
    print('Generating poster charts...')
    chart_r1_headline()       # Fig 4
    chart_r3_wp_ablation()    # Fig 5
    chart_r2_category_accuracy()  # Fig 6
    chart_r5_progression()    # Fig 7
    chart_r4_latency()        # Fig 8
    chart_r6_version_categories()  # Fig 9
    chart_r7_energy_configs()      # Fig 10
    print(f'\nAll charts saved to {OUT_DIR}/')
