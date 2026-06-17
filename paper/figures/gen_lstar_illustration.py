#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.2))
fig.patch.set_facecolor('white')

# ── LEFT: Observation Table ─────────────────────────────────────────────────
ax_l.set_xlim(0, 1)
ax_l.set_ylim(0, 1)
ax_l.axis('off')
ax_l.set_title('Observation Table $(S, E, T)$', fontsize=13, fontweight='bold', pad=10)

col_labels = ['$S \\setminus E$', '$\\varepsilon$', 'CMD', 'CMD·NUM']
col_x = [0.18, 0.42, 0.65, 0.88]
row_labels = ['$\\varepsilon$', 'CMD', 'CMD·NUM', 'CMD·NUM·OP', 'CMD·NUM·OP·NUM']
row_y = [0.82, 0.67, 0.52, 0.33, 0.18]

# T values: rows = ε, CMD, CMD·NUM, CMD·NUM·OP, CMD·NUM·OP·NUM
#           cols = ε, CMD, CMD·NUM
T = [
    [False, False, False],
    [False, False, False],
    [False, True,  True ],
    [False, False, True ],
    [False, True,  True ],
]

# Header row
for j, lbl in enumerate(col_labels):
    weight = 'bold' if j == 0 else 'normal'
    ax_l.text(col_x[j], 0.94, lbl, ha='center', va='center',
              fontsize=11, fontweight=weight)

ax_l.axhline(0.89, xmin=0.0, xmax=1.0, color='black', linewidth=1.5)

# Separator between S and S·Σ
ax_l.axhline(0.59, xmin=0.0, xmax=1.0, color='gray', linewidth=1.0, linestyle='--')
ax_l.text(0.01, 0.56, '$S{\\cdot}\\Sigma$', fontsize=8.5, color='gray', style='italic')

for i, (rl, ry) in enumerate(zip(row_labels, row_y)):
    ax_l.text(col_x[0], ry, rl, ha='center', va='center', fontsize=11)
    for j, val in enumerate(T[i]):
        color = '#27AE60' if val else '#C0392B'
        sym = '✓' if val else '✗'
        ax_l.text(col_x[j+1], ry, sym, ha='center', va='center',
                  fontsize=13, color=color, fontweight='bold')

# Vertical divider after first column
ax_l.axvline(0.29, ymin=0.08, ymax=0.97, color='black', linewidth=1.0)

# MQ annotation on CMD·NUM row
ax_l.annotate('MQ\n(SLM)', xy=(0.95, 0.52), xytext=(0.95, 0.52),
              fontsize=8, color='#2980B9', ha='left', va='center',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='#EBF5FB',
                        edgecolor='#2980B9', linewidth=0.8))
ax_l.annotate('', xy=(0.88, 0.52), xytext=(0.93, 0.52),
              arrowprops=dict(arrowstyle='<-', color='#2980B9', lw=1.2))

# ── RIGHT: Learned DFA ──────────────────────────────────────────────────────
ax_r.set_xlim(-0.3, 4.3)
ax_r.set_ylim(-0.8, 1.8)
ax_r.axis('off')
ax_r.set_title('Learned DFA  $\\mathcal{A}_{AR}$', fontsize=13, fontweight='bold', pad=10)

states = {'q0': (0.3, 0.5), 'q1': (1.9, 0.5), 'q2': (3.5, 0.5), 'qd': (1.9, -0.5)}
r = 0.38

def draw_state(ax, name, pos, accept=False, label=''):
    x, y = pos
    face = '#D5F5E3' if accept else '#EBF5FB'
    edge = '#27AE60' if accept else '#2980B9'
    lw = 2.5 if accept else 1.8
    c = plt.Circle((x, y), r, facecolor=face, edgecolor=edge, linewidth=lw, zorder=3)
    ax.add_patch(c)
    if accept:
        c2 = plt.Circle((x, y), r*0.75, facecolor='none', edgecolor='#27AE60',
                         linewidth=1.2, zorder=4)
        ax.add_patch(c2)
    ax.text(x, y, label, ha='center', va='center', fontsize=10,
            fontweight='bold', color='#1A252F', zorder=5)

draw_state(ax_r, 'q0', states['q0'], label='$q_0$')
draw_state(ax_r, 'q1', states['q1'], label='$q_1$')
draw_state(ax_r, 'q2', states['q2'], accept=True, label='$q_2$')
draw_state(ax_r, 'qd', states['qd'], label='$q_\\perp$')

# State labels below
for name, (x, y) in states.items():
    sublabel = {'q0': '$[\\varepsilon]$', 'q1': '[CMD]',
                'q2': '[CMD·NUM]', 'qd': '(dead)'}[name]
    ax_r.text(x, y - r - 0.12, sublabel, ha='center', va='top',
              fontsize=8.5, color='#555')

# Start arrow
ax_r.annotate('', xy=(states['q0'][0]-r, states['q0'][1]),
              xytext=(states['q0'][0]-r-0.5, states['q0'][1]),
              arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax_r.text(states['q0'][0]-r-0.55, states['q0'][1]+0.1, 'start', fontsize=9)

def arrow(ax, s1, s2, lbl, dy=0.18, rad=0.0, color='#2980B9'):
    x0, y0 = states[s1]; x1, y1 = states[s2]
    dx, dy_ = x1-x0, y1-y0
    dist = np.hypot(dx, dy_)
    sx, sy = x0+dx/dist*r, y0+dy_/dist*r
    ex, ey = x1-dx/dist*r, y1-dy_/dist*r
    ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                connectionstyle=f'arc3,rad={rad}'))
    mx, my = (sx+ex)/2, (sy+ey)/2
    ax.text(mx, my+dy, lbl, ha='center', va='center', fontsize=10, color=color,
            bbox=dict(facecolor='white', edgecolor='none', pad=1))

arrow(ax_r, 'q0', 'q1', 'CMD', dy=0.2)
arrow(ax_r, 'q1', 'q2', 'NUM', dy=0.2)
arrow(ax_r, 'q0', 'qd', '¬CMD', dy=0, rad=-0.3, color='#E74C3C')
arrow(ax_r, 'q1', 'qd', '¬NUM', dy=-0.22, rad=0.0, color='#E74C3C')
arrow(ax_r, 'q2', 'qd', 'other', dy=0, rad=-0.35, color='#E74C3C')

# Self-loop on qd
cx, cy = states['qd']
theta = np.linspace(0.3, 2*np.pi-0.3, 80)
lx = cx + 0.45*np.cos(theta)
ly = cy - 0.38 + 0.32*np.sin(theta)
ax_r.plot(lx, ly, color='#E74C3C', lw=1.3)
ax_r.annotate('', xy=(lx[-1], ly[-1]), xytext=(lx[-2], ly[-2]),
              arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.3))
ax_r.text(cx+0.55, cy-0.38, 'Σ', fontsize=10, color='#E74C3C')

# EQ box at top
ax_r.text(2.0, 1.55,
          'Equivalence query → counterexample → expand $S$ → repeat',
          ha='center', va='top', fontsize=8.5, color='#555',
          style='italic',
          bbox=dict(facecolor='#FDFEFE', edgecolor='#BDC3C7',
                    boxstyle='round,pad=0.4', linewidth=0.8))

plt.tight_layout(pad=1.5)
plt.savefig('/Users/avyaysadhu/Documents/pi-neurosymbolic-routing/paper/figures/fig_lstar_illustration.png',
            dpi=220, bbox_inches='tight', facecolor='white')
print("Done")
