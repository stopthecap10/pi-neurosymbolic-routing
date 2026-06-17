#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor('white')
ax.set_xlim(-0.8, 5.8)
ax.set_ylim(-1.4, 2.0)
ax.axis('off')
ax.set_title('Learned DFA $\\mathcal{A}_{AR}$ from $L^\\star$',
             fontsize=15, fontweight='bold', pad=14)

states = {
    'q0': (0.8, 0.5),
    'q1': (2.5, 0.5),
    'q2': (4.2, 0.5),
    'qd': (2.5, -0.9),
}
r = 0.42

def draw_state(ax, pos, label, sublabel, accept=False):
    x, y = pos
    face = '#D5F5E3' if accept else '#EBF5FB'
    edge = '#27AE60' if accept else '#2980B9'
    lw = 2.8 if accept else 2.0
    c = plt.Circle((x, y), r, facecolor=face, edgecolor=edge,
                   linewidth=lw, zorder=3)
    ax.add_patch(c)
    if accept:
        c2 = plt.Circle((x, y), r * 0.74, facecolor='none',
                         edgecolor='#27AE60', linewidth=1.4, zorder=4)
        ax.add_patch(c2)
    ax.text(x, y + 0.05, label, ha='center', va='center',
            fontsize=13, fontweight='bold', color='#1A252F', zorder=5)
    ax.text(x, y - 0.16, sublabel, ha='center', va='center',
            fontsize=8.5, color='#555555', zorder=5)

draw_state(ax, states['q0'], '$q_0$', '$[\\varepsilon]$')
draw_state(ax, states['q1'], '$q_1$', '[CMD]')
draw_state(ax, states['q2'], '$q_2$', '[CMD·NUM]', accept=True)
draw_state(ax, states['qd'], '$q_\\perp$', '(dead/reject)', accept=False)

# Start arrow
x0, y0 = states['q0']
ax.annotate('', xy=(x0 - r, y0),
            xytext=(x0 - r - 0.55, y0),
            arrowprops=dict(arrowstyle='->', color='black', lw=2.0))
ax.text(x0 - r - 0.6, y0 + 0.15, 'start', fontsize=10, ha='right')

def edge(ax, s1, s2, lbl, rad=0.0, lbl_dy=0.22, lbl_dx=0.0, color='#2980B9'):
    x0, y0 = states[s1]
    x1, y1 = states[s2]
    dx, dy = x1 - x0, y1 - y0
    dist = np.hypot(dx, dy)
    sx, sy = x0 + dx/dist*r, y0 + dy/dist*r
    ex, ey = x1 - dx/dist*r, y1 - dy/dist*r
    style = f'arc3,rad={rad}'
    ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0,
                                connectionstyle=style))
    mx = (sx + ex) / 2 + lbl_dx
    my = (sy + ey) / 2 + lbl_dy
    ax.text(mx, my, lbl, ha='center', va='center', fontsize=11,
            color=color,
            bbox=dict(facecolor='white', edgecolor='none', pad=2))

# Main transitions (blue)
edge(ax, 'q0', 'q1', 'CMD',  lbl_dy=0.26)
edge(ax, 'q1', 'q2', 'NUM',  lbl_dy=0.26)

# Dead transitions (red)
edge(ax, 'q0', 'qd', '¬CMD', rad=-0.28, lbl_dy=0.0, lbl_dx=-0.35, color='#E74C3C')
edge(ax, 'q1', 'qd', '¬NUM', rad=0.0,  lbl_dy=-0.28, color='#E74C3C')
edge(ax, 'q2', 'qd', 'other', rad=-0.32, lbl_dy=0.0, lbl_dx=0.45, color='#E74C3C')

# Self-loop on qd
cx, cy = states['qd']
theta = np.linspace(0.25, 2*np.pi - 0.25, 100)
lx = cx + 0.52 * np.cos(theta)
ly = cy - 0.36 + 0.34 * np.sin(theta)
ax.plot(lx, ly, color='#E74C3C', lw=1.8)
ax.annotate('', xy=(lx[-1], ly[-1]), xytext=(lx[-2], ly[-2]),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.8))
ax.text(cx + 0.78, cy - 0.36, 'Σ', fontsize=12, color='#E74C3C', va='center')

# Equivalence query note at top
ax.text(2.5, 1.75,
        'If hypothesis wrong → equivalence query returns counterexample\n'
        '→ add to $S$, re-fill table, rebuild DFA',
        ha='center', va='top', fontsize=9, color='#555',
        style='italic',
        bbox=dict(facecolor='#FDFEFE', edgecolor='#BDC3C7',
                  boxstyle='round,pad=0.5', linewidth=1.0))

plt.tight_layout()
plt.savefig(
    '/Users/avyaysadhu/Documents/pi-neurosymbolic-routing/paper/figures/fig_lstar_dfa.png',
    dpi=220, bbox_inches='tight', facecolor='white')
print("Done")
