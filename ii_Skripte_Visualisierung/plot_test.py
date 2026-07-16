import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt

from ii_Skripte_Visualisierung.farben import SONNE, ERDE, tj_optimal_farbe
from ii_Skripte_Visualisierung.setup_plot import (farben_rauschen, labels_rauschen,
                                                  farben_balken, style_ax,
                                                  get_manifold_window, save_diagram)
from ii_Skripte_Visualisierung.hintergrund import (draw_hintergrund_2d,
                                                   draw_hintergrund_3d,
                                                   Axenbeschriftung)
from i_Programmierung.Modellierung.lagrange import l_punkte
from i_Programmierung.Konstanten import mu

from setup_plot import fontsize
fontsize = fontsize

# 2d Plot
x = np.linspace(0, 2 * np.pi, 200)
fig, ax = plt.subplots(figsize=(5, 3.5))
style_ax(ax)
for i, (farbe, label) in enumerate(zip(farben_rauschen, labels_rauschen)):
    ax.plot(x, np.sin(x + i), color=farbe, label=f'Rauschen {label}')
ax.plot(x, np.cos(x), color=farben_balken[0], linestyle='--', label='CS-Farbe')
ax.plot(x, -np.cos(x), color=farben_balken[1], linestyle='--', label='Bayes-Farbe')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('plot_test: 2D-Stil (style_ax + Paletten)')
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
save_diagram(fig, 'plot_test_stil_2d.png')
plt.close(fig)

# 2d Plot
t = np.linspace(0, 4 * np.pi, 300)
fig = plt.figure(figsize=(5, 4.5))
ax = fig.add_subplot(projection='3d')
for i, farbe in enumerate(farben_rauschen):
    ax.plot(np.cos(t + i), np.sin(t + i), t / (4 * np.pi), color=farbe)
ax.scatter([0], [0], [0], color=SONNE, s=40)
ax.scatter([1], [0], [0], color=ERDE, s=20)
ax.set_title('plot_test: 3D-Achsen')
fig.tight_layout()
save_diagram(fig, 'plot_test_stil_3d.png')
plt.close(fig)

xlim, ylim, zlim = get_manifold_window(l_punkte[1][0], mu)   # Fenster L2 <-> Erde

fig = plt.figure(figsize=(11, 5))
ax2d = fig.add_subplot(1, 2, 1)
draw_hintergrund_2d(ax2d, xlim=xlim, ylim=ylim)
Axenbeschriftung(ax2d)
ax2d.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], color=tj_optimal_farbe, lw=1.5)
ax2d.set_title('Hintergrund 2D')

ax3d = fig.add_subplot(1, 2, 2, projection='3d')
draw_hintergrund_3d(ax3d, xlim=xlim, ylim=ylim, zlim=zlim)
ax3d.set_xlim(xlim); ax3d.set_ylim(ylim); ax3d.set_zlim(zlim)
Axenbeschriftung(ax3d)
ax3d.set_title('Hintergrund 3D')

fig.tight_layout()
save_diagram(fig, 'plot_test_hintergrund.png')
plt.close(fig)
