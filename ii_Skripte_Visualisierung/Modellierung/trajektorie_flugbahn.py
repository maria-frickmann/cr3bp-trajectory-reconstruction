import sys
from pathlib import Path
my_path  = Path(__file__).resolve().parents[2]
my_file  = 'trajektorie_optimal.pdf'
sys.path.append(str(my_path))
ziel_ordner = my_path / 'iii_Daten_Plots' / 'Modellierung'
speicher_pfad = ziel_ordner / my_file
ziel_ordner.mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from i_Programmierung.Konstanten import mu
from ii_Skripte_Visualisierung import berechnungen as ber
from i_Programmierung.Modellierung.lagrange import L2
from i_Programmierung.Modellierung.orbit import sol
from i_Programmierung.Modellierung.mannigfaltigkeiten import q1_opt, q2_opt, q3_opt
from ii_Skripte_Visualisierung.farben import *
from ii_Skripte_Visualisierung.hintergrund import draw_hintergrund_3d, Axenbeschriftung
from ii_Skripte_Visualisierung.setup_plot import (get_manifold_window, save_diagram,
                                                  HALO_ORBIT_LABEL, LABEL_TRANSFERBAHN,
                                                  LAGRANGE_LABELS)
from ii_Skripte_Visualisierung.farben import ERDE

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize

daten            = ber.trajektorien_rekonstruktion()
meta             = daten['meta']
rekonstruktionen = daten['rekonstruktionen']
X_ref            = meta['X_ref']
q1_ref, q2_ref, q3_ref = X_ref[:, 0], X_ref[:, 1], X_ref[:, 2]

pos_sonne = np.array([-mu,    0.0, 0.0])
pos_erde  = np.array([1 - mu, 0.0, 0.0])

xlim, ylim, zlim = get_manifold_window(L2[0], mu, abst=0.007)

fig = plt.figure(figsize=(9, 8), dpi=300)
ax  = fig.add_subplot(projection='3d', computed_zorder=False)

draw_hintergrund_3d(ax, xlim=xlim, ylim=ylim, zlim=zlim)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
ax.view_init(elev=12, azim=-110)

# zuerst Halo-Orbit, danach Mannigfaltigkeit
ax.plot(sol.y[0], sol.y[1], sol.y[2],
        color=ORBIT, lw=2.2, zorder=5, label=HALO_ORBIT_LABEL)
ax.plot(q1_opt, q2_opt, q3_opt,
        color=tj_optimal_farbe, lw=1.5, alpha=0.8, zorder=10,
        label=LABEL_TRANSFERBAHN)

# Erde
dx = 0.03 * (xlim[1] - xlim[0])
ax.scatter(*pos_erde, color=ERDE, s=60, zorder=12, depthshade=False)

# Beschriftung des blauen Punktes (Erde)
ax.text(*pos_erde + 0.03 * (zlim[1] - zlim[0]), 'Erde',
        fontsize=fontsize, color=SCHWARZ, ha='center', va='bottom', zorder=7)

Axenbeschriftung(ax)
ax.xaxis.labelpad = 12
ax.yaxis.labelpad = 12
ax.zaxis.labelpad = 10
ax.tick_params(axis='z', pad=6)
ax.zaxis.set_rotate_label(False)
ax.zaxis.label.set_rotation(90)

titel = ax.set_title(f'Halo-Orbit um {LAGRANGE_LABELS[1]}' + '\n', fontsize=fontsize, y=0.83)
ax.legend(frameon=False, fontsize=fontsize,
          loc='upper center', bbox_to_anchor=(0.5, 0.10), ncol=2)

ax.set_box_aspect((1, 1, 0.75), zoom=0.92)
fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.16)

save_diagram(fig, speicher_pfad, oben_bis=titel)