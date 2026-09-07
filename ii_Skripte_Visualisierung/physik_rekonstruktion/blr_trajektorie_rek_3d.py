import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from ii_Skripte_Visualisierung import berechnungen as ber
from ii_Skripte_Visualisierung.setup_plot import (
    farben_rauschen, labels_rauschen, save_diagram, LABEL_REFERENZ, fontsize,
    SYM_ALPHA_MIN_BLR, LABEL_ALPHA_MIN_BLR)
from ii_Skripte_Visualisierung.hintergrund import draw_hintergrund_3d, Axenbeschriftung
from ii_Skripte_Visualisierung.farben import *
from ii_Skripte_Visualisierung.physik_rekonstruktion._gemeinsam import (vorhandene_rekonstruktionen, achsen_limits)
from i_Programmierung.Konstanten import mu
from i_Programmierung.Modellierung.mannigfaltigkeiten import q1_opt, q2_opt, q3_opt


daten            = ber.blr_trajektorien_rekonstruktion()
meta             = daten['meta']
rekonstruktionen = daten['rekonstruktionen']
X_ref            = meta['X_ref']
q1_ref, q2_ref, q3_ref = X_ref[:, 0], X_ref[:, 1], X_ref[:, 2]

vorhanden = vorhandene_rekonstruktionen(meta, rekonstruktionen, 'alpha_min_blr',
                                        farben_rauschen, labels_rauschen)

alle_q1 = np.concatenate([q1_ref] + [rekonstruktionen[r]['q1_rek'] for r, _, _ in vorhanden])
xlim = achsen_limits(alle_q1)
ylim = achsen_limits(q2_ref)
zlim = achsen_limits(q3_ref)

fig = plt.figure(figsize=(9, 8), dpi=300)
ax  = fig.add_subplot(projection='3d')
draw_hintergrund_3d(ax, xlim=xlim, ylim=ylim, zlim=zlim)
ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
ax.view_init(elev=12, azim=-125)

ax.plot(q1_ref, q2_ref, q3_ref, color=SCHWARZ, lw=1.8,
        label=LABEL_REFERENZ, zorder=6)

for rauschen, farbe, label_r in vorhanden:
    rek = rekonstruktionen[rauschen]
    ax.plot(rek['q1_rek'], q2_ref, q3_ref, color=farbe, lw=1.3, ls='--',
            label=rf'{label_r} (${SYM_ALPHA_MIN_BLR} = {rek["alpha_min_blr"]:.3f}$)', zorder=5)

# Erde
dx = 0.03 * (xlim[1] - xlim[0])
ax.scatter(-mu, 0, 0, color=ERDE, s=60, zorder=12, depthshade=False)
ax.text(-mu - dx, 0, 0, 'Erde', fontsize=fontsize, color=ERDE,
        ha='right', va='center', zorder=13)

Axenbeschriftung(ax)
ax.xaxis.labelpad = 12
ax.yaxis.labelpad = 12
ax.zaxis.labelpad = 10
ax.tick_params(axis='z', pad=6)
ax.zaxis.set_rotate_label(False)
ax.zaxis.label.set_rotation(90)

# Erde
dx = 0.03 * (xlim[1] - xlim[0])
ax.scatter(-mu, 0, 0, color=ERDE, s=60, zorder=12, depthshade=False)
ax.text(q1_opt[0] - dx, q2_opt[0], q3_opt[0], 'Erde', fontsize=fontsize, color=ERDE, ha='right', va='center', zorder=13)


titel = ax.set_title(f'Trajektorienrekonstruktion bei {LABEL_ALPHA_MIN_BLR}' + '\n',
                     fontsize=fontsize, y=0.83)
ax.legend(frameon=False, fontsize=fontsize,
          loc='upper center', bbox_to_anchor=(0.5, 0.10), ncol=2)

fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.08)
ax.set_box_aspect((1, 1, 0.75), zoom=0.92)
save_diagram(fig, 'blr_trajektorienrekonstruktion_3d.pdf',
             unterordner='physik_rekonstruktion', oben_bis=titel)
plt.show()