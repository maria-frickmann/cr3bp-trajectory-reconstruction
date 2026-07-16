import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from ii_Skripte_Visualisierung import berechnungen as ber
from ii_Skripte_Visualisierung.setup_plot import (farben_rauschen, labels_rauschen,
                                                  save_diagram, LABEL_REFERENZ, fontsize)
from ii_Skripte_Visualisierung.hintergrund import draw_hintergrund_3d, Axenbeschriftung
from ii_Skripte_Visualisierung.farben import SCHWARZ
from ii_Skripte_Visualisierung.physik_rekonstruktion._gemeinsam import (
    vorhandene_rekonstruktionen, achsen_limits)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

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

ax.plot(q1_ref, q2_ref, q3_ref, color=SCHWARZ, lw=1.8,
        label=LABEL_REFERENZ, zorder=6)

for rauschen, farbe, label_r in vorhanden:
    rek = rekonstruktionen[rauschen]
    ax.plot(rek['q1_rek'], q2_ref, q3_ref, color=farbe, lw=1.3, ls='--',
            label=f'{label_r} ($\\alpha_{{\\min}}={rek["alpha_min_blr"]:.3f}$)', zorder=5)

Axenbeschriftung(ax)
ax.set_title(r'Trajektorienrekonstruktion bei $\alpha_{\min}^{\mathrm{BLR}}$' + '\n',
             fontsize=fontsize)
ax.legend(frameon=False, fontsize=fontsize, loc='best')
fig.tight_layout()
save_diagram(fig, 'blr_trajektorienrekonstruktion_3d.pdf', unterordner = 'physik_rekonstruktion')
plt.show()