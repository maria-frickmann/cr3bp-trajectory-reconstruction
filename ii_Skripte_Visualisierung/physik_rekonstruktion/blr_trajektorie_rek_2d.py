import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from ii_Skripte_Visualisierung import berechnungen as ber
from ii_Skripte_Visualisierung.setup_plot import (farben_rauschen, labels_rauschen,
                                                  style_ax, save_diagram, LABEL_Q1,
                                                  LABEL_Q2, LABEL_ZEITINDEX,
                                                  LABEL_REFERENZ, fontsize)
from ii_Skripte_Visualisierung.farben import SCHWARZ
from ii_Skripte_Visualisierung.physik_rekonstruktion._gemeinsam import (
    vorhandene_rekonstruktionen, q1_grenzen)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

daten            = ber.blr_trajektorien_rekonstruktion()
meta             = daten['meta']
rekonstruktionen = daten['rekonstruktionen']
X_ref            = meta['X_ref']
q1_ref, q2_ref   = X_ref[:, 0], X_ref[:, 1]
N                = meta['N']

vorhanden = vorhandene_rekonstruktionen(meta, rekonstruktionen, 'alpha_min_blr',
                                        farben_rauschen, labels_rauschen)
grenzen_q1 = q1_grenzen(q1_ref, rekonstruktionen, vorhanden)
fig_bahn, ax_bahn = plt.subplots(figsize=(6, 4.2), dpi=300)
style_ax(ax_bahn)
ax_bahn.plot(q1_ref, q2_ref, color=SCHWARZ, lw=1.8, label=LABEL_REFERENZ, zorder=5)
for rauschen, farbe, label_r in vorhanden:
    rek = rekonstruktionen[rauschen]
    ax_bahn.plot(rek['q1_rek'], q2_ref, color=farbe, lw=1.2, ls='--', alpha=0.9,
                 label=f'{label_r} ($\\alpha_{{\\min}}={rek["alpha_min_blr"]:.3f}$)')
ax_bahn.set_xlim(*grenzen_q1)
ax_bahn.set_xlabel(LABEL_Q1)
ax_bahn.set_ylabel(LABEL_Q2, fontsize=fontsize)
ax_bahn.set_title(r'Bahnprojektion bei $\alpha_{\min}^{\mathrm{BLR}}$' + '\n', fontsize=fontsize)

ax_bahn.legend(frameon=False, fontsize=fontsize, loc='best')
ax_bahn.grid(True, which='both', linewidth=0.3, alpha=0.4)

fig_bahn.tight_layout()
save_diagram(fig_bahn, 'blr_trajektorienrekonstruktion_2d.pdf', unterordner = 'physik_rekonstruktion')
plt.show()
