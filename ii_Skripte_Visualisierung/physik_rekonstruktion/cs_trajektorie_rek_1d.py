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

daten            = ber.trajektorien_rekonstruktion()
meta             = daten['meta']
rekonstruktionen = daten['rekonstruktionen']
X_ref            = meta['X_ref']
q1_ref, q2_ref   = X_ref[:, 0], X_ref[:, 1]
N                = meta['N']
vorhanden = vorhandene_rekonstruktionen(meta, rekonstruktionen, 'alpha_min_cs',
                                        farben_rauschen, labels_rauschen)
grenzen_q1 = q1_grenzen(q1_ref, rekonstruktionen, vorhanden)
fig_zeit, ax_zeit = plt.subplots(figsize=(6, 4.2), dpi=300)
style_ax(ax_zeit)

ax_zeit.plot(np.arange(N), q1_ref, color=SCHWARZ, lw=1.5, label=LABEL_REFERENZ, zorder=5)
for rauschen, farbe, label_r in vorhanden:
    rek = rekonstruktionen[rauschen]
    ax_zeit.plot(np.arange(N), rek['q1_rek'], color=farbe, lw=1.2, ls='--', alpha=0.9,
                 label=f'{label_r} ($Δ_{{\\mathrm{{total}}}}='
                       f'{rek["delta_total"]:.4f}$)')

ax_zeit.set_ylim(*grenzen_q1)
ax_zeit.set_xlabel(LABEL_ZEITINDEX)
ax_zeit.set_ylabel(LABEL_Q1, fontsize=fontsize)
ax_zeit.set_title(r'$q_1$ über der Zeit bei $\alpha_{\min}^{\mathrm{CS}}$' + '\n',
                   fontsize=fontsize)

ax_zeit.legend(frameon=False, fontsize=fontsize, loc='best')
ax_zeit.grid(True, which='both', linewidth=0.3, alpha=0.4)

fig_zeit.tight_layout()
save_diagram(fig_zeit, 'cs_trajektorienrekonstruktion_1d.pdf', unterordner = 'physik_rekonstruktion')
plt.show()