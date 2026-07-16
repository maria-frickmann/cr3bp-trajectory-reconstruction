import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from ii_Skripte_Visualisierung import berechnungen as ber
from ii_Skripte_Visualisierung.setup_plot import (labels_rauschen, farben_balken,
                                                  style_ax, save_diagram, fontsize)
from ii_Skripte_Visualisierung.farben import WEISS, SCHWARZ

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

cs_daten  = ber.cs_kurven()
blr_daten = ber.blr_kurven()
rauschen_liste = list(cs_daten['meta']['rauschen_liste'])
werte_cs  = [cs_daten['kurven'][float(r)]['alpha_min_cs'] for r in rauschen_liste]
werte_blr = [blr_daten['kurven'][float(r)]['alpha_min_blr'] for r in rauschen_liste]

fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
style_ax(ax)

x      = np.arange(len(rauschen_liste))
breite = 0.35

def _balken(positionen, werte, farbe, label):
    hoehen = [wert if wert is not None else 0.0 for wert in werte]
    balken = ax.bar(positionen, hoehen, breite, color=farbe, label=label)
    for pos, wert in zip(positionen, werte):
        if wert is not None:
            ax.text(pos, wert, f'{wert:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(pos, 0.005, '—', ha='center', va='bottom', fontsize=11)
    return balken

_balken(x - breite / 2, werte_cs,  farben_balken[0], r'$\alpha_{\min}^{\mathrm{CS}}$')
_balken(x + breite / 2, werte_blr, farben_balken[1], r'$\alpha_{\min}^{\mathrm{BLR}}$')

ax.set_xticks(x)
ax.set_xticklabels(labels_rauschen)
ax.set_xlabel('Rauschpegel')
ax.set_ylabel(r'$\alpha_{\min}$', fontsize=fontsize)
ax.set_title(r'Vergleich $\alpha_{\min}^{\mathrm{CS}}$ vs. $\alpha_{\min}^{\mathrm{BLR}}$', fontsize=fontsize)

legend = ax.legend(fontsize=fontsize, loc='upper right',
                    frameon=True, facecolor=WEISS, edgecolor=SCHWARZ)
legend.get_frame().set_linewidth(0.6)

ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
fig.tight_layout()
save_diagram(fig, 'vergleich_alpha_min_cs_blr_balken.pdf', unterordner = 'Kriterium_alpha_min')
plt.show()