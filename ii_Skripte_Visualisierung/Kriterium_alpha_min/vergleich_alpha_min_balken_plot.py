import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt

from ii_Skripte_Visualisierung import berechnungen as ber
from ii_Skripte_Visualisierung.setup_plot import (labels_rauschen, farben_balken,
                                                  style_ax, save_diagram, fontsize,
                                                  LABEL_ALPHA_MIN, LABEL_ALPHA_MIN_CS,
                                                  LABEL_ALPHA_MIN_BLR, LABEL_RAUSCHPEGEL)
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

x = np.arange(len(rauschen_liste))
breite = 0.35

def _balken(positionen, werte, farbe, label):
    hoehen = [wert if wert is not None else 0.0 for wert in werte]
    balken = ax.bar(positionen, hoehen, breite, color=farbe, label=label)
    for pos, wert in zip(positionen, werte):
        if wert is not None:
            ax.text(pos, wert, f'{wert:.3f}', ha='center', va='bottom', fontsize=fontsize)
        else:
            ax.text(pos, 0.005, '—', ha='center', va='bottom', fontsize=fontsize)
    return balken

_balken(x - breite / 2, werte_cs,  farben_balken[0], LABEL_ALPHA_MIN_CS)
_balken(x + breite / 2, werte_blr, farben_balken[1], LABEL_ALPHA_MIN_BLR)

alle = [w for w in werte_cs + werte_blr if w is not None]
ax.set_ylim(0, max(alle) * 1.15)

ax.set_xticks(x)
ax.set_xticklabels(labels_rauschen)
ax.set_xlabel(LABEL_RAUSCHPEGEL, fontsize=fontsize)
ax.set_ylabel(LABEL_ALPHA_MIN, fontsize=fontsize)
ax.set_title(f'Vergleich {LABEL_ALPHA_MIN_CS} vs. {LABEL_ALPHA_MIN_BLR}', fontsize=fontsize)

legend = ax.legend(fontsize=fontsize, loc='lower right', frameon=True, facecolor=WEISS, edgecolor=SCHWARZ)
legend.get_frame().set_linewidth(0.6)

ax.grid(True, axis='y', linewidth=0.3, alpha=0.4)
fig.tight_layout()
save_diagram(fig, 'vergleich_alpha_min_cs_blr_balken.pdf', unterordner = 'Kriterium_alpha_min')
plt.close(fig)