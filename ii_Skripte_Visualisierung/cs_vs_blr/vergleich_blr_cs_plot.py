import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from ii_Skripte_Visualisierung import berechnungen as ber
from ii_Skripte_Visualisierung.setup_plot import (labels_rauschen, farben_balken,
                                                  style_ax, save_diagram, fontsize,
                                                  LABEL_KORR_CS, LABEL_KORR_BLR,
                                                  LABEL_ALPHA_MIN_CS, LABEL_ALPHA_MIN_BLR,
                                                  LABEL_MESSANTEIL, LABEL_FORMKORRELATION)
from ii_Skripte_Visualisierung.farben import WEISS, SCHWARZ

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

cs_daten  = ber.cs_kurven()
blr_daten = ber.blr_kurven()
meta      = cs_daten['meta']
RAUSCHEN = 0.001

def _rauschen_aus_argv(standard):
    if len(sys.argv) > 1:
        try:
            return float(sys.argv[1])
        except ValueError:
            pass
    return standard

rauschen_plot = _rauschen_aus_argv(RAUSCHEN)
if not any(np.isclose(rauschen_plot, meta['rauschen_liste'])):
    raise SystemExit(f"Rauschpegel {rauschen_plot} liegt nicht in "
                     f"rauschen_liste={meta['rauschen_liste']}")
index_rauschen = int(np.argmin(np.abs(np.asarray(meta['rauschen_liste']) - rauschen_plot)))
rauschen_plot  = float(meta['rauschen_liste'][index_rauschen])
label_rauschen = labels_rauschen[index_rauschen]
cs_kurve  = cs_daten['kurven'][rauschen_plot]
blr_kurve = blr_daten['kurven'][rauschen_plot]
farbe_cs, farbe_blr = farben_balken[0], farben_balken[1]
fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
style_ax(ax)

rho = cs_kurve['rho']
ax.axhline(rho, color='0.35', linestyle='dashed', linewidth=1.1,
           label=rf'$\rho = {rho:.2f}$')

ax.plot(cs_kurve['alpha'], cs_kurve['korr_avg'], marker='o', markersize=4,
        linewidth=1.4, color=farbe_cs,
        label=LABEL_KORR_CS)
ax.plot(blr_kurve['alpha'], blr_kurve['korr_avg'], marker='s', markersize=4,
        linewidth=1.4, color=farbe_blr,
        label=LABEL_KORR_BLR)

if cs_kurve['alpha_min_cs'] is not None:
    ax.axvline(cs_kurve['alpha_min_cs'], color=farbe_cs, linestyle='-.', linewidth=1.2,
               label=LABEL_ALPHA_MIN_CS)
if blr_kurve['alpha_min_blr'] is not None:
    ax.axvline(blr_kurve['alpha_min_blr'], color=farbe_blr, linestyle='-.', linewidth=1.2,
               label=LABEL_ALPHA_MIN_BLR)

ax.set_xlabel(LABEL_MESSANTEIL, fontsize=fontsize)
ax.set_ylabel(LABEL_FORMKORRELATION, fontsize=fontsize)
ax.set_ylim(-0.15, 1.05)
ax.set_title(f'Vergleich von CS und BLR bei Rauschen {label_rauschen}', fontsize=fontsize)

handles, labels = ax.get_legend_handles_labels()

def sortier_schluessel(label):
    if r'\rho' in label:
        return 0
    if r'\alpha_{\min}' in label:
        return 1
    return 2

reihenfolge = sorted(range(len(labels)), key=lambda i: sortier_schluessel(labels[i]))
handles = [handles[i] for i in reihenfolge]
labels  = [labels[i] for i in reihenfolge]

legend = ax.legend(handles, labels, fontsize=fontsize, loc='lower right',
                    frameon=True, facecolor=WEISS, edgecolor=SCHWARZ)
legend.get_frame().set_linewidth(0.6)

ax.grid(True, which='both', linewidth=0.3, alpha=0.4)
fig.tight_layout()
save_diagram(fig, 'vergleich_blr_cs.pdf', unterordner='cs_vs_blr')
plt.show()