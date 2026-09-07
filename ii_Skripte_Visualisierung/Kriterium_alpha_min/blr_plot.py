import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import matplotlib.pyplot as plt
from ii_Skripte_Visualisierung import berechnungen as ber
from ii_Skripte_Visualisierung.setup_plot import (farben_rauschen, labels_rauschen,
                                                  style_ax, save_diagram, fontsize,
                                                  LABEL_KORR_BLR, LABEL_MESSANTEIL,
                                                  LABEL_FORMKORRELATION, SYM_ALPHA_MIN_BLR)
from ii_Skripte_Visualisierung.farben import WEISS, SCHWARZ

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

daten  = ber.blr_kurven()
meta   = daten['meta']
kurven = daten['kurven']

fig, ax = plt.subplots(figsize=(6, 4.2), dpi=300)
style_ax(ax)

rho = kurven[float(meta['rauschen_liste'][0])]['rho']
ax.axhline(rho, color='0.35', linestyle='dashed', linewidth=1.1,
           label=rf'$\rho = {rho:.2f}$')

for rauschen, farbe, label_rauschen in zip(meta['rauschen_liste'],
                                           farben_rauschen, labels_rauschen):
    kurve = kurven[float(rauschen)]
    ax.plot(kurve['alpha'], kurve['korr_avg'], marker='o', markersize=4, linewidth=1.4,
            color=farbe,
            label=f'{LABEL_KORR_BLR} Rauschen {label_rauschen}')
    if kurve['alpha_min_blr'] is not None:
        a_min = kurve['alpha_min_blr']
        ax.axvline(a_min, color=farbe, linestyle='-.', linewidth=1.2,
                   label=rf'${SYM_ALPHA_MIN_BLR} = {a_min:.2f}$ ({label_rauschen})')

ax.set_xlabel(LABEL_MESSANTEIL, fontsize=fontsize)
ax.set_ylabel(LABEL_FORMKORRELATION, fontsize=fontsize)
ax.set_ylim(-0.15, 1.05)
ax.set_title('Formkorrelation der BLR Rekonstruktion für verschiedene Rauschen', fontsize=fontsize)

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
legend = ax.legend(handles, labels, fontsize=fontsize, ncol=1,
                   loc='lower right', bbox_to_anchor=(1.0, 0.0),
                   borderaxespad=0.4, labelspacing=0.35, handlelength=1.6,
                   frameon=True, facecolor=WEISS, edgecolor=SCHWARZ)
legend.get_frame().set_linewidth(0.6)
ax.grid(True, which='both', linewidth=0.3, alpha=0.4)
fig.tight_layout()
save_diagram(fig, 'blr_verschiedene_rauschwerte.pdf', unterordner='Kriterium_alpha_min')
plt.show()