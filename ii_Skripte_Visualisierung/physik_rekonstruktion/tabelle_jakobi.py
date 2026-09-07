import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(current_dir))
import matplotlib.pyplot as plt
from ii_Skripte_Visualisierung.farben import SCHWARZ
from i_Programmierung.Methoden import compressed_sensing as cs
from ii_Skripte_Visualisierung import berechnungen as b
from i_Programmierung import Konfiguration as k
from ii_Skripte_Visualisierung.setup_plot import (save_diagram, prozent, zahl,
                                                  LABEL_RAUSCHPEGEL)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = 15

cs_basis  = b.cs_kurven()
blr_basis = b.blr_kurven()
C_ref = float(k.C_referenz)
rauschen_liste = sorted(float(r) for r in cs_basis['meta']['rauschen_liste'])

def _fmt(x, spezifikation):
    return zahl(x, spezifikation) if x is not None else "—"

zeilen = []
for r in rauschen_liste:
    label = prozent(r)

    cs_daten  = cs_basis['kurven'][r]
    blr_daten = blr_basis['kurven'][r]

    C_rek_cs  = cs_daten['C_rek_bei_alpha_min']
    dC_cs     = cs_daten['delta_C_bei_alpha_min']
    C_rek_blr = blr_daten['C_rek_bei_alpha_min']
    dC_blr    = blr_daten['delta_C_bei_alpha_min']

    zeilen.append((
        label,
        _fmt(C_ref, ".6f"),
        _fmt(C_rek_cs, ".6f"),
        _fmt(dC_cs, ".2e"),
        _fmt(C_rek_blr, ".6f"),
        _fmt(dC_blr, ".2e"),
    ))

col_w = [1.4, 1.1, 1.1, 1.1, 1.1, 1.1]
col_x = [0]
for w_col in col_w:
    col_x.append(col_x[-1] + w_col)
table_w = col_x[-1]

row_h_main = 0.7
row_h_data = 0.65

header_h = row_h_main
total_h = header_h + len(zeilen) * row_h_data

fig, ax = plt.subplots(figsize=(9, 2.2))
ax.set_xlim(0, table_w)
ax.set_ylim(0, total_h)
ax.invert_yaxis()
ax.axis("off")

def cx(i):
    return (col_x[i] + col_x[i + 1]) / 2

haupt = [
    LABEL_RAUSCHPEGEL,
    r"$C_{\mathrm{ref}}$",
    r"$C_{\mathrm{rek}}^{\mathrm{CS}}$",
    r"$\Delta C^{\mathrm{CS}}$",
    r"$C_{\mathrm{rek}}^{\mathrm{BLR}}$",
    r"$\Delta C^{\mathrm{BLR}}$",
]
y_main = row_h_main / 2
for i, txt in enumerate(haupt):
    ax.text(cx(i), y_main, txt, ha="center", va="center", fontsize=fontsize, fontweight="normal")

ax.plot([0, table_w], [header_h, header_h], color=SCHWARZ, linewidth=1)

for r_idx, zeile in enumerate(zeilen):
    y_top = header_h + r_idx * row_h_data
    y_center = y_top + row_h_data / 2
    if r_idx % 2 == 1:
        ax.add_patch(plt.Rectangle((0, y_top), table_w, row_h_data,
                                    facecolor="#f5f5f5", edgecolor="none", zorder=0))
    for i, val in enumerate(zeile):
        ax.text(cx(i), y_center, val, ha="center", va="center", fontsize=fontsize, fontweight="normal")

plt.tight_layout()
save_diagram(fig, 'jacobi_tabelle.pdf', unterordner = 'physik_rekonstruktion')
plt.show()
