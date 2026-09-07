import sys
from pathlib import Path
my_path = Path(__file__).resolve().parents[2]
my_file = "parameter_uebersicht.pdf"
sys.path.append(str(my_path))
ziel_ordner = my_path / "iii_Daten_Plots" / "Modellierung"
speicher_pfad = ziel_ordner / my_file
ziel_ordner.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
from i_Programmierung.Konstanten import N_WDH, RHO, w, startwert_zufall, delta
from ii_Skripte_Visualisierung.farben import SCHWARZ
from ii_Skripte_Visualisierung.farben import *
sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize, zahl
from ii_Skripte_Visualisierung.setup_plot import save_diagram

fontsize = fontsize

parameter = [
    (r"$N$", "wird aus der Referenzbahn bestimmt"),
    (r"$N_{\mathrm{WDH}}$", N_WDH),
    (r"$\rho$", RHO),
    (r"$w$", w),
    (r"$\delta$", delta)
]

def _fmt(x, spezifikation):
    return format(x, spezifikation) if x is not None else "—"

# Tabelleninhalt
zeilen = [[name, wert if isinstance(wert, str) else zahl(wert)] for name, wert in parameter]

col_w = [1.0, 2.0]
col_x = [0]
for w_col in col_w:
    col_x.append(col_x[-1] + w_col)
table_w = col_x[-1]

row_h_main = 0.7
row_h_data = 0.65

header_h = row_h_main
total_h = header_h + len(zeilen) * row_h_data

fig_st, ax = plt.subplots(figsize=(10, 2.2))
ax.set_xlim(0, table_w)
ax.set_ylim(0, total_h)
ax.invert_yaxis()
ax.axis("off")

def cx(i):
    return (col_x[i] + col_x[i + 1]) / 2

haupt = ["Parameter", "Wert"]

y_main = row_h_main / 2
for i, txt in enumerate(haupt):
    ax.text( cx(i), y_main, txt, ha="center", va="center", fontsize=fontsize, fontweight="normal",)

ax.plot([0, table_w], [header_h, header_h], color=SCHWARZ, linewidth=1)

for r_idx, zeile in enumerate(zeilen):
    y_top = header_h + r_idx * row_h_data
    y_center = y_top + row_h_data / 2

    if r_idx % 2 == 1:
        ax.add_patch(
            plt.Rectangle( (0, y_top), table_w, row_h_data, facecolor="#f5f5f5", edgecolor="none", zorder=0,))

    for i, val in enumerate(zeile):
        ax.text( cx(i), y_center, val, ha="center", va="center", fontsize=fontsize, fontweight="normal", )

save_diagram(fig_st, speicher_pfad)

plt.close(fig_st)