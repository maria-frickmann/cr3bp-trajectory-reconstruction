# st_diagramm.py
# Import
import sys
from pathlib import Path
my_path  = Path(__file__).resolve().parents[2]
my_file  = 'st_vt_at_diagramm.pdf'
sys.path.append(str(my_path ))
ziel_ordner = my_path / 'iii_Daten_Plots' / 'Modelle'
speicher_pfad = ziel_ordner / my_file
ziel_ordner.mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
from i_Programmierung.Konstanten import Radius, L_fundamental
from i_Programmierung.Modellierung.mannigfaltigkeiten import t, abstand_km, v_opt, a_opt
from ii_Skripte_Visualisierung.farben import *
from ii_Skripte_Visualisierung.setup_plot import save_diagram

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

radius_erde_km = Radius * L_fundamental / 1e3

fig_st, axes = plt.subplots(3, 1, figsize=(10, 16), facecolor=WEISS, sharex=True)

# s-t-Diagramm
axes[0].plot(t, abstand_km, color=tj_optimal_farbe, lw=1.5, label='Optimale Transferbahn')
axes[0].axhline(radius_erde_km, color=ERDE, ls='--', lw=1.0,
                label=f'Erdradius ({radius_erde_km:.0f} km)')
axes[0].set_ylabel('Abstand [km]')
axes[0].set_title('Abstand Trajektorie - Erde')
axes[0].legend()

# v-t-Diagramm
axes[1].plot(t, v_opt, color=tj_optimal_farbe, lw=1.5)
axes[1].set_ylabel('Geschwindigkeit [normiert]')
axes[1].set_title('Betrag der Geschwindigkeit')

# a-t-Diagramm
axes[2].plot(t, a_opt, color=tj_optimal_farbe, lw=1.5)
axes[2].set_xlabel('Zeit [TU]')
axes[2].set_ylabel('Beschleunigung [normiert]')
axes[2].set_title('Betrag der Beschleunigung')

for ax in axes:
    ax.set_facecolor(WEISS)
    ax.grid(True)

plt.tight_layout()
save_diagram(fig_st, speicher_pfad)