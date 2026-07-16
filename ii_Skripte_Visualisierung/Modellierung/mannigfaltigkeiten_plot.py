# manifold.py

import sys
from pathlib import Path
my_path  = Path(__file__).resolve().parents[2]
my_file  = 'Mannigfaltigkeiten_Halo.pdf'
sys.path.append(str(my_path ))
ziel_ordner = my_path / 'iii_Daten_Plots' / 'Modelle'
speicher_pfad = ziel_ordner / my_file
ziel_ordner.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from i_Programmierung.Konstanten import mu
from i_Programmierung.Modellierung.lagrange import L2
from i_Programmierung.Modellierung.orbit import sol
from i_Programmierung.Modellierung.mannigfaltigkeiten import (tj_stabil, tj_instabil)
from ii_Skripte_Visualisierung.farben import *
from ii_Skripte_Visualisierung.hintergrund import draw_hintergrund_3d, Axenbeschriftung
from ii_Skripte_Visualisierung.setup_plot import PLOT_STYLE, get_manifold_window, save_diagram, HALO_ORBIT_LABEL

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

xlim, ylim, zlim = get_manifold_window(L2[0], mu, abst=0.007)
fig = plt.figure(figsize=PLOT_STYLE['figsize'], facecolor=WEISS)
ax = fig.add_subplot(111, projection='3d')
pad = 0.02                      

draw_hintergrund_3d(ax, xlim=xlim, ylim=ylim, zlim=zlim)
Axenbeschriftung(ax)
ax.view_init(elev=40, azim=-110)
ax.plot(sol.y[0], sol.y[1], sol.y[2], 
        color=ORBIT, label=HALO_ORBIT_LABEL, **PLOT_STYLE['orbit_line'])
for i, traj in enumerate(tj_stabil):
    ax.plot(traj.y[0], traj.y[1], traj.y[2], color=tj_stabil_farbe, 
            label='Stabile Mannigfaltigkeit $W^s$' if i == 0 else None, **PLOT_STYLE['manifold_line'])
for i, traj in enumerate(tj_instabil):
    ax.plot(traj.y[0], traj.y[1], traj.y[2], color=tj_instabil_farbe, 
            label='Instabile Mannigfaltigkeit $W^u$' if i == 0 else None, **PLOT_STYLE['manifold_line'])
ax.set_xlim3d(*xlim)
ax.set_ylim3d(*ylim)
ax.set_zlim3d(*zlim)
ax.legend()
ax.set_title('Mannigfaltigkeiten des Halo-Orbits um L2', fontsize=PLOT_STYLE['title_size'])
save_diagram(fig, speicher_pfad)