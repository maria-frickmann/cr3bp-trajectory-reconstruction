import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap, to_rgb
from scipy.optimize import fsolve
from i_Programmierung.Modellierung.physik import U_mu
from i_Programmierung.Modellierung.lagrange import polynom_l1, polynom_l2, polynom_l3
from ii_Skripte_Visualisierung.farben import SONNE, ERDE, SCHWARZ, WEISS
from ii_Skripte_Visualisierung.setup_plot import (fontsize, PLOT_STYLE, save_diagram,
                                                  inhalt_bereich, LAGRANGE_LABELS)
sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

mu = 0.08
X1, X2 = -mu, 1.0 - mu

def dV_dx(x, mu):
    r1 = np.abs(x - X1)
    r2 = np.abs(x - X2)
    return -x + (1 - mu) * (x - X1) / r1**3 + mu * (x - X2) / r2**3

start_L1 = (X1 + X2) / 2          
start_L2 = X2 + 0.5               
start_L3 = X1 - 1.0               

x_L1 = fsolve(dV_dx, start_L1, args=(mu,))[0]
x_L2 = fsolve(dV_dx, start_L2, args=(mu,))[0]
x_L3 = fsolve(dV_dx, start_L3, args=(mu,))[0]

L1 = np.array([x_L1, 0.0, 0.0])
L2 = np.array([x_L2, 0.0, 0.0])
L3 = np.array([x_L3, 0.0, 0.0])
L4 = np.array([0.5 - mu, np.sqrt(3) / 2, 0.0])
L5 = np.array([0.5 - mu, -np.sqrt(3) / 2, 0.0]) 

l_punkte = [L1, L2, L3, L4, L5]

def style_ax_potential(fig, ax):
    fig.patch.set_facecolor(WEISS)
    ax.set_facecolor(WEISS)
    ax.set_axis_off()

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)

N = 900
LIM = 1.9
x = np.linspace(-LIM, LIM, N)
y = np.linspace(-LIM, LIM, N)
X, Y = np.meshgrid(x, y)
Z = U_mu(X, Y, 0.0, mu)
Z_FLOOR = -3.6
Zc = np.clip(Z, Z_FLOOR, None)
R = np.sqrt(X ** 2 + Y ** 2)
mask = R > LIM * 0.98
Zm = np.where(mask, np.nan, Zc)
base_cmap = LinearSegmentedColormap.from_list("darksurf", [SCHWARZ, SCHWARZ, SCHWARZ, SCHWARZ])
ls = LightSource(azdeg=315, altdeg=55)
rgb = ls.shade(np.nan_to_num(Zm, nan=Z_FLOOR), cmap=base_cmap, vert_exag=0.25, blend_mode="soft")

r1 = np.sqrt((X - X1) ** 2 + Y ** 2)
glow_sonne = np.exp(-(r1 / 0.55) ** 2)[..., None]
sonne_farbe = np.array(to_rgb("#E8E9AA"))
rgb[..., :3] = np.clip(rgb[..., :3] * (1 - 0.85 * glow_sonne) + sonne_farbe * 0.9 * glow_sonne, 0, 1)

r2 = np.sqrt((X - X2) ** 2 + Y ** 2)
glow_erde = np.exp(-(r2 / 0.16) ** 2)[..., None]
erde_farbe = np.array(to_rgb("#DBE6F3"))
rgb[..., :3] = np.clip(rgb[..., :3] * (1 - 0.5 * glow_erde) + erde_farbe * 0.45 * glow_erde, 0, 1)

stride = 3
levels = -np.unique(np.concatenate([np.linspace(1.52, 2.2, 26), np.geomspace(2.2, -Z_FLOOR, 18),]))
levels = np.sort(levels[levels > Z_FLOOR + 0.02])

pixel = stride * 2 * LIM / N
gy, gx = np.gradient(Zc, y, x)
grad = np.sqrt(gx ** 2 + gy ** 2)
halbbreite = np.maximum(grad * pixel * 0.7, 1e-6)

linien_maske = np.zeros_like(Zc)
for lv in levels:
    d = np.abs(Zc - lv)
    linien_maske = np.maximum(
        linien_maske, np.clip(1.0 - d / halbbreite, 0.0, 1.0))

linien_maske = np.where(mask, 0.0, linien_maske)
linien_maske = np.where(Zc <= Z_FLOOR + 0.01, 0.0, linien_maske)
lm = (0.9 * linien_maske)[..., None]
linien_farbe = np.array(to_rgb("#9da0c9"))
rgb[..., :3] = np.clip(rgb[..., :3] * (1 - lm) + linien_farbe * lm, 0, 1)

rgb[..., 3] = np.where(mask, 0.0, 1.0)

fig = plt.figure(figsize=PLOT_STYLE['figsize'], facecolor=WEISS, dpi=1200)
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
style_ax_potential(fig, ax)

ax.plot_surface(X, Y, Zm,
                facecolors=rgb,
                rstride=1, cstride=1, 
                linewidth=0,
                antialiased=True,
                shade=False, zorder=1)

for name, p in zip(LAGRANGE_LABELS, l_punkte):
    lx, ly = p[0], p[1]
    lz = U_mu(lx, ly, 0.0, mu) + 0.02
    ax.scatter([lx], [ly], [lz], color=WEISS, s=14, zorder=6)
    ax.plot([lx, lx], [ly, ly], [lz, lz + 0.14],
            color=WEISS, lw=1.0, zorder=6)
    ax.text(lx, ly, lz + 0.18, name, color=SCHWARZ, fontsize=fontsize + 1,
            ha="center", va="bottom", zorder=7,
            bbox=dict(facecolor=WEISS, edgecolor=SCHWARZ,
                      boxstyle="square,pad=0.25", linewidth=0.4))
ax.view_init(elev=38, azim=-75)
ax.set_zlim(Z_FLOOR, -1.1)
ax.set_xlim(-LIM, LIM)
ax.set_ylim(-LIM, LIM)
ax.set_box_aspect((1, 1, 0.55))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

save_diagram(fig, 'lagrange_potentialflaeche.png', unterordner='Modellierung',
             bereich=inhalt_bereich(fig))

plt.close(fig)