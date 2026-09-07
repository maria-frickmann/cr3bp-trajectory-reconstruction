from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ZIEL_ORDNER = Path(__file__).resolve().parents[2] / 'iii_Daten_Plots' / 'weitere'
ZIEL_ORDNER.mkdir(parents=True, exist_ok=True)
ZIEL_PFAD = ZIEL_ORDNER / 'cr3bp_setup_2d.pdf'

fontsize=11

fig, ax = plt.subplots(figsize=(6, 3.8))

mu = 0.155
pos_ms = np.array([-mu, 0.0])
pos_me = np.array([1.0 - mu, 0.0])
baryzentrum = np.array([0.0, 0.0])
pos_jwst = np.array([0.26, 0.40])

theta = np.linspace(0, 2 * np.pi, 400)
ax.plot(mu * np.cos(theta), mu * np.sin(theta), color="#5B9BD5", lw=1.2, zorder=1)
ax.plot((1 - mu) * np.cos(theta), (1 - mu) * np.sin(theta), color="#D9D9D9", lw=1.2, zorder=1)

ax.annotate("", xy=(1.22, 0), xytext=(-0.52, 0), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
ax.annotate("", xy=(0, 0.62), xytext=(0, -0.40), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

ax.text(1.26, 0, r"$q_1$", fontsize=fontsize, va="center")
ax.text(0.015, 0.66, r"$q_2$", fontsize=fontsize, ha="center")

ax.plot([pos_ms[0], pos_jwst[0]], [pos_ms[1], pos_jwst[1]], "k--", lw=1.6, zorder=2)
ax.plot([pos_me[0], pos_jwst[0]], [pos_me[1], pos_jwst[1]], "k--", lw=1.6, zorder=2)

ax.scatter(*pos_ms, s=200, color="#5B9BD5", edgecolor="black", zorder=5)
ax.scatter(*pos_me, s=200, color="#D9D9D9", edgecolor="black", zorder=5)
ax.scatter(*pos_jwst, s=50, color="#2CA05A", edgecolor="black", zorder=5)
ax.scatter(*baryzentrum, s=30, color="red", zorder=6)

ax.text(pos_ms[0] - 0.01, pos_ms[1] + 0.10, r"$q_{Sonne}$", fontsize=fontsize, ha="center", style="italic")
ax.text(pos_me[0] + 0.02, pos_me[1] + 0.08, r"$q_{Erde}$", fontsize=fontsize, ha="center", style="italic")
ax.text(pos_jwst[0], pos_jwst[1] + 0.055, r"$q_{Sonde}$", fontsize=fontsize, ha="center", style="italic")
ax.text(baryzentrum[0] + 0.12, baryzentrum[1] - 0.075, "Baryzentrum", fontsize=fontsize, style="italic", ha="center", zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

mid_rs = (pos_ms + pos_jwst) / 2
mid_re = (pos_me + pos_jwst) / 2
ax.text(mid_rs[0] - 0.005, mid_rs[1] + 0.05, r"$r_{\overline{SS}}$", fontsize=10, ha="left")
ax.text(mid_re[0] + 0.02, mid_re[1] + 0.02, r"$r_{\overline{ES}}$", fontsize=10, ha="left")

ax.set_xlim(-0.55, 1.32)
ax.set_ylim(-0.42, 0.74)
ax.set_aspect("equal")
ax.set_axis_off()

plt.tight_layout()
plt.savefig(ZIEL_PFAD, dpi=1200, bbox_inches="tight")
plt.close()
