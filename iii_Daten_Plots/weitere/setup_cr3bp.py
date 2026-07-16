

import matplotlib.pyplot as plt
import numpy as np
fontsize=11


fig = plt.figure(figsize=(6, 3.5))
ax = fig.add_subplot(111, projection="3d")
ax.computed_zorder = False

mu = 0.25
pos_ms = np.array([-mu, 0.0, 0.0])
pos_me = np.array([1.4, 0.0, 0.0])
baryzentrum = np.array([0.0, 0.0, 0.0])
pos_jwst = np.array([0.42, 0.65, 0.45])

ax.quiver(-0.75, 0, 0, 2.55, 0, 0, color="black", lw=1.5, arrow_length_ratio=0.03)
ax.quiver(0, -0.55, 0, 0, 1.55, 0, color="black", lw=1.5, arrow_length_ratio=0.05)
ax.quiver(0, 0, -0.45, 0, 0, 1.25, color="black", lw=1.5, arrow_length_ratio=0.06)

ax.text(1.92, 0, -0.06, r"$q_1$", fontsize=10)
ax.text(0.03, 1.08, 0, r"$q_2$", fontsize=10)
ax.text(0.03, 0, 0.85, r"$q_3$", fontsize=10)

ax.plot([pos_ms[0], pos_jwst[0]], [pos_ms[1], pos_jwst[1]], [pos_ms[2], pos_jwst[2]], "k--", lw=1.3)
ax.plot([pos_me[0], pos_jwst[0]], [pos_me[1], pos_jwst[1]], [pos_me[2], pos_jwst[2]], "k--", lw=1.3)

ax.scatter(*pos_ms, s=200, color="#5B9BD5", edgecolor="black", depthshade=False, zorder=5)
ax.scatter(*pos_me, s=200, color="#D9D9D9", edgecolor="black", depthshade=False, zorder=5)
ax.scatter(*pos_jwst, s=50, color="#2CA05A", edgecolor="black", depthshade=False, zorder=2)
ax.scatter(*baryzentrum, s=30, color="red", depthshade=False, zorder=6)

ax.text(pos_ms[0], pos_ms[1], pos_ms[2] + 0.16, r"$m_s$", fontsize=fontsize, ha="center", fontweight="bold")
ax.text(pos_me[0], pos_me[1], pos_me[2] + 0.12, r"$m_e$", fontsize=fontsize, ha="center", fontweight="bold")
ax.text(pos_jwst[0], pos_jwst[1], pos_jwst[2] + 0.10, r"$m_{jwst}$", fontsize=fontsize, ha="center", fontweight="bold")
ax.text(baryzentrum[0] + 0.2, baryzentrum[1] - 0.32, baryzentrum[2], "Baryzentrum", fontsize=6, style="italic", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


mid_rs = (pos_ms + pos_jwst) / 2
mid_re = (pos_me + pos_jwst) / 2
ax.text(mid_rs[0] - 0.06, mid_rs[1] + 0.06, mid_rs[2] + 0.06, r"$r_{\overline{SS}}$", fontsize=9)
ax.text(mid_re[0] + 0.05, mid_re[1] + 0.05, mid_re[2] + 0.06, r"$r_{\overline{ES}}$", fontsize=9)

ax.text(pos_ms[0], pos_ms[1], pos_ms[2] - 0.30, r"$(-\mu,\,0,\,0)$", fontsize=9, ha="center")
ax.text(pos_me[0], pos_me[1], pos_me[2] - 0.26, r"$(1-\mu,\,0,\,0)$", fontsize=9, ha="center")

ax.set_xlim(-0.8, 1.9)
ax.set_ylim(-0.6, 1.1)
ax.set_zlim(-0.5, 0.9)
ax.set_box_aspect((2.7, 1.7, 1.4))
ax.set_axis_off()
ax.view_init(elev=18, azim=-65)

plt.tight_layout()
plt.savefig("cr3bp_setup_3d.png", dpi=1200, bbox_inches="tight")
plt.close()
