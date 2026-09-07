import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.linalg import solve_triangular
from ii_Skripte_Visualisierung import berechnungen as ber
from ii_Skripte_Visualisierung.setup_plot import (farben_rauschen, labels_rauschen,
                                                  style_ax, save_diagram, LABEL_Q1,
                                                  LABEL_ZEITINDEX, LABEL_REFERENZ, fontsize,
                                                  klartext, prozent, LABEL_ALPHA_MIN_BLR,
                                                  SYM_ALPHA_MIN_BLR)
from ii_Skripte_Visualisierung.farben import WEISS, SCHWARZ
from i_Programmierung.Methoden import compressed_sensing as cs
from i_Programmierung.Methoden import bayessche_lineare_regression as blr
from i_Programmierung.Modellierung.mannigfaltigkeiten import X_ref as _X_ref_manifold

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

NIVEAU = 0.95

_C_REF = float(np.max(np.abs(np.asarray(_X_ref_manifold)[0])))


def blr_posterior(rauschen, alpha, wiederholung=0):
    y_liste, A, idx, sigma_liste, _eps = blr.erzeuge_messdaten(rauschen, float(alpha), wiederholung)
    y     = np.asarray(y_liste[0], dtype=np.float64)
    A     = np.asarray(A, dtype=np.float64)
    sigma = float(sigma_liste[0]) if sigma_liste[0] > 0 else 1e-6
    dc    = max(float(np.max(np.abs(A.T @ y))), _C_REF)

    Nt    = A.shape[1]
    scale = dc / (np.arange(Nt) + 1.0)
    Lam   = np.diag(1.0 / scale**2) + (A.T @ A) / sigma**2
    Lam   = 0.5 * (Lam + Lam.T)
    mu    = np.linalg.solve(Lam, (A.T @ y) / sigma**2)
    assert np.allclose(mu, blr.posterior_mean(A, y, sigma, dc), rtol=0, atol=1e-12)

    L = np.linalg.cholesky(Lam)
    return {'y': y, 'A': A, 'idx': np.asarray(idx), 'sigma': sigma, 'mu': mu, 'L': L}


def q1_standardabweichung(L):
    M = solve_triangular(L, cs.Psi_inv.T, lower=True)
    return np.sqrt(np.einsum('ij,ij->j', M, M))

blr_daten = ber.blr_kurven()
meta      = blr_daten['meta']

ergebnisse = []
for rauschen, farbe, label_r in zip(meta['rauschen_liste'], farben_rauschen, labels_rauschen):
    alpha_min = blr_daten['kurven'][float(rauschen)]['alpha_min_blr']
    if alpha_min is None:
        print(f' kein alpha_min_blr für Rauschen {klartext(label_r)}')
        continue
    post = blr_posterior(float(rauschen), float(alpha_min))
    ergebnisse.append({
        'farbe': farbe, 'label': label_r, 'alpha_min': float(alpha_min), 'post': post,
        'q1_mittel': cs.Psi_inv @ post['mu'],
        'sd_q1': q1_standardabweichung(post['L'])})
if not ergebnisse:
    raise SystemExit('kein alpha_min_blr vorhanden')

q1_ref = np.asarray(cs.X_ref)[:, 0]
N      = meta['N']
n_git  = np.arange(N)
z      = stats.norm.ppf(0.5 + NIVEAU / 2)

# --- Plot -------------------------------------------------------------------
fig, (ax_band, ax_zoom) = plt.subplots(2, 1, figsize=(7.0, 7.2), dpi=300, sharex=True)
style_ax(ax_band)
style_ax(ax_zoom)

for erg in ergebnisse:
    q1_mittel, sd_q1 = erg['q1_mittel'], erg['sd_q1']
    ax_band.fill_between(n_git, q1_mittel - z * sd_q1, q1_mittel + z * sd_q1,
                         color=erg['farbe'], alpha=0.20, linewidth=0, zorder=1)
    ax_band.plot(n_git, q1_mittel, color=erg['farbe'], lw=1.3, zorder=3)
    ax_zoom.plot(n_git, q1_mittel, color=erg['farbe'], lw=1.1, ls='--', zorder=3)

ax_band.plot(n_git, q1_ref, color=SCHWARZ, lw=1.6, zorder=5)
ax_zoom.plot(n_git, q1_ref, color=SCHWARZ, lw=1.6, zorder=5)

unten = min(float((e['q1_mittel'] - z * e['sd_q1']).min()) for e in ergebnisse)
oben  = max(float((e['q1_mittel'] + z * e['sd_q1']).max()) for e in ergebnisse)
ax_band.set_ylim(unten - 0.04 * (oben - unten), oben + 0.04 * (oben - unten))
ax_band.set_ylabel(LABEL_Q1, fontsize=fontsize)
ax_band.set_title(f'{prozent(NIVEAU)}-Kredibilitätsbänder von $q_1$', fontsize=fontsize)
ax_band.grid(True, which='both', linewidth=0.3, alpha=0.4)

rand = 0.08 * (q1_ref.max() - q1_ref.min())
ax_zoom.set_ylim(q1_ref.min() - rand, q1_ref.max() + rand)
ax_zoom.set_xlabel(LABEL_ZEITINDEX, fontsize=fontsize)
ax_zoom.set_ylabel(LABEL_Q1, fontsize=fontsize)
ax_zoom.set_title('Ausschnitt auf die Schwingungsamplitude', fontsize=fontsize)
ax_zoom.grid(True, which='both', linewidth=0.3, alpha=0.4)

handles = [Line2D([], [], color=SCHWARZ, lw=1.6, label=LABEL_REFERENZ)]
handles += [Line2D([], [], color=e['farbe'], lw=1.4,
                   label=f'Rauschen {e["label"]} '
                         rf'(${SYM_ALPHA_MIN_BLR} = {e["alpha_min"]:.3f}$, '
                         rf'$k = {e["post"]["idx"].size}$)')
            for e in ergebnisse]
handles += [Patch(facecolor='0.55', alpha=0.20, edgecolor='none',
                  label=f'{prozent(NIVEAU)} Kredibilitätsband')]

fig.suptitle(f'BLR-Posterior von $q_1$ bei {LABEL_ALPHA_MIN_BLR} '
             'für alle untersuchten Rauschpegel', fontsize=fontsize)
fig.tight_layout(rect=(0, 0.11, 1, 1))
legende = fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=fontsize - 2,
                     frameon=True, facecolor=WEISS, edgecolor=SCHWARZ, framealpha=1.0)
legende.get_frame().set_linewidth(0.6)
save_diagram(fig, 'blr_kredibilitaetsbaender.pdf', unterordner='bayes')
plt.show()

for erg in ergebnisse:
    q1_mittel, post = erg['q1_mittel'], erg['post']
    korr = float(np.corrcoef(q1_mittel - q1_mittel.mean(), q1_ref - q1_ref.mean())[0, 1])
    print(f'Rauschen {klartext(erg["label"])}: alpha_min_blr = {erg["alpha_min"]:.4f}, '
          f'k = {post["idx"].size}, sigma = {post["sigma"]:.3e}, '
          f'Formkorrelation = {korr:+.4f}, Median sd(q1) = {np.median(erg["sd_q1"]):.3e}')
print(f'Schwingungsnorm der Referenz = {cs.norm_q1_schwingung(cs.X_ref):.4f}')
