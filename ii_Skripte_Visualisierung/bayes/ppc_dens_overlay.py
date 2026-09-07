import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.linalg import solve_triangular
from ii_Skripte_Visualisierung import berechnungen as ber
from ii_Skripte_Visualisierung.setup_plot import (farben_rauschen, labels_rauschen,
                                                  style_ax, save_diagram, fontsize,
                                                  klartext, LABEL_MESSWERT,
                                                  LABEL_ALPHA_MIN_BLR, SYM_ALPHA_MIN_BLR)
from ii_Skripte_Visualisierung.farben import WEISS, SCHWARZ
from i_Programmierung.Methoden import bayessche_lineare_regression as blr
from i_Programmierung.Konstanten import startwert_zufall
from i_Programmierung.Modellierung.mannigfaltigkeiten import X_ref as _X_ref_manifold

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

S_ZIEHUNGEN = 200

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


def posterior_predictive(post, anzahl, zufallsgenerator):
    eps    = zufallsgenerator.normal(size=(post['mu'].size, anzahl))
    theta  = post['mu'][:, None] + solve_triangular(post['L'].T, eps, lower=False)
    rausch = zufallsgenerator.normal(0.0, post['sigma'], size=(anzahl, post['y'].size))
    return (post['A'] @ theta).T + rausch


blr_daten = ber.blr_kurven()
meta      = blr_daten['meta']
rng       = np.random.default_rng(startwert_zufall)

ergebnisse = []
for rauschen, farbe, label_r in zip(meta['rauschen_liste'], farben_rauschen, labels_rauschen):
    alpha_min = blr_daten['kurven'][float(rauschen)]['alpha_min_blr']
    if alpha_min is None:
        print(f' kein alpha_min_blr für Rauschen {klartext(label_r)}')
        continue
    post  = blr_posterior(float(rauschen), float(alpha_min))
    y     = post['y']
    y_rep = posterior_predictive(post, S_ZIEHUNGEN, rng)
    ergebnisse.append({'farbe': farbe, 'label': label_r, 'alpha_min': float(alpha_min),
                       'post': post, 'y': y, 'y_rep': y_rep})
if not ergebnisse:
    raise SystemExit('kein alpha_min_blr vorhanden')

alle = np.concatenate([np.concatenate([e['y'], e['y_rep'].ravel()]) for e in ergebnisse])
spanne = max(alle.max() - alle.min(), 1e-12)
gitter = np.linspace(alle.min() - 0.05 * spanne, alle.max() + 0.05 * spanne, 512)
for erg in ergebnisse:
    erg['dichte_y']   = stats.gaussian_kde(erg['y'])(gitter)
    erg['dichte_rep'] = np.array([stats.gaussian_kde(erg['y_rep'][s])(gitter)
                                  for s in range(S_ZIEHUNGEN)])

fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=300)
style_ax(ax)

for erg in ergebnisse:
    ax.plot(gitter, erg['dichte_rep'].T, color=erg['farbe'], linewidth=0.6,
            alpha=0.10, zorder=1)
for erg in ergebnisse:
    ax.plot(gitter, erg['dichte_y'], color=erg['farbe'], linewidth=1.8, ls='--', zorder=3)

ax.set_xlabel(LABEL_MESSWERT, fontsize=fontsize)
ax.set_ylabel('Dichte', fontsize=fontsize)
ax.set_xlim(gitter[0], gitter[-1])
ax.set_ylim(0, max(max(e['dichte_y'].max(), e['dichte_rep'].max())
                   for e in ergebnisse) * 1.12)
ax.set_title(f'Posterior Predictive Check der BLR bei {LABEL_ALPHA_MIN_BLR}'
             '\nfür alle untersuchten Rauschpegel', fontsize=fontsize)
ax.grid(True, which='both', linewidth=0.3, alpha=0.4)

handles = [Line2D([], [], color=e['farbe'], lw=1.8, ls='--',
                  label=f'$y$, Rauschen {e["label"]} '
                        rf'(${SYM_ALPHA_MIN_BLR} = {e["alpha_min"]:.3f}$, '
                        rf'$k = {e["y"].size}$)')
           for e in ergebnisse]
handles += [Line2D([], [], color='0.55', lw=1.8,
                   label=rf'$y_{{\mathrm{{rep}}}}$ ({S_ZIEHUNGEN} Ziehungen je Rauschen)')]
legende = ax.legend(handles=handles, fontsize=fontsize - 3, loc='upper left',
                    frameon=True, facecolor=WEISS, edgecolor=SCHWARZ, framealpha=1.0)
legende.get_frame().set_linewidth(0.6)

fig.tight_layout()
save_diagram(fig, 'blr_ppc_dichte_overlay.pdf', unterordner='bayes')
plt.show()

for erg in ergebnisse:
    y, y_rep = erg['y'], erg['y_rep']
    print(f'Rauschen {klartext(erg["label"])}: alpha_min_blr = {erg["alpha_min"]:.4f}, k = {y.size}, '
          f'sigma = {erg["post"]["sigma"]:.3e}')
    print(f'  Mittelwert  y = {y.mean():.6f},  y_rep = {y_rep.mean():.6f}')
    print(f'  Streuung    y = {y.std():.6e},  y_rep = {y_rep.std():.6e}')
