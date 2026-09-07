import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    Lam   = 0.5 * (Lam + Lam.T)                       # exakte Symmetrie erzwingen
    mu    = np.linalg.solve(Lam, (A.T @ y) / sigma**2)
    assert np.allclose(mu, blr.posterior_mean(A, y, sigma, dc), rtol=0, atol=1e-12)

    L = np.linalg.cholesky(Lam)                       # Lam = L L^T  ->  Sigma = L^-T L^-1
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
    post   = blr_posterior(float(rauschen), float(alpha_min))
    y_rep  = posterior_predictive(post, S_ZIEHUNGEN, rng)
    fehler = (post['y'][None, :] - y_rep).mean(axis=0)
    ergebnisse.append({'farbe': farbe, 'label': label_r, 'alpha_min': float(alpha_min),
                       'post': post, 'y': post['y'], 'fehler': fehler,
                       'fehler_norm': fehler / post['sigma']})
if not ergebnisse:
    raise SystemExit('kein alpha_min_blr vorhanden')

fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=300)
style_ax(ax)

ax.axvline(0.0, color='0.35', linestyle='dashed', linewidth=1.1, zorder=1)
for erg in ergebnisse:
    ax.scatter(erg['fehler_norm'], erg['y'], s=22, color=erg['farbe'], alpha=0.75,
               edgecolors='none', zorder=3)

ax.set_xlabel(r'$\mathrm{mean}(y - y_{\mathrm{rep}})\,/\,\sigma$', fontsize=fontsize)
ax.set_ylabel(LABEL_MESSWERT, fontsize=fontsize)
grenze = max(np.max(np.abs(e['fehler_norm'])) for e in ergebnisse) * 1.15
ax.set_xlim(-grenze, grenze)
ax.set_title(f'Mittlerer Residualfehler der BLR bei {LABEL_ALPHA_MIN_BLR}'
             '\nfür alle untersuchten Rauschpegel', fontsize=fontsize)
ax.grid(True, which='both', linewidth=0.3, alpha=0.4)

handles = [Line2D([], [], color=e['farbe'], lw=0, marker='o', markersize=5,
                  label=f'Rauschen {e["label"]} '
                        rf'(${SYM_ALPHA_MIN_BLR} = {e["alpha_min"]:.3f}$, '
                        rf'$k = {e["y"].size}$)')
           for e in ergebnisse]
handles += [Line2D([], [], color='0.35', ls='dashed', lw=1.1,
                   label='kein systematischer Fehler')]
legende = ax.legend(handles=handles, fontsize=fontsize - 3, loc='lower left',
                    frameon=True, facecolor=WEISS, edgecolor=SCHWARZ, framealpha=1.0)
legende.get_frame().set_linewidth(0.6)

fig.tight_layout()
save_diagram(fig, 'blr_ppc_mittlerer_fehler.pdf', unterordner='bayes')
plt.show()

for erg in ergebnisse:
    print(f'Rauschen {klartext(erg["label"])}: alpha_min_blr = {erg["alpha_min"]:.4f}, '
          f'k = {erg["y"].size}, sigma = {erg["post"]["sigma"]:.3e}')
    print(f'  mittlerer Fehler über alle Messungen = {erg["fehler"].mean():+.4e} '
          f'({erg["fehler_norm"].mean():+.4f} sigma)')
    print(f'  grösster Betrag = {np.max(np.abs(erg["fehler"])):.4e} '
          f'({np.max(np.abs(erg["fehler_norm"])):.4f} sigma)')
