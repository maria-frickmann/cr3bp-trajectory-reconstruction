import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from ii_Skripte_Visualisierung import berechnungen as ber
from ii_Skripte_Visualisierung.setup_plot import (farben_rauschen, labels_rauschen,
                                                  style_ax, save_diagram, fontsize,
                                                  klartext, SYM_ALPHA_MIN_BLR)
from ii_Skripte_Visualisierung.farben import WEISS, SCHWARZ
from i_Programmierung.Methoden import bayessche_lineare_regression as blr
from i_Programmierung.Modellierung.mannigfaltigkeiten import X_ref as _X_ref_manifold

sys.path.append(str(Path(__file__).resolve().parents[1]))
from setup_plot import fontsize
fontsize = fontsize

FARBE_PRIOR, FARBE_LIK, FARBE_POST = farben_rauschen[0], farben_rauschen[1], farben_rauschen[2]

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

    return {'y': y, 'A': A, 'idx': np.asarray(idx), 'sigma': sigma,
            'scale': scale, 'Lam': Lam, 'mu': mu}


def normal_normal_update(post):
    A, y, mu, Lam = post['A'], post['y'], post['mu'], post['Lam']
    n = int(np.argmax(np.abs(mu[1:]))) + 1

    tau_prior = 1.0 / post['scale'][n]**2
    tau_lik   = (A.T @ A)[n, n] / post['sigma']**2
    tau_post  = tau_prior + tau_lik

    b       = (A.T @ y) / post['sigma']**2
    rest    = Lam[n, :] @ mu - Lam[n, n] * mu[n]
    mu_lik  = (b[n] - rest) / tau_lik
    mu_post = (tau_prior * 0.0 + tau_lik * mu_lik) / tau_post
    assert np.isclose(mu_post, mu[n], rtol=0, atol=1e-12)

    return {'n': n, 'tau_prior': tau_prior, 'tau_lik': tau_lik, 'tau_post': tau_post,
            'mu_lik': mu_lik, 'mu_post': mu_post,
            'sd_prior': 1 / np.sqrt(tau_prior), 'sd_lik': 1 / np.sqrt(tau_lik),
            'sd_post': 1 / np.sqrt(tau_post)}

blr_daten = ber.blr_kurven()
meta      = blr_daten['meta']

ergebnisse = []
for rauschen, _farbe, label_r in zip(meta['rauschen_liste'], farben_rauschen, labels_rauschen):
    alpha_min = blr_daten['kurven'][float(rauschen)]['alpha_min_blr']
    if alpha_min is None:
        print(f' kein alpha_min_blr für Rauschen {klartext(label_r)}')
        continue
    post = blr_posterior(float(rauschen), float(alpha_min))
    ergebnisse.append({'rauschen': float(rauschen), 'label': label_r,
                       'alpha_min': float(alpha_min), 'post': post,
                       'upd': normal_normal_update(post)})
if not ergebnisse:
    raise SystemExit('kein alpha_min_blr vorhanden')

for erg in ergebnisse:
    upd, post = erg['upd'], erg['post']

    def dichten(gitter):
        return (stats.norm.pdf(gitter, 0.0, upd['sd_prior']),
                stats.norm.pdf(gitter, upd['mu_lik'], upd['sd_lik']),
                stats.norm.pdf(gitter, upd['mu_post'], upd['sd_post']))

    fig, (ax_prior, ax_zoom) = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=300)
    style_ax(ax_prior)
    style_ax(ax_zoom)

    gitter_prior = np.linspace(-4 * upd['sd_prior'], 4 * upd['sd_prior'], 4000)
    d_prior, _d_lik, _d_post = dichten(gitter_prior)
    faktor_prior = d_prior.max()

    ax_prior.fill_between(gitter_prior, d_prior / faktor_prior, color=FARBE_PRIOR,
                          alpha=0.45, linewidth=0, zorder=2)
    ax_prior.plot(gitter_prior, d_prior / faktor_prior, color=FARBE_PRIOR, lw=1.5, zorder=3)
    ax_prior.axvline(upd['mu_lik'],  color=FARBE_LIK,  lw=3.4, zorder=4)
    ax_prior.axvline(upd['mu_post'], color=FARBE_POST, lw=1.6, ls='--', zorder=5)

    ax_prior.set_ylim(0, 1.20)
    ax_prior.set_xlim(gitter_prior[0], gitter_prior[-1])
    ax_prior.set_xlabel(rf'$\theta_{{{upd["n"]}}}$', fontsize=fontsize)
    ax_prior.set_ylabel('Dichte (normiert)', fontsize=fontsize)
    ax_prior.grid(True, which='both', linewidth=0.3, alpha=0.4)

    gitter_zoom = np.linspace(upd['mu_post'] - 5 * upd['sd_lik'],
                              upd['mu_post'] + 5 * upd['sd_lik'], 4000)
    d_prior_z, d_lik_z, d_post_z = dichten(gitter_zoom)
    faktor_post = d_post_z.max()

    ax_zoom.fill_between(gitter_zoom, d_lik_z / faktor_post, color=FARBE_LIK,
                         alpha=0.40, linewidth=0, zorder=2)
    ax_zoom.plot(gitter_zoom, d_lik_z / faktor_post,   color=FARBE_LIK,  lw=1.8, zorder=3)
    ax_zoom.plot(gitter_zoom, d_post_z / faktor_post,  color=FARBE_POST, lw=1.8, ls='--', zorder=4)
    ax_zoom.plot(gitter_zoom, d_prior_z / faktor_post, color=FARBE_PRIOR, lw=1.8, zorder=5)

    ax_zoom.set_ylim(0, 1.15)
    ax_zoom.set_xlim(gitter_zoom[0], gitter_zoom[-1])
    ax_zoom.set_xlabel(rf'$\theta_{{{upd["n"]}}}$', fontsize=fontsize)
    ax_zoom.set_ylabel('Dichte (normiert)', fontsize=fontsize)
    ax_zoom.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
    ax_zoom.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax_zoom.grid(True, which='both', linewidth=0.3, alpha=0.4)

    handles = [Line2D([], [], color=FARBE_PRIOR, lw=1.8,
                      label=rf'Prior $\mathcal{{N}}(0,\ {upd["sd_prior"]:.2f}^2)$'),
               Line2D([], [], color=FARBE_LIK, lw=1.8,
                      label=rf'Likelihood $\mathrm{{sd}} = {upd["sd_lik"]:.2e}$'),
               Line2D([], [], color=FARBE_POST, lw=1.8, ls='--',
                      label=rf'Posterior $\mathrm{{sd}} = {upd["sd_post"]:.2e}$')]

    fig.suptitle(rf'Bayes Update des DCT Koeffizienten $\theta_{{{upd["n"]}}}$ bei '
                 rf'${SYM_ALPHA_MIN_BLR} = {erg["alpha_min"]:.3f}$ '
                 rf'(Rauschen {erg["label"]}, $k = {post["idx"].size}$)', fontsize=fontsize)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    legende = fig.legend(handles=handles, loc='lower center', ncol=3,
                         fontsize=fontsize - 2, frameon=True, facecolor=WEISS,
                         edgecolor=SCHWARZ, framealpha=1.0)
    legende.get_frame().set_linewidth(0.6)
    save_diagram(fig, f'blr_prior_likelihood_posterior_{erg["rauschen"]:g}.pdf',
                 unterordner='bayes')
    plt.show()

    print(f'Rauschen {klartext(erg["label"])}: alpha_min_blr = {erg["alpha_min"]:.4f}, '
          f'k = {post["idx"].size}, sigma = {post["sigma"]:.3e}, n = {upd["n"]}')
    print(f'  Prior     : mu = 0,            sd = {upd["sd_prior"]:.4e}')
    print(f'  Likelihood: mu = {upd["mu_lik"]:+.6f}, sd = {upd["sd_lik"]:.4e}')
    print(f'  Posterior : mu = {upd["mu_post"]:+.6f}, sd = {upd["sd_post"]:.4e}')
    print(f'  Shrinkage tau_prior/tau_post = {upd["tau_prior"] / upd["tau_post"]:.3e}')
