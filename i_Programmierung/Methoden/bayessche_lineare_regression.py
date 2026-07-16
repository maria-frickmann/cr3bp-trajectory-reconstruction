# Pfad definieren
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(current_dir))
ERGEBNISSE_PFAD = Path(__file__).resolve().parent / "ergebnisse_blr.pkl"

# Imports
import pickle
import numpy as np
from scipy.fft import idct
from i_Programmierung.Methoden.compressed_sensing import (X_ref, C_referenz, erzeuge_messdaten, delta_total_metrik, alpha_cs_liste, rauschen_liste, tau, GLAETTUNG, C_FLOOR)
from i_Programmierung.Methoden import compressed_sensing as sv
from i_Programmierung.Methoden.kriterium import alpha_min_form_kriterium
from i_Programmierung.Konstanten import mu, w, N_WDH, KAPPA
from i_Programmierung.Modellierung.mannigfaltigkeiten import X_ref as _X_ref_manifold

def posterior_mean(A, y, sigma_likelihood, dc_amplitude):
    """Das Bayes Modell wird definiert. """
    Nt = A.shape[1]
    scale = dc_amplitude / (np.arange(Nt) + 1.0)
    Lam = np.diag(1.0 / scale**2) + (A.T @ A) / sigma_likelihood**2
    return np.linalg.solve(Lam, (A.T @ y) / sigma_likelihood**2)

_q1_ref_np  = np.asarray(X_ref)[:, 0]
_schwingung = _q1_ref_np - _q1_ref_np.mean()

def berechne_bayes_quantil(rauschen, alpha, n_wiederholungen=N_WDH, kappa=KAPPA):
    """Wiederholt die Bayes Rekonstruktion n_wiederholungen-mal mit neuem Zufallsrauschen. """
    v, v_C, v_korr, v_C_rek = [], [], [], []

    for m in range(n_wiederholungen):
        y_list, A, idx, sig_l, eps_l = erzeuge_messdaten(rauschen, alpha, m)
        y = np.asarray(y_list[0]); A = np.asarray(A)
        sigma = float(sig_l[0]) if sig_l[0] > 0 else 1e-6
        _C_REF = float(np.max(np.abs(_X_ref_manifold[0])))
        dc = max(float(np.max(np.abs(A.T @ y))), _C_REF)
        theta  = posterior_mean(A, y, sigma, dc)
        q1_rek = idct(theta, norm='ortho')
        delta_total, C_rek, delta_C, _delta_X = delta_total_metrik(q1_rek, X_ref, C_referenz, mu)
        v.append(delta_total)
        v_C.append(delta_C)
        v_korr.append(float(np.corrcoef(q1_rek - q1_rek.mean(), _schwingung)[0, 1]))
        v_C_rek.append(C_rek)

    return (float(np.quantile(v, kappa)),            float(np.mean(v)),
            float(np.quantile(v_C, kappa)),          float(np.mean(v_C)),
            float(np.quantile(v_korr, 1.0 - kappa)), float(np.mean(v_korr)),
            float(np.mean(v_C_rek)))

def berechne_alle_blr():
    """BLR Werte über alle Rauschpegel und alle Messanteile von alpha."""
    alpha_arr = np.asarray(alpha_cs_liste)

    bayes = {r: {} for r in rauschen_liste}
    for alpha in alpha_arr:
        alpha = float(alpha)
        for r in rauschen_liste:
            bayes[r][alpha] = berechne_bayes_quantil(r, alpha)
            q, _avg, q_C, _avg_C, q_korr, _avg_korr, _C_rek_avg = bayes[r][alpha]
    return bayes, alpha_arr