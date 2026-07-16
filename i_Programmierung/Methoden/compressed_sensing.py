# Pfad definieren
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Imports
import numpy as np
import cvxpy as cp
from scipy.fft import dct, idct
from i_Programmierung.Modellierung.physik import jacobi_konstante
from i_Programmierung.Modellierung.mannigfaltigkeiten import X_ref
from i_Programmierung.Konstanten import (mu, w, startwert_zufall, RHO, TAU_FIX, N_WDH, KAPPA)
from i_Programmierung.Konfiguration import rauschen_liste, C_referenz, N, alpha_cs_liste
from i_Programmierung.Methoden.kriterium import (alpha_min_form_kriterium)

# Parameter
GLAETTUNG = True
C_FLOOR   = 3.0
tau = float(TAU_FIX)
def als_zeitreihen_matrix(X):
    """Bringt eine Trajektorie in die Matrixform (N Zeitschritte, 6 Zustandsvariablen)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] == 6 and X.shape[1] != 6:
        X = X.T
    return X

X_ref = als_zeitreihen_matrix(X_ref)
N_KOMPONENTEN = X_ref.shape[1]

def dct_basis(N):
    """Die Trajektorie wird als Signal aufgefasst, das sich aus Cosinusschwingungen zusammensetzt."""
    return dct(np.eye(N), norm='ortho', axis=0)

# DCT-Matrix und ihre Inverse bestimmen
Psi     = dct_basis(N)
Psi_inv = Psi.T

def random_sampling(N, X_ref, p, rauschen, rng_cs):
    """
    Zufällige Zeitabtastung mit Anteil p und relativem Rauschpegel.
    Formel:     y = A * x_dct + z  mit  A = R * Psi_inv  und  z ~ N(0, sigma^2 * I)
    """
    X_ref = als_zeitreihen_matrix(X_ref)  # Referenz-Signalmatrix
    k   = int(round(p * N))               # Anzahl der Datenpunkte
    k   = max(1, min(k, N))
    idx = np.sort(rng_cs.choice(N, k, replace=False)) # Indexvektor
    R   = np.eye(N)[idx]  #Messmatrix
    A   = R @ Psi_inv     #CS-Messmatrix, A = R * Psi_inv

    sigma_liste   = np.zeros(N_KOMPONENTEN) # Vektor der absoluten Rauschstandardabweichungen
    epsilon_liste = np.zeros(N_KOMPONENTEN) # Vektor der Rauschtoleranzgrenzen
    y_liste = []

    for j in range(N_KOMPONENTEN):
        """Messdaten pro Kanal j erzeugen."""
        sigma_j = rauschen * np.std(X_ref[:, j])
        sigma_liste[j]   = sigma_j
        epsilon_liste[j] = sigma_j * np.sqrt(k) if sigma_j > 0 else 1e-6
        x_dct_j = dct(X_ref[:, j], norm='ortho')
        z = rng_cs.normal(0.0, sigma_j, size=k)
        y_liste.append(A @ x_dct_j + z)
    return y_liste, A, idx, sigma_liste, epsilon_liste

def _rauschen_index(rauschen):
    """Ordnet einem Rauschpegel einen festen Index zu."""
    array   = np.asarray(rauschen_liste, dtype=np.float64)
    treffer = np.where(np.isclose(array, rauschen, rtol=1e-12, atol=0.0))[0]
    return int(treffer[0]) if len(treffer) else int(round(abs(float(rauschen)) * 1e9))

def _alpha_array_index(alpha):
    """Ordnet einem alpha Wert einen festen Index zu."""
    array   = np.asarray(alpha_cs_liste, dtype=np.float64)
    treffer = np.where(np.isclose(array, alpha, rtol=1e-9, atol=0.0))[0]
    return int(treffer[0]) if len(treffer) else int(round(abs(float(alpha)) * 1e9))

def erzeuge_messdaten(rauschen, alpha, wiederholung):
    """
    Erzeugt die Messdaten reproduzierbar aus (startwert_zufall, idx_r, idx_a, wdh).
    Formel:     A = R @ Psi_inv = Psi_inv[idx, :].
    """
    schluessel = (_rauschen_index(rauschen), _alpha_array_index(alpha), int(wiederholung))
    seed_sequenz     = np.random.SeedSequence(startwert_zufall, spawn_key=schluessel)
    zufallsgenerator = np.random.default_rng(seed_sequenz)
    y_liste, _A, idx, sigma_liste, epsilon_liste = random_sampling(
        N, X_ref, alpha, rauschen, zufallsgenerator)
    return y_liste, Psi_inv[idx, :], idx, sigma_liste, epsilon_liste

def l1_minimierung(A, y, epsilon):
    """Löst das Optimierungsproblem: min ||x||_1 unter der Nebenbedingung ||A x - y||_2 <= epsilon."""
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k, N_cols = A.shape
    x_var = cp.Variable(N_cols)
    prob  = cp.Problem(cp.Minimize(cp.norm1(x_var)),
                      [cp.norm2(A @ x_var - y) <= float(epsilon)])
    prob.solve(solver=cp.CLARABEL, verbose=False)
    if x_var.value is None:
        raise RuntimeError(f"L1-Minimierung fehlgeschlagen (Status: {prob.status}, "
                           f"k={k}, epsilon={float(epsilon):.3e})")
    return np.asarray(x_var.value, dtype=np.float64)

def norm_q1_schwingung(X_ref):
    """Bezugsgrösse von delta_X := ||q1 - mean(q1)||_2, also die Norm der Schwingung von q1."""
    q1_ref = als_zeitreihen_matrix(X_ref)[:, 0]
    return float(np.linalg.norm(q1_ref - q1_ref.mean()))

def delta_total_metrik(q1_rek, X_ref, C_ref, mu):
    """Fehlermetrik für q1:

        Formfehler:   delta_X     = ||q1_rek - q1_ref||_2 / ||q1_ref - mean(q1_ref)||_2
        Jacobifehler: delta_C     = |C_rek - C_ref|       / |C_ref|
        delta_total = delta_C + delta_X
    """

    X_ref  = als_zeitreihen_matrix(X_ref)
    q1_rek = np.asarray(q1_rek, dtype=np.float64).reshape(-1)
    q1_ref  = X_ref[:, 0]
    C_rek   = jacobi_konstante(q1_rek, X_ref[:, 1], X_ref[:, 2], X_ref[:, 3],
                               X_ref[:, 4], X_ref[:, 5], mu).mean()
    delta_C = abs(C_rek - C_ref) / abs(C_ref)
    delta_X = np.linalg.norm(q1_rek - q1_ref) / norm_q1_schwingung(X_ref)
    return float(delta_C + delta_X), float(C_rek), float(delta_C), float(delta_X)

def berechne_alpha_werte(alpha_liste, X_ref, C_ref, rauschen,tau=None, n_wiederholungen=N_WDH, kappa=KAPPA, glaettung=GLAETTUNG, c_floor=C_FLOOR):
    """Bestimme alle Werte für compressed sensing. """
    tau    = float(TAU_FIX if tau is None else tau)
    X_ref  = als_zeitreihen_matrix(X_ref)
    q1_ref = X_ref[:, 0]
    schwingung = q1_ref - q1_ref.mean()
    alphas, delta_q, delta_avg = [], [], []
    delta_C_q, delta_C_avg, korr_q, korr_avg = [], [], [], []
    C_rek_serie = []

    for alpha in alpha_liste:
        deltas, deltas_C, korrs, C_reks = [], [], [], []
        for wdh in range(n_wiederholungen):
            y_liste, A, idx, _sigma, epsilon = erzeuge_messdaten(rauschen, float(alpha), wdh)
            x_tilde = l1_minimierung(A, y_liste[0], epsilon[0])
            q1_rek  = idct(x_tilde, norm='ortho')
            delta_total, C_rek, delta_C, _delta_X = delta_total_metrik(q1_rek, X_ref, C_ref, mu)
            deltas.append(delta_total)
            deltas_C.append(delta_C)
            korrs.append(float(np.corrcoef(q1_rek - q1_rek.mean(), schwingung)[0, 1]))
            C_reks.append(C_rek)
        alphas.append(float(alpha))
        delta_avg.append(float(np.mean(deltas)))
        delta_q.append(float(np.quantile(deltas, kappa)))
        delta_C_avg.append(float(np.mean(deltas_C)))
        delta_C_q.append(float(np.quantile(deltas_C, kappa)))
        korr_avg.append(float(np.mean(korrs)))
        korr_q.append(float(np.quantile(korrs, 1.0 - kappa)))
        C_rek_serie.append(float(np.mean(C_reks)))

    alphas, delta_q, delta_avg = np.array(alphas), np.array(delta_q), np.array(delta_avg)
    delta_C_q, delta_C_avg     = np.array(delta_C_q), np.array(delta_C_avg)
    korr_q, korr_avg           = np.array(korr_q), np.array(korr_avg)
    C_rek_serie                = np.array(C_rek_serie)

    a_min, delta_C_bei_amin, C_rek_bei_amin = alpha_min_form_kriterium(
        alphas, korr_q, delta_C_q, C_rek_serie, RHO, w)

    return {
        'alpha'                 : alphas,
        'delta_total_q'         : delta_q,
        'delta_total_avg'       : delta_avg,
        'delta_C_q'             : delta_C_q,
        'delta_C_avg'           : delta_C_avg,
        'korr_q'                : korr_q,
        'korr_avg'              : korr_avg,
        'C_rek_serie'           : C_rek_serie,
        'floor_bei_alpha1'      : float(delta_q[-1]),
        'alpha_min_cs'          : a_min,
        'rho'                   : float(RHO),
        'delta_C_bei_alpha_min' : delta_C_bei_amin,
        'C_rek_bei_alpha_min'   : C_rek_bei_amin,
        'C_ref'                 : float(C_ref),
        'delta_C_floor'         : float(delta_C_q[-1]),
        'korr_floor'            : float(korr_q[-1]),
        'tau'                   : tau,
        'c_floor'               : float(c_floor),
        'glaettung'             : bool(glaettung),
    }


def berechne_cs_ergebnisse(n_wiederholungen=N_WDH, kappa=KAPPA, alpha_liste=None,
                           tau=None, rauschen_auswahl=None):
    """Ergebnisse für alle Rauschpegel und für alle alpha_min bestimmen und speichern."""
    gitter = np.asarray(alpha_cs_liste if alpha_liste is None else alpha_liste,
                        dtype=np.float64)
    ergebnisse = {}
    for r in rauschen_liste:
        if rauschen_auswahl is not None and r not in rauschen_auswahl:
            continue
        ergebnisse[float(r)] = berechne_alpha_werte(
            gitter, X_ref, C_referenz, float(r),
            tau=tau, n_wiederholungen=n_wiederholungen, kappa=kappa)
    return ergebnisse