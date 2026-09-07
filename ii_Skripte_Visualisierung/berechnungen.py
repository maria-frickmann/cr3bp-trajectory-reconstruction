import os
import sys
from pathlib import Path
_O2_WURZEL = Path(__file__).resolve().parents[1]
if str(_O2_WURZEL) not in sys.path:
    sys.path.append(str(_O2_WURZEL))
import pickle
from joblib import Parallel, delayed
import numpy as np
from scipy.fft import idct
from i_Programmierung.Konstanten import w, mu, N_WDH, KAPPA
from i_Programmierung.Konfiguration import rauschen_liste, alpha_cs_liste
from i_Programmierung.Methoden.kriterium import alpha_min_form_kriterium
from i_Programmierung.Methoden import bayessche_lineare_regression as blr
from i_Programmierung.Methoden import compressed_sensing as cs

N_JOBS = max(1, (os.cpu_count() or 4) // 2)

def _cs_punkt(rauschen, alpha, n_wiederholungen=N_WDH):
    """Ein einzelner Gitterpunkt (Rauschen, alpha) der CS-Pipeline: aus alpha*N verrauschten Messungen wird via l1-Minimierung der dünnbesetzte DCT-Köffizientenvektor x_tilde geschätzt, den die inverse DCT zurück in die q1-Bahn transformiert. Über n_wiederholungen werden Fehlermass, Jacobi-Konstante und Korrelation zur Referenzschwingung gesammelt, um Streuung statt Einzelfall zu sehen."""
    q1_ref = np.asarray(cs.X_ref)[:, 0]
    schwingung = q1_ref - q1_ref.mean()

    deltas_total, deltas_C, korr_liste, C_rek_liste = [], [], [], []
    q1_rek_erste, idx_erste = None, None
    for wdh in range(n_wiederholungen):
        y_liste, A, idx, _sigma, epsilon = cs.erzeuge_messdaten(rauschen, float(alpha), wdh)
        x_tilde = cs.l1_minimierung(A, y_liste[0], epsilon[0])
        q1_rek = idct(x_tilde, norm='ortho')
        d_tot, C_rek, d_C, _d_X = cs.delta_total_metrik(q1_rek, cs.X_ref, cs.C_referenz, mu)
        deltas_total.append(d_tot)
        deltas_C.append(d_C)
        korr_liste.append(float(np.corrcoef(q1_rek - q1_rek.mean(), schwingung)[0, 1]))
        C_rek_liste.append(C_rek)
        if wdh == 0:
            q1_rek_erste, idx_erste = q1_rek, idx
    return {
        'delta_total': deltas_total,
        'delta_C': deltas_C,
        'korr': korr_liste,
        'C_rek': C_rek_liste,
        'q1_rek_erste': q1_rek_erste,
        'idx_erste': idx_erste,}


def _blr_punkt(rauschen, alpha, n_wiederholungen=1):
    """Dasselbe wie _cs_punkt, aber mit bayesscher linearer Regression: statt der l1-Nebenbedingung liefert der Posterior-Mittelwert die DCT-Köffizienten in geschlossener Form. Weil die Lösung bei gegebenen Daten deterministisch ist, genügt hier eine kleinere Zahl an Wiederholungen. Die Metriken werden identisch zur CS-Seite berechnet, damit beide vergleichbar sind."""
    q1_ref = np.asarray(blr.X_ref)[:, 0]
    schwingung = q1_ref - q1_ref.mean()

    deltas_total, deltas_C, korr_liste, C_rek_liste = [], [], [], []
    q1_rek_erste, idx_erste = None, None
    for wdh in range(n_wiederholungen):
        y_liste, A, idx, sigma_liste, epsilon_liste = blr.erzeuge_messdaten(rauschen, float(alpha), wdh)
        y = np.asarray(y_liste[0]); A = np.asarray(A)
        sigma = float(sigma_liste[0]) if sigma_liste[0] > 0 else 1e-6
        dc = max(float(np.max(np.abs(A.T @ y))), 1.0)
        theta = blr.posterior_mean(A, y, sigma, dc)
        q1_rek = idct(theta, norm='ortho')
        d_tot, C_rek, d_C, _d_X = blr.delta_total_metrik(q1_rek, blr.X_ref, blr.C_referenz, mu)
        deltas_total.append(d_tot)
        deltas_C.append(d_C)
        korr_liste.append(float(np.corrcoef(q1_rek - q1_rek.mean(), schwingung)[0, 1]))
        C_rek_liste.append(C_rek)
        if wdh == 0:
            q1_rek_erste, idx_erste = q1_rek, idx
    return {
        'delta_total': deltas_total,
        'delta_C': deltas_C,
        'korr': korr_liste,
        'C_rek': C_rek_liste,
        'q1_rek_erste': q1_rek_erste,
        'idx_erste': idx_erste,}

def _pipeline_meta():
    """Sammelt alle fixen Rahmenparameter eines Laufs (Signallänge N, Zeitschritt tau, Toleranz rho, Gewicht w, alpha-Gitter, Rauschstufen, Referenzbahn) an einer Stelle. So hängen CS- und BLR-Auswertung nachweislich am selben Gitter und sind direkt vergleichbar."""
    return {
        'N': cs.N,
        'tau': float(cs.tau),
        'rho': float(cs.RHO),
        'w': float(w),
        'alpha_gitter': np.asarray(alpha_cs_liste, dtype=np.float64),
        'rauschen_liste': list(rauschen_liste),
        'X_ref': np.asarray(cs.X_ref, dtype=np.float64),}

def _form_kriterium(alpha, delta_C_serie, korr_serie, C_rek_serie, meta):
    """Bestimmt entlang des alpha-Gitters das kleinste alpha, ab dem die Rekonstruktion die Formbedingung erfüllt: Korrelation und Abweichung der Jacobi-Konstante müssen gemeinsam (gewichtet mit w) unterhalb bzw. oberhalb der Toleranz rho liegen. Dieses alpha_min ist die eigentliche Zielgrösse: die minimale Abtastrate für eine physikalisch noch gültige Bahn."""
    a_min, delta_C_bei_amin, C_rek_bei_amin = alpha_min_form_kriterium(
        alpha, korr_serie, delta_C_serie, C_rek_serie, cs.RHO, meta['w'])
    return {
        'alpha_min_form': a_min,
        'rho': float(cs.RHO),
        'delta_C_bei_alpha_min': delta_C_bei_amin,
        'C_rek_bei_alpha_min': C_rek_bei_amin,}

def _cs_sweep(rauschen_werte, alpha_gitter, n_wiederholungen, kappa, n_jobs):
    """Rechnet das volle Kreuzprodukt Rauschen x alpha - wenn möglich parallel, sonst als Fallback seriell - und faltet die Wiederholungen pro Punkt zu einer Kurve zusammen. Fehlermasse werden über das kappa-Quantil, die Korrelation über das (1-kappa)-Quantil aggregiert."""
    paare = [(float(r), float(a)) for r in rauschen_werte for a in alpha_gitter]

    ergebnisse = None
    if n_jobs > 1:
        try:
            ergebnisse = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_cs_punkt)(r, a, n_wiederholungen) for r, a in paare)
        except Exception:
            pass
    if ergebnisse is None:
        ergebnisse = [_cs_punkt(r, a, n_wiederholungen) for r, a in paare]
    kurven = {}
    n_alpha = len(alpha_gitter)
    for i, rauschen in enumerate(rauschen_werte):
        block = ergebnisse[i * n_alpha:(i + 1) * n_alpha]
        kurven[float(rauschen)] = {
            'alpha': np.asarray(alpha_gitter, dtype=np.float64),
            'delta_total_q': np.array([np.quantile(p['delta_total'], kappa) for p in block]),
            'delta_C_q': np.array([np.quantile(p['delta_C'], kappa) for p in block]),
            'korr_q': np.array([np.quantile(p['korr'], 1.0 - kappa) for p in block]),
            'korr_avg': np.array([np.mean(p['korr']) for p in block]),
            'C_rek_avg': np.array([np.mean(p['C_rek']) for p in block]),
        }
    return kurven

def cs_kurven(n_wiederholungen=N_WDH, kappa=KAPPA, n_jobs=N_JOBS):
    """Öffentlicher Einstiegspunkt der CS-Auswertung: führt den Sweep aus und hängt an jede Rauschkurve das zugehörige alpha_min aus dem Formkriterium. Ergebnis ist pro Rauschstufe eine Fehler-über-alpha-Kurve samt Schwellwert."""
    meta = _pipeline_meta()
    kurven = _cs_sweep(meta['rauschen_liste'], meta['alpha_gitter'],
                       n_wiederholungen, kappa, n_jobs)
    for rauschen, daten in kurven.items():
        daten.update(_form_kriterium(daten['alpha'], daten['delta_C_q'], daten['korr_avg'],
                                     daten['C_rek_avg'], meta))
        daten['alpha_min_cs'] = daten['alpha_min_form']
    meta.update(n_wiederholungen=n_wiederholungen, kappa=kappa)
    return {'meta': meta, 'kurven': kurven}

def blr_kurven():
    """Gegenstück zu cs_kurven für die BLR-Seite: die Rohwerte kommen fertig aus blr.berechne_alle_blr() und werden hier nur in dasselbe Kurvenformat umgepackt. Anschliessend läuft dasselbe Formkriterium darüber, damit alpha_min_blr und alpha_min_cs dieselbe Bedeutung haben."""
    meta = _pipeline_meta()
    alpha_soll = meta['alpha_gitter']
    blr_ergebnisse, alpha_gitter = blr.berechne_alle_blr()
    kurven = {}
    for rauschen in meta['rauschen_liste']:
        delta_total_q = np.array([blr_ergebnisse[rauschen][float(a)][0] for a in alpha_gitter])
        delta_C_q = np.array([blr_ergebnisse[rauschen][float(a)][2] for a in alpha_gitter])
        korr_q = np.array([blr_ergebnisse[rauschen][float(a)][4] for a in alpha_gitter])
        korr_avg = np.array([blr_ergebnisse[rauschen][float(a)][5] for a in alpha_gitter])
        C_rek_avg = np.array([blr_ergebnisse[rauschen][float(a)][6] for a in alpha_gitter])
        kurven[float(rauschen)] = {
            'alpha': np.asarray(alpha_gitter, dtype=np.float64),
            'delta_total_q': delta_total_q,
            'delta_C_q': delta_C_q,
            'korr_q': korr_q,
            'korr_avg': korr_avg,
            'C_rek_avg': C_rek_avg,
        }
        kurven[float(rauschen)].update(_form_kriterium(alpha_gitter, delta_C_q, korr_avg,
                                                       C_rek_avg, meta))
        kurven[float(rauschen)]['alpha_min_blr'] = kurven[float(rauschen)]['alpha_min_form']
    meta.update(n_wiederholungen=blr.N_WDH, kappa=blr.KAPPA)
    return {'meta': meta, 'kurven': kurven}

def trajektorien_rekonstruktion():
    """Rechnet pro Rauschstufe genau eine CS-Rekonstruktion nach, nämlich bei dem zuvor bestimmten alpha_min, und gibt die Bahn q1_rek samt Messindizes zurück."""
    basis = cs_kurven()
    meta = basis['meta']
    rekonstruktionen = {}
    for rauschen in meta['rauschen_liste']:
        alpha_min = basis['kurven'][float(rauschen)]['alpha_min_cs']
        if alpha_min is None:
            rekonstruktionen[float(rauschen)] = None
            continue
        punkt = _cs_punkt(float(rauschen), float(alpha_min), n_wiederholungen=1)
        rekonstruktionen[float(rauschen)] = {
            'alpha_min_cs': float(alpha_min),
            'q1_rek': np.asarray(punkt['q1_rek_erste']),
            'idx': np.asarray(punkt['idx_erste']),
            'delta_total': float(punkt['delta_total'][0]),
        }
    return {'meta': meta, 'rekonstruktionen': rekonstruktionen}

def blr_trajektorien_rekonstruktion():
    """Identisch zu trajektorien_rekonstruktion, nur mit alpha_min_blr und _blr_punkt."""
    basis = blr_kurven()
    meta = basis['meta']
    rekonstruktionen = {}
    for rauschen in meta['rauschen_liste']:
        alpha_min = basis['kurven'][float(rauschen)]['alpha_min_blr']
        if alpha_min is None:
            rekonstruktionen[float(rauschen)] = None
            continue
        punkt = _blr_punkt(float(rauschen), float(alpha_min), n_wiederholungen=1)
        rekonstruktionen[float(rauschen)] = {
            'alpha_min_blr': float(alpha_min),
            'q1_rek': np.asarray(punkt['q1_rek_erste']),
            'idx': np.asarray(punkt['idx_erste']),
            'delta_total': float(punkt['delta_total'][0]),
        }
    return {'meta': meta, 'rekonstruktionen': rekonstruktionen}

def alpha_min_uebersicht():
    """Stellt die beiden Schwellwerte alpha_min_cs und alpha_min_blr je Rauschstufe nebeneinander."""
    cs_basis = cs_kurven()
    blr_basis = blr_kurven()
    uebersicht = {}
    for rauschen in cs_basis['meta']['rauschen_liste']:
        rauschen = float(rauschen)
        uebersicht[rauschen] = {
            'alpha_min_cs': cs_basis['kurven'][rauschen]['alpha_min_cs'],
            'alpha_min_blr': blr_basis['kurven'][rauschen]['alpha_min_blr'],
        }
    return {'meta': cs_basis['meta'], 'uebersicht': uebersicht}