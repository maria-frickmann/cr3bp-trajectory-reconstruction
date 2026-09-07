import numpy as np
from scipy.signal import savgol_filter

def alpha_min_kriterium_unten(alphas_sortiert, werte_sortiert, schranke, w):
    """Definition Nebenbedingung für das Hauptkriterium"""
    alphas_sortiert = np.asarray(alphas_sortiert, dtype=np.float64)
    werte_sortiert = np.asarray(werte_sortiert, dtype=np.float64)
    for aktuelles_alpha in alphas_sortiert:
        fenster_ende = min(aktuelles_alpha + w, 1.0)
        im_fenster   = (alphas_sortiert >= aktuelles_alpha) & (alphas_sortiert <= fenster_ende)
        if np.min(werte_sortiert[im_fenster]) >= schranke:
            return float(aktuelles_alpha)
    return None

def alpha_min_form_kriterium(alphas, corr_serie, delta_C_serie, C_rek_serie, rho, w):
    """Defintion des Hauptkriteriums"""
    a_min = alpha_min_kriterium_unten(alphas, corr_serie, rho, w)
    if a_min is None:
        return None, None, None
    i = int(np.argmin(np.abs(np.asarray(alphas, float) - a_min)))
    return float(a_min), float(delta_C_serie[i]), float(C_rek_serie[i])
