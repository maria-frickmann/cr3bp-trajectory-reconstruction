"""Gemeinsame Hilfsfunktionen der Trajektorien-Rekonstruktions-Plots (cs_trajektorie_rek_*,
blr_trajektorie_rek_*): Filtern der Rauschpegel mit vorhandenem alpha_min sowie
Achsengrenzen-Berechnung. Reine Hilfsfunktionen, keine eigenen Formeln/Berechnungen.
"""
import numpy as np


def vorhandene_rekonstruktionen(meta, rekonstruktionen, alpha_min_name, farben_rauschen, labels_rauschen):
    """Filtert (rauschen, farbe, label) auf Rauschpegel, fuer die alpha_min_XXX existiert.

    Druckt fuer uebersprungene Rauschpegel einen Hinweis (alpha_min_name z.B. 'alpha_min_cs').
    """
    vorhanden = [(float(r), farbe, label_r)
                 for r, farbe, label_r in zip(meta['rauschen_liste'], farben_rauschen, labels_rauschen)
                 if rekonstruktionen[float(r)] is not None]
    for rauschen, _f, label_r in zip(meta['rauschen_liste'], farben_rauschen, labels_rauschen):
        if rekonstruktionen[float(rauschen)] is None:
            print(f" kein {alpha_min_name} für Rauschen {label_r}")
    return vorhanden

def q1_grenzen(q1_ref, rekonstruktionen, vorhanden, rand_anteil=0.08):
    alle_q1 = np.concatenate([q1_ref] + [rekonstruktionen[r]['q1_rek'] for r, _, _ in vorhanden])
    rand = rand_anteil * (alle_q1.max() - alle_q1.min())
    return (alle_q1.min() - rand, alle_q1.max() + rand)

def achsen_limits(werte, rand_anteil=0.15):
    werte  = np.asarray(werte)
    spanne = werte.max() - werte.min()
    rand   = max(spanne * rand_anteil, 1e-4)
    return (werte.min() - rand, werte.max() + rand)
