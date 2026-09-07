# Pfad definieren
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Imports
import numpy as np
from i_Programmierung.Modellierung.mannigfaltigkeiten import (q1_opt, q2_opt, q3_opt, p1_opt, p2_opt, p3_opt)
from i_Programmierung.Modellierung.physik import jacobi_konstante
from i_Programmierung.Konstanten import mu, w, startwert_zufall

# Länge von q1
N = len(q1_opt)

# Berechnen der Jakobi Konstante der Referenztrajektorie
C_referenz = jacobi_konstante(q1_opt, q2_opt, q3_opt, p1_opt, p2_opt, p3_opt, mu).mean()

# In der Nähe von möglichen alpha_min Werten wird genauer gerechnet.
k_fein   = np.arange(1, 25, 2)
k_mittel = np.arange(25, 120, 4)
k_grob   = np.unique(np.linspace(120, N, 8).astype(int))
k_liste  = np.unique(np.concatenate([k_fein, k_mittel, k_grob]))
k_liste  = k_liste[(k_liste >= 1) & (k_liste <= N)]
alpha_cs_liste = k_liste / N

# Definieren von Rauschen, die berechnet werden
rauschen_liste = [0.0001, 0.001, 0.01]
ss = np.random.SeedSequence(startwert_zufall)
abgeleitete_startwerte = ss.spawn(len(rauschen_liste))