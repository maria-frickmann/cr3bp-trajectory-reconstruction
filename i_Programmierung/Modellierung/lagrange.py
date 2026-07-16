# Pfad definieren
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(current_dir))

# Imports
import numpy as np
from scipy.optimize import fsolve
from i_Programmierung.Konstanten import mu
"""
Die Lagrangepunkte L1-L5 werden durch die Lösung von Polynomgleichungen 5. Grades bestimmt.
"""

# Definieren der Funktion des Polynom 5. Grades für L1
def polynom_l1(r, mu):
    return (r**5 - (3-mu)*r**4 + (3-2*mu)*r**3 - mu*r**2 + 2*mu*r - mu)

# Definieren der Funktion des Polynom 5. Grades für L2
def polynom_l2(r, mu):
    return (r**5 + (3-mu)*r**4 + (3-2*mu)*r**3 - mu*r**2 - 2*mu*r - mu)

# Definieren der Funktion des Polynom 5. Grades für L3
def polynom_l3(r, mu):
    return (r**5 + (2+mu)*r**4 + (1+2*mu)*r**3 - (1-mu)*r**2 - 2*(1-mu)*r - (1-mu))

# Startwerte definieren für die Nullstellensuche
sw_l1 = 1-mu
sw_l2 = 1
sw_l3 = -mu

# Berechnung Nullstelle von den Lagrangepunkten, indem das Polynom gelöst wird mit Angabe des Startwertes und dem Massenverhältnis mu
root_l1 = fsolve(polynom_l1, sw_l1, args=(mu))
root_l2 = fsolve(polynom_l2, sw_l2, args=(mu))
root_l3 = fsolve(polynom_l3, sw_l3, args=(mu))

# Koordinaten der Lagrange-Punkte L1-L5 zusammensetzen
L1 = np.array([(1 - mu) - root_l1[0], 0.0, 0.0])
L2 = np.array([(1 - mu) + root_l2[0], 0.0, 0.0])
L3 = np.array([(1 - mu) + root_l3[0], 0.0, 0.0])
L4 = np.array([0.5 - mu,  np.sqrt(3)/2, 0.0])
L5 = np.array([0.5 - mu, -np.sqrt(3)/2, 0.0])

# Eintragen der Lagrange-Punkt Koordinaten in eine Liste
l_punkte = [L1, L2, L3, L4, L5]