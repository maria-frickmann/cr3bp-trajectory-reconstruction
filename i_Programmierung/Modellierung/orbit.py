# Pfad definieren
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(current_dir))

# Imports
import numpy as np
from scipy.integrate import solve_ivp
from i_Programmierung.Konstanten import mu, Radius, L_fundamental
from i_Programmierung.Modellierung.physik import bewegungsgleichungen

"""
Für die Bestimmung des Halo-Orbits werden Anfangswerte aus der NASA-Datenbank entnommen. 
Quelle: https://ssd.jpl.nasa.gov/tools/periodic_orbits.html
Constraints:
    - System:          Sun-Earth
    - Orbit family:    Halo-Northern
    - Libration point: L2
    - Jacobi constant: Any
    - Period: 	       Any
    - Stability index: Any
    - Id:              1212
"""

# Entnehmen wesentlicher Daten
q1_halo_nasa =  1.0111429782193371E+0
q2_halo_nasa =  2.7011837844516203E-23
q3_halo_nasa =  3.4701931834359183E-3	
v1_halo_nasa =  3.0416434297969863E-17     
v2_halo_nasa = -1.0492956284545975E-2
v3_halo_nasa = -3.6887981429649052E-16
periode_halo_nasa       = 3.0903666478803373E+0

# Koordinatentransformation von Kartesisch zu Hamiltonisch
q1_0 = float(q1_halo_nasa)
q2_0 = float(q2_halo_nasa)
q3_0 = float(q3_halo_nasa)
p1_0 = float(v1_halo_nasa) - float(q2_halo_nasa)
p2_0 = float(v2_halo_nasa) + float(q1_halo_nasa)
p3_0 = float(v3_halo_nasa)

# Anfangswerte als Zustandsvektor definieren
state_0 = [q1_0, q2_0, q3_0, p1_0, p2_0, p3_0]

t_eval = np.linspace(0, periode_halo_nasa, 90) 

# Das Anfangswertproblem der Bewegungsgleichungen wird numerisch gelöst
sol = solve_ivp(bewegungsgleichungen, 
    [0, periode_halo_nasa], 
    state_0, method='DOP853', 
    args=(mu,), t_eval = t_eval,
    rtol=1e-7, atol=1e-7)