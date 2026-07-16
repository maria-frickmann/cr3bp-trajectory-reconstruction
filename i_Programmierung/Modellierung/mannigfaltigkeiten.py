# Pfad definieren
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(current_dir))

# Imports
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from scipy.signal import find_peaks
from i_Programmierung.Modellierung.physik import cr3bp_hamilton
from i_Programmierung.Modellierung.orbit import state_0, periode_halo_nasa, t_eval, sol
from i_Programmierung.Konstanten import mu, Radius, L_fundamental, EIGENWERT_TOLERANZ

"""
Um die Mannigfaltigkeiten zu finden, wird zunächst um den bereits berechneten Halo-Orbit linearisiert. Anschliessend werden deren Eigenwerte und Eigenvektoren bestimmt und voneinander sortiert. Eigenwerte grösser null liefern instabile Mannigfaltigkeiten und führen von L2 zur Erde. Das Gegenteil für stabile Mannigfaltigkeiten. Im zweiten Teil wird die gesuchte Trajektorie der stabilen Mannigfaltigkeiten entnommen. Dies folgt, da er der erste kleinste Abstand zur Erde ist. 
"""

_q1, _q2, _q3, _mu = sp.symbols('q1 q2 q3 mu', real=True)
_r_s = sp.sqrt((_q1 + _mu)**2 + _q2**2 + _q3**2)
_r_p = sp.sqrt((_q1 - (1 - _mu))**2 + _q2**2 + _q3**2)
_V   = -(1 - _mu) / _r_s - _mu / _r_p - sp.Rational(1, 2) * (_q1**2 + _q2**2 + _q3**2) 

# Ableitungen symbolisch berechnen und in Numpy-Funktionen umwandeln
_V11_num = sp.lambdify((_q1, _q2, _q3, _mu), sp.diff(_V, _q1, 2),   'numpy')
_V22_num = sp.lambdify((_q1, _q2, _q3, _mu), sp.diff(_V, _q2, 2),   'numpy')
_V33_num = sp.lambdify((_q1, _q2, _q3, _mu), sp.diff(_V, _q3, 2),   'numpy')
_V12_num = sp.lambdify((_q1, _q2, _q3, _mu), sp.diff(_V, _q1, _q2), 'numpy')
_V13_num = sp.lambdify((_q1, _q2, _q3, _mu), sp.diff(_V, _q1, _q3), 'numpy')
_V23_num = sp.lambdify((_q1, _q2, _q3, _mu), sp.diff(_V, _q2, _q3), 'numpy')

def jacobian(state, mu):
    """Jacobi-Matrix der linearisierten Bewegungsgleichungen am gegebenen Zustand, gebildet aus den zweiten Ableitungen des Effektiven Potentials V. """

    # Zustandsvariablen definieren
    q1, q2, q3, p1, p2, p3 = state
    
    # Partiellen Ableitungen vorbereiten
    V11 = float(_V11_num(q1, q2, q3, mu))
    V22 = float(_V22_num(q1, q2, q3, mu))
    V33 = float(_V33_num(q1, q2, q3, mu))
    V12 = float(_V12_num(q1, q2, q3, mu))
    V13 = float(_V13_num(q1, q2, q3, mu))
    V23 = float(_V23_num(q1, q2, q3, mu))

    # Jacobi-Matrix definieren    
    return np.array([
        [ 0,    1,    0,   1,  0,  0],
        [-1,    0,    0,   0,  1,  0],
        [ 0,    0,    0,   0,  0,  1],
        [-V11, -V12, -V13,  0,  1,  0],
        [-V12, -V22, -V23, -1,  0,  0],
        [-V13, -V23, -V33,  0,  0,  0],
    ])


def monodromy(t, y_kombiniert, mu):
    """
    Rechte Seite für die gekoppelte Integration von Zustand und Monodromiematrix Phi: 
    d_state = Bewegungsgleichungen, d_Phi = Df(state) @ Phi.
    """

    state = y_kombiniert[:6]
    Phi = y_kombiniert[6:].reshape((6, 6))
    
    # Bewegungsgleichungen der Zustandsvektoren
    d_state = cr3bp_hamilton(t, state, mu)
    
    # Ableitung der Monodromiematrix
    Df = jacobian(state, mu)
    d_Phi = Df @ Phi
    
    return np.concatenate([d_state, d_Phi.flatten()])

# Entnehmen des Anfangszustands des Zustandsvektors und der Identitätsmatrix für Phi
phi_0 = np.eye(6).flatten()
y_start = np.concatenate([state_0, phi_0])

# Integrieren des Halo-Orbits bzw. der Monodromy Matrix innerhalb einer Periode 
sol_monodromy = solve_ivp(
    monodromy, 
    [0, periode_halo_nasa], 
    y_start, 
    args=(mu,), 
    method='DOP853', 
    t_eval = t_eval,
    rtol=1e-12, atol=1e-12)

# Die Monodromiematrix am Ende der Periode T
M = sol_monodromy.y[6:, -1].reshape((6, 6))

def separierung_eigenwerte (M):
    """Die Monodromiematrix liefert am Ende der Periode T unterschiedliche Eigenwerte und -vektoren. Diese werden voneinander getrennt, um die Mannigfaltigkeiten voneinander zu trennen. """
    # Speichern der stabilen und instabilen Eigenwerte und -vektoren als leere Liste
    ew_stabil   = []
    ew_instabil = []

    ev_stabil   = []
    ev_instabil = []

    # Berechnung der Eigenwerte und Eigenvektoren
    eigenvalues, eigenvectors  = eig(M)

    # stabile und instabile Eigenwerte voneinander separieren
    for i in range (len(eigenvalues)):

        # Speichern aller reellen Eigenwerte
        ew_betrag   = np.abs(eigenvalues[i])
        ew_real     = np.real(eigenvalues[i])

        # Wenn |Eigenwert| kleiner als 1 ist, stabilen Eigenwert in der Liste anhängen
        if ew_betrag < 1.0 - EIGENWERT_TOLERANZ:
            ew_stabil.append(ew_real)
            ev_stabil.append(np.real(eigenvectors[:, i]))

        # Wenn |Eigenwert| groesser als 1 ist, instabilen Eigenwert in der Liste anhängen
        elif ew_betrag > 1.0 + EIGENWERT_TOLERANZ:
            ew_instabil.append(ew_real)
            ev_instabil.append(np.real(eigenvectors[:, i]))

    return ew_stabil, ew_instabil, ev_stabil, ev_instabil

# Separierte Mannigfaltigkeiten der Monodromy-Matrix als "ergebnisse" speichern
ergebnisse = separierung_eigenwerte(M)  

# Stabile und instabile Trajektorie (der Mannigfaltigkeit) als leere Liste speichern
tj_stabil   = []
tj_instabil = []

# Anstupsgeschwindigkeit
epsilon = 1e-6

# maximale Zeiteinheit für alle Mannigfaltigkeiten
t_max   = 10

if ergebnisse:
    """
    Zunächst werden die Mannigkeitigkeiten voneinander getrennt. Anschliessend werden die Mannigfaltigkeiten numerisch berechnet. 
    """
    # Falls zu den Ergebnissen ew_stabil, ew_instabil, ev_stabil, ev_instabil gehören...
    ew_stabil, ew_instabil, ev_stabil, ev_instabil = ergebnisse 

    for vorzeichen in [1, -1]:
        # ... wird nach deren positive und negative Vorzeichen getestet
        for j in range (sol.y.shape[1]):

            # Alle Startpunkte berechnen, die zum Orbit gehören
            m0_orbit = sol.y[:, j]
            Phi_j = sol_monodromy.y[6:, j].reshape((6, 6))

            for v_opt in ev_stabil:
                """ Berechnung der gesamten zeitabhängigen stabilen Mannigfaltigkeit. Bei der numerischen Lösung werden die Anfangswerte eingesetzt und rückwerts integriert. """
                v_lokal  = Phi_j @ v_opt
                v_lokal = v_lokal / np.linalg.norm(v_lokal)
                m0_pert = m0_orbit + vorzeichen * epsilon * v_lokal
                sol_traj_s = solve_ivp(
                    cr3bp_hamilton,                         
                    [t_max, 0],
                    m0_pert, args=(mu,),
                    t_eval=np.linspace(t_max, 0, 600),
                    rtol=1e-12, atol=1e-14,
                    method='DOP853'
                )
                tj_stabil.append(sol_traj_s)

            for v_opt in ev_instabil:
                """ Berechnung der instabilen Mannigfaltigkeit """
                v_lokal  = Phi_j @ v_opt
                v_lokal = v_lokal / np.linalg.norm(v_lokal)
                m0_pert = m0_orbit + vorzeichen * epsilon * v_lokal
                sol_traj_i = solve_ivp(
                    cr3bp_hamilton,     # vorwärts
                    [0, t_max],
                    m0_pert,
                    args=(mu,),
                    t_eval=np.linspace(0, t_max, 600),
                    rtol=1e-12,
                    atol=1e-14,
                    method='DOP853'
                )
                tj_instabil.append(sol_traj_i)


# Mögliche Trajektorien als leere Liste speichern.
moegliche_trajektorien = []

for trajektorie in tj_stabil:
    """ für alle stabilen Mannigfaltigkeiten wird der Abstand der Sonde zur Erde nach der Zeit t bestimmt. Anschliessend wird für jede Mannigfaltigkeit der erste kleinste Peak gesucht und in einer Liste gespeichert."""

    q1_t = trajektorie.y[0]
    q2_t = trajektorie.y[1]
    q3_t = trajektorie.y[2]

    # Abstand der Sonde zur Erde
    abstand = np.sqrt((q1_t - (1-mu))**2 + q2_t**2 + q3_t**2)

    # Prominenz wird definiert, um Oszillationen nahe des Halo-Orbits auszufiltern.
    prominenz = 0.3 * (abstand.max() - abstand.min())

    minima_idx, _ = find_peaks(-abstand, prominence=prominenz)
    for i in minima_idx:
        """Suche nach Peaks in den möglichen Trajektorien"""

        if abstand[i] >= Radius:
            """Abstand zur Erde soll grösser sein als der Erdradius"""

            moegliche_trajektorien.append((abstand[i], trajektorie, i))
            break

if moegliche_trajektorien:
    """Die optimale Trajektorie wird herausgesucht, in dem der erste kleinste Abstand aller möglichen Trajektorien. """

    # Optimale Trajektorie aus den möglichen Trajektorien auswählen 
    trajektorie_optimal = min(moegliche_trajektorien, key=lambda x: x[0])
    trajektorie_obj     = trajektorie_optimal[1]
    idx_schnitt         = trajektorie_optimal[2]

    # Optimierte Anfangswerte der Zustandsvektoren definieren
    q1_opt = trajektorie_obj.y[0]
    q1_opt = q1_opt[:idx_schnitt+1][::-1]

    q2_opt = trajektorie_obj.y[1]
    q2_opt = q2_opt[:idx_schnitt+1][::-1]
 
    q3_opt = trajektorie_obj.y[2]
    q3_opt = q3_opt[:idx_schnitt+1][::-1]
 
    p1_opt = trajektorie_obj.y[3]
    p1_opt = p1_opt[:idx_schnitt+1][::-1]
 
    p2_opt = trajektorie_obj.y[4]
    p2_opt = p2_opt[:idx_schnitt+1][::-1]

    p3_opt = trajektorie_obj.y[5]
    p3_opt = p3_opt[:idx_schnitt+1][::-1]

# Kleinsten Abstand zur Erde von der optimierten Trajektorie berechnen
abstand_opt = np.sqrt((q1_opt - (1-mu))**2 + q2_opt**2 + q3_opt**2)

idx_min     = np.argmin(abstand_opt)
trajektorie_obj = trajektorie_optimal[1]

# Zeit, wie lange die Traj
t_opt       = trajektorie_obj.t[idx_min] 
state_at_topt = trajektorie_obj.y[:, idx_min]

# Numerisches Lösen der Trajektorie
trajektorie_optimal_sol = solve_ivp(cr3bp_hamilton,
    [t_opt, 0],
    state_at_topt,
    args=(mu,),
    t_eval=np.linspace(t_opt, 0, 600),
    rtol=1e-12,
    atol=1e-14,
    method='DOP853')

# Als Zwischenresultat die optimierten Zustandsvariablen definieren
q1_o = trajektorie_optimal_sol.y[0]
q2_o = trajektorie_optimal_sol.y[1]
q3_o = trajektorie_optimal_sol.y[2]
p1_o = trajektorie_optimal_sol.y[3]
p2_o = trajektorie_optimal_sol.y[4]
p3_o = trajektorie_optimal_sol.y[5]
t_raw = trajektorie_optimal_sol.t

# Abstand in der neuen Lösung berechnen
abstand_neu = np.sqrt((q1_o - (1-mu))**2 + q2_o**2 + q3_o**2)

# Erstes lokales Minimum in der neuen Lösung finden
idx_min_neu = None
prominenz_neu = 0.3 * (abstand_neu.max() - abstand_neu.min())
minima_neu_idx, _ = find_peaks(-abstand_neu, prominence=prominenz_neu)

for i in minima_neu_idx:
    if abstand_neu[i] >= Radius:
        idx_min_neu = int(i)
        break
if idx_min_neu is None:
    idx_min_neu = len(abstand_neu) - 1

# Länge der Trajektorie zuschneiden, sodass sie bei der Erde stopt. Danach umkehren
q1_opt = q1_o[:idx_min_neu+1][::-1]
q2_opt = q2_o[:idx_min_neu+1][::-1]
q3_opt = q3_o[:idx_min_neu+1][::-1]
p1_opt = p1_o[:idx_min_neu+1][::-1]
p2_opt = p2_o[:idx_min_neu+1][::-1]
p3_opt = p3_o[:idx_min_neu+1][::-1]

# Zeitachse: echte Integrationszeiten, auf denselben Abschnitt zugeschnitten und umgekehrt wie q1_opt..p3_opt.
t = t_raw[:idx_min_neu+1][::-1]
t = t - t[0]

# Optimierte Geschwindigkeit definieren
v1_opt = p1_opt + q2_opt
v2_opt = p2_opt - q1_opt
v3_opt = p3_opt
v_opt  = np.sqrt(v1_opt**2 + v2_opt**2 + v3_opt**2)

# Optimierte Beschleunigung definieren
a1_opt = np.gradient(v1_opt, t)
a2_opt = np.gradient(v2_opt, t)
a3_opt = np.gradient(v3_opt, t)
a_opt  = np.sqrt(a1_opt**2 + a2_opt**2 + a3_opt**2)

# Abstände definieren in km
abstand_km = np.sqrt((q1_opt - (1-mu))**2 + q2_opt**2 + q3_opt**2) * L_fundamental / 1e3

# Zustandsvektor definieren für die optimierte Referenztrajektorie
X_ref = np.vstack((q1_opt, q2_opt, q3_opt, p1_opt, p2_opt, p3_opt))