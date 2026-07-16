# Pfad definieren
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(current_dir))

# Imports
import numpy as np
from scipy.integrate import solve_ivp
from i_Programmierung.Konstanten import mu

def abstand_primaerkoerper(q1, q2, q3, mu):
    """ Die Funktion berechnet die Abstände des dritten Körpers (Sonde) zu den beiden Primärkörpern (Sonne und Erde) im rotierenden Bezugssystem. """
    r_e = np.sqrt((q1 - (1 - mu))**2 + q2**2 + q3**2)
    r_s = np.sqrt((q1 + mu)**2 + q2**2 + q3**2)
    return r_e, r_s

def V_mu(q1, q2, q3, mu):
    """ Berechnet das effektive Potential im rotierenden System. Berücksichtigt die Gravitation von Sonne und Erde sowie die Zentrifugalkraft."""
    r_e, r_s = abstand_primaerkoerper(q1, q2, q3, mu)
    return - (1 - mu) / r_s - mu / r_e - 0.5 * (q1**2 + q2**2) 

def jacobi_konstante(q1, q2, q3, p1, p2, p3, mu):
    """ Berechnet die Jacobi-Konstante, eine Erhaltungsgrösse im CR3BP, um die Bewegung des dritten Körpers zu analysieren. """
    # Berechnung des effektiven Potentials
    Omega = -V_mu(q1, q2, q3, mu)

    # Geschwindigkeiten definieren
    dq1_dt = p1 + q2
    dq2_dt = p2 - q1
    dq3_dt = p3

    return 2*Omega - dq1_dt**2 - dq2_dt**2 - dq3_dt**2

def cr3bp_hamilton(t, state, mu):
    """ Liefert die Bewegungsgleichungen im Hamilton-Koordinatensystem für das CR3BP. Dazu wird der Hamiltonian nach den Zustandsvariablen abgeleitet. In der q1-q2-Ebene wirkt zusätzlich die Coriolis-Kraft. """
    # Zustandsvariablen
    q1, q2, q3, p1, p2, p3 = state

    # Abstände berechnen zu den Primärkörpern
    r_e, r_s = abstand_primaerkoerper(q1, q2, q3, mu)

    # Geschwindigkeiten definieren 
    dq1_dt = p1 + q2
    dq2_dt = p2 - q1 
    dq3_dt = p3

    # Beschleunigung berechnen
    dv1_dt =  2*dq2_dt + q1 - (1-mu)*(q1+mu)/r_s**3- mu*(q1-1+mu)/r_e**3
    dv2_dt = -2*dq1_dt + q2 - (1-mu)*q2/r_s**3     - mu*q2/r_e**3
    dv3_dt = - (1-mu)*q3/r_s**3 - mu*q3/r_e**3          

    # Ableitung des Impulses nach der Zeit
    dp1_dt = dv1_dt - dq2_dt
    dp2_dt = dv2_dt + dq1_dt
    dp3_dt = dv3_dt

    return [dq1_dt, dq2_dt, dq3_dt, dp1_dt, dp2_dt, dp3_dt]