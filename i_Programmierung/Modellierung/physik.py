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

def U_mu(q1, q2, q3, mu):
    """ Berechnet das effektive Potential im rotierenden System. """
    r_e, r_s = abstand_primaerkoerper(q1, q2, q3, mu)
    return - (1 - mu) / r_s - mu / r_e - 0.5 * (q1**2 + q2**2)


def partial_U_mu(q1, q2, q3, mu):
    """ Berechnet die partiellen Ableitungen des effektiven Potentials im rotierenden System. """

    r_e, r_s = abstand_primaerkoerper(q1, q2, q3, mu)

    dU_dq1 = (1 - mu)*(q1 + mu)/r_s**3 + mu*(q1 - 1 + mu)/r_e**3 - q1
    dU_dq2 = (1 - mu)*q2/r_s**3        + mu*q2/r_e**3            - q2
    dU_dq3 = (1 - mu)*q3/r_s**3        + mu*q3/r_e**3

    return dU_dq1, dU_dq2, dU_dq3

def cr3bp_hamilton(t, state, mu):
    """ Definiert den Hamiltonian für das CR3BP"""

    # Zustandsvariabeln
    q1, q2, q3, p1, p2, p3 = state

    return 0.5 * ((p1 + q2)**2 + (p2 - q1)**2 + p3**2) + U_mu(q1, q2, q3, mu)

def bewegungsgleichungen(t, state, mu):
    """ Definiert die Bewegungsgleichungen des Hamiltonians """

    q1, q2, q3, p1, p2, p3 = state
    dU_dq1, dU_dq2, dU_dq3 = partial_U_mu(q1, q2, q3, mu)

    # Geschwindigkeiten definieren 
    dq1_dt = p1 + q2
    dq2_dt = p2 - q1 
    dq3_dt = p3

    # Ableitung des Impulses nach der Zeit
    dp1_dt =  (p2 - q1) - dU_dq1
    dp2_dt = -(p1 + q2) - dU_dq2
    dp3_dt =            - dU_dq3

    return np.array([dq1_dt, dq2_dt, dq3_dt, dp1_dt, dp2_dt, dp3_dt])

def jacobi_konstante(q1, q2, q3, p1, p2, p3, mu):
    """ Berechnet die Jacobi Konstante. """
    return -2 * cr3bp_hamilton(0.0, [q1,q2,q3,p1,p2,p3], mu)
