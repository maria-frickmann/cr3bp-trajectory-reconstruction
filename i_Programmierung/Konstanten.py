import numpy as np
# Fundamentale Konstanten
mu               = 3.054200000000000E-6
L_fundamental    = 1.496e+11
Radius_m         = 6371000
Radius           = Radius_m / L_fundamental

# Parameter für compressed sensing
w                = 0.1
startwert_zufall = 42

# Parameter für alpha_min Kriterium 
RHO     = 0.90    # Schwelle der Formkorrelation
TAU_FIX = 0.10    # Fixschwelle: Fehler <= 10 % der Schwingungsamplitude
N_WDH   = 10      # Anzahl Wiederholungen
KAPPA   = 0.95    # Sicherheitsquantil

# Toleranz für die Eigenwertklassifikation der Monodromiematrix
EIGENWERT_TOLERANZ = 0.01

# Anstupsfaktor
delta = 1e-6