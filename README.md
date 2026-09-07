# Rekonstruktion einer Transfertrajektorie zu einem Halo-Orbit um den Lagrange-Punkt $L_2$

**Vergleich von Compressed Sensing und bayesscher linearer Regression bei unvollständigen und verrauschten Messdaten**

Programmcode zur Maturaarbeit von Maria Frickmann, Kantonsschule Kollegium Schwyz.


## Kontext


Eine Raumsonde auf dem Weg zu einem Halo-Orbit um den Lagrange-Punkt $L_2$ des Sonne–Erde-Systems wird nur an wenigen, verrauschten Zeitpunkten vermessen. Die Frage ist, wie gross der Messanteil $\alpha$ (Anteil der tatsächlich gemessenen Bahnpunkte) mindestens sein muss, damit sich die Form der vollständigen Trajektorie zuverlässig rekonstruieren lässt.

Dazu werden zwei Verfahren auf denselben Daten verglichen:

- **Compressed Sensing (CS):** $\ell_1$-Minimierung im DCT-Raum, gelöst mit `cvxpy`.
- **Bayessche lineare Regression (BLR):** geschlossener Posterior mit einem
  Prior, dessen Skala mit dem Frequenzindex abfällt.

Bewertet werden beide über dasselbe Kriterium $\alpha_{\min}$: den kleinsten
Messanteil, ab dem die Formkorrelation $\mathrm{r}(\alpha)$ eine Schwelle
$\rho$ über ein ganzes Fenster der Breite $w$ hinweg nicht mehr unterschreitet.
Zusätzlich wird geprüft, ob die rekonstruierte Bahn die Jacobi-Konstante des
CR3BP erhält.

## Aufbau des Repositories

| Ordner | Inhalt |
|---|---|
| `i_Programmierung/` | Physik und Methoden werden definiert und berechnet |
| `ii_Skripte_Visualisierung/` | Plot-Skripte werden erstellt. |
| `iii_Daten_Plots/` | Die erzeugten Diagramme werden als PDF ausgegeben. |

```
i_Programmierung/
  Konstanten.py            Massenverhältnis mu, Schwellen rho/tau, Wiederholungen, Startwert
  Konfiguration.py         Referenz-Jacobi-Konstante, alpha- und Rauschen-Listen
  Modellierung/
    physik.py              Bewegungsgleichungen und Jacobi-Konstante des CR3BP
    lagrange.py            Lage der fünf Lagrange-Punkte
    orbit.py               Halo-Orbit (NASA Id 1212), Monodromiematrix
    mannigfaltigkeiten.py  Stabile/instabile Mannigfaltigkeit, Referenztrajektorie X_ref
  Methoden/
    compressed_sensing.py           Messdaten, l1-Minimierung, Metrik Delta_total
    bayessche_lineare_regression.py Posterior-Mittelwert und Quantile
    kriterium.py                    Kriterium alpha_min

ii_Skripte_Visualisierung/
  berechnungen.py          Einziger Ort, an dem für die Plots gerechnet wird
  setup_plot.py            Schriftbild, Beschriftungen, save_diagram()
  farben.py, hintergrund.py
  Modellierung/            Potentialfläche, optimale Trajektorie, Parametertabelle
  Kriterium_alpha_min/     alpha_min für CS, für BLR, Balkenvergleich
  cs_vs_blr/               Direkter Vergleich beider Verfahren
  bayes/                   Prior/Likelihood/Posterior, Kredibilitätsbänder, PPC
  physik_rekonstruktion/   3D-Rekonstruktionen, Jacobi-Tabelle
  weitere/                 Schemazeichnung des CR3BP-Aufbaus
```

## Lizenz

[MIT](LICENSE) © 2026 Maria Frickmann
