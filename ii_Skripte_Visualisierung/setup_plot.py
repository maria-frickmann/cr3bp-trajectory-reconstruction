import shutil
from pathlib import Path
import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.transforms import Bbox
try:
    from .farben import WEISS, SCHWARZ
except ImportError:
    from farben import WEISS, SCHWARZ
fontsize = 14

# --- LaTeX-Schriftbild ------------------------------------------------------
LATEX_PRAEAMBEL = '\n'.join([
    r'\usepackage[T1]{fontenc}',
    r'\usepackage{lmodern}',
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
    r'\usepackage[ngerman]{babel}',
])

def latex_schriftbild():
    """Schriftbild aller Diagramme auf die LaTeX-Schreibweise umstellen."""
    matplotlib.rcParams.update({
        'text.usetex': shutil.which('latex') is not None,
        'text.latex.preamble': LATEX_PRAEAMBEL,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
        'axes.formatter.use_mathtext': True,
    })

latex_schriftbild()

farben_rauschen = ["#3271a5", "#658354", "#dc6601"]
farben_balken   = ["#26637F", "#A62C2B"]

# --- Standardisierte LaTeX-Schreibweise -------------------------------------
def prozent(anteil):
    """Relativen Anteil als standardisierte Prozentangabe im Mathe-Modus."""
    return rf'${anteil * 100:g}\%$'

def zahl(wert, spezifikation='g'):
    """Zahlenwert als standardisierte Angabe im Mathe-Modus. """
    text = format(wert, spezifikation)
    if 'e' in text:
        mantisse, exponent = text.split('e')
        potenz = rf'10^{{{int(exponent)}}}'
        text = potenz if float(mantisse) == 1 else rf'{mantisse} \cdot {potenz}'
    return f'${text}$'

def klartext(beschriftung):
    """LaTeX-Beschriftung als reiner Text, für Konsolenausgaben."""
    return (beschriftung.replace(r'\%', '%').replace(r'\,', ' ')
                        .replace('$', '').strip())

labels_rauschen      = [prozent(0.0001), prozent(0.001), prozent(0.01)]
HALO_ORBIT_LABEL     = 'Halo-Orbit (NASA Id 1212)'

SYM_ALPHA_MIN         = r'\alpha_{\min}'
SYM_ALPHA_MIN_CS      = r'\alpha_{\min}^{\mathrm{CS}}'
SYM_ALPHA_MIN_BLR     = r'\alpha_{\min}^{\mathrm{BLR}}'
SYM_KORR_CS           = r'\mathrm{r}^{\mathrm{CS}}'
SYM_KORR_BLR          = r'\mathrm{r}^{\mathrm{BLR}}'
SYM_DELTA_TOTAL       = r'\Delta_{\mathrm{total}}'

LABEL_ZEITINDEX       = r'Zeitindex $n$'
LABEL_Q1              = r'$q_1$ [normiert]'
LABEL_Q2              = r'$q_2$ [normiert]'
LABEL_Q3              = r'$q_3$ [normiert]'
LABEL_MESSWERT        = rf'Messwert $y$ ({LABEL_Q1})'

LABEL_ALPHA_MIN       = f'${SYM_ALPHA_MIN}$'
LABEL_ALPHA_MIN_CS    = f'${SYM_ALPHA_MIN_CS}$'
LABEL_ALPHA_MIN_BLR   = f'${SYM_ALPHA_MIN_BLR}$'
LABEL_KORR_CS         = f'${SYM_KORR_CS}$'
LABEL_KORR_BLR        = f'${SYM_KORR_BLR}$'

LABEL_MESSANTEIL      = r'Messanteil $\alpha$'
LABEL_FORMKORRELATION = r'Formkorrelation $\mathrm{r}(\alpha)$'
LABEL_RAUSCHPEGEL     = r'Rauschpegel $\sigma_{\mathrm{rel}}$'
LABEL_REFERENZ        = 'Referenztrajektorie'
LABEL_TRANSFERBAHN    = 'Optimale Transferbahn'
LABEL_W_STABIL        = r'Stabile Mannigfaltigkeit $W^{\mathrm{s}}$'
LABEL_W_INSTABIL      = r'Instabile Mannigfaltigkeit $W^{\mathrm{u}}$'
LAGRANGE_LABELS       = [r'$L_1$', r'$L_2$', r'$L_3$', r'$L_4$', r'$L_5$']

PLOT_STYLE = {
    'figsize': (7, 7),
    'title_size': 12,
    'orbit_line': {'lw': 2.2, 'zorder': 10},
    'manifold_line': {'lw': 0.55, 'alpha': 0.50}
}

def style_ax(ax):
    ax.set_facecolor(WEISS)
    ax.tick_params(colors=SCHWARZ)
    ax.xaxis.label.set_color(SCHWARZ)
    ax.yaxis.label.set_color(SCHWARZ)
    ax.title.set_color(SCHWARZ)
    for spine in ax.spines.values():
        spine.set_edgecolor(SCHWARZ)

def get_manifold_window(l2_q1, mu, abst=0.007):
    mitte = (l2_q1 + (1 - mu)) / 2
    return (mitte - abst, mitte + abst), (-abst, +abst), (-abst, +abst)

def speicher_bereich(fig, oben_bis=None):
    """Bildausschnitt für das Speichern. """
    if oben_bis is None:
        return 'tight'
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rand   = matplotlib.rcParams['savefig.pad_inches']
    bbox   = fig.get_tightbbox(renderer).padded(rand)
    zoll   = fig.dpi_scale_trans.inverted()
    oben   = oben_bis.get_window_extent(renderer).transformed(zoll)
    return Bbox.from_extents(bbox.x0, bbox.y0, bbox.x1, min(bbox.y1, oben.y1 + rand))

def inhalt_bereich(fig, mess_dpi=150, schwelle=8, anschnitt=(0.0, 0.0, 0.0, 0.0)):
    """Bildausschnitt, der genau den gezeichneten Inhalt umfasst. """
    alte_dpi = fig.dpi
    fig.set_dpi(mess_dpi)
    leinwand = FigureCanvasAgg(fig)
    leinwand.draw()
    bild = np.asarray(leinwand.buffer_rgba()).astype(float)
    fig.set_dpi(alte_dpi)

    hintergrund = np.array(matplotlib.colors.to_rgba(fig.get_facecolor())) * 255
    inhalt = np.abs(bild - hintergrund).max(axis=2) > schwelle
    zeilen  = np.flatnonzero(inhalt.any(axis=1))
    spalten = np.flatnonzero(inhalt.any(axis=0))
    if zeilen.size == 0 or spalten.size == 0:
        return 'tight'

    hoehe = bild.shape[0]
    x0, y0 = spalten[0] / mess_dpi, (hoehe - 1 - zeilen[-1]) / mess_dpi
    x1, y1 = (spalten[-1] + 1) / mess_dpi, (hoehe - zeilen[0]) / mess_dpi
    links, rechts, unten, oben = anschnitt
    breite, hohe = x1 - x0, y1 - y0
    return Bbox.from_extents(x0 + links * breite, y0 + unten * hohe,
                             x1 - rechts * breite, y1 - oben * hohe)

BILD_ENDUNGEN = ('.png', '.jpg', '.jpeg', '.svg', '.eps', '.tif', '.tiff')

def save_diagram(fig, dateiname, unterordner='Methoden', oben_bis=None, bereich=None):
    basis_pfad = Path(__file__).resolve().parents[1]
    ziel_ordner = basis_pfad / 'iii_Daten_Plots' / unterordner
    ziel_ordner.mkdir(parents=True, exist_ok=True)
    vollständiger_pfad = ziel_ordner / dateiname
    if vollständiger_pfad.suffix.lower() in BILD_ENDUNGEN:
        vollständiger_pfad = vollständiger_pfad.with_suffix('.pdf')
    elif vollständiger_pfad.suffix.lower() != '.pdf':
        vollständiger_pfad = vollständiger_pfad.with_name(vollständiger_pfad.name + '.pdf')
    if bereich is None:
        bereich = speicher_bereich(fig, oben_bis)
    fig.savefig(vollständiger_pfad, bbox_inches=bereich, dpi=3000)