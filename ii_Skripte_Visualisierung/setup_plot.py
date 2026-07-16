from pathlib import Path
try:
    from .farben import WEISS, SCHWARZ
except ImportError:
    from farben import WEISS, SCHWARZ
fontsize = 11
farben_rauschen = ["#3271a5", "#658354", "#dc6601"]
farben_balken   = ["#26637F", "#A62C2B"]

labels_rauschen      = ['0.01%', '0.1%', '1%']
HALO_ORBIT_LABEL     = 'Halo-Orbit (NASA Id 1212)'

LABEL_ZEITINDEX       = r'Zeitindex $n$'
LABEL_Q1              = r'$q_1$ [normiert]'
LABEL_Q2              = r'$q_2$ [normiert]'
LABEL_Q3              = r'$q_3$ [normiert]'

LABEL_ALPHA_MIN_CS    = r'$\alpha_{\min}^{\mathrm{CS}}$'
LABEL_ALPHA_MIN_BLR   = r'$\alpha_{\min}^{\mathrm{BLR}}$'
LABEL_REFERENZ        = 'Referenztrajektorie'

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

def save_diagram(fig, dateiname, unterordner='Methoden'):
    basis_pfad = Path(__file__).resolve().parents[1]
    ziel_ordner = basis_pfad / 'iii_Daten_Plots' / unterordner
    ziel_ordner.mkdir(parents=True, exist_ok=True)
    vollständiger_pfad = ziel_ordner / dateiname
    fig.savefig(vollständiger_pfad, bbox_inches='tight', dpi=1200)