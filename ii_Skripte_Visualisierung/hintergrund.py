import numpy as np
import matplotlib.pyplot as plt
from .farben import *

from i_Programmierung.Modellierung.physik import U_mu
from i_Programmierung.Modellierung.lagrange import l_punkte
from i_Programmierung.Konstanten import mu
from ii_Skripte_Visualisierung.setup_plot import (fontsize, LAGRANGE_LABELS,
                                                  LABEL_Q1, LABEL_Q2, LABEL_Q3)

pos_sonne = np.array([-mu,    0.0, 0.0])
pos_erde  = np.array([1 - mu, 0.0, 0.0])

def draw_hintergrund_2d(ax, xlim, ylim, potential = True):
    ax.set_facecolor(WEISS)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if potential:
        xg = np.linspace(*xlim, 350)
        yg = np.linspace(*ylim, 350)
        Q1, Q2 = np.meshgrid(xg, yg) 
        U = U_mu(Q1, Q2, 0.0, mu)    
        
        ax.contour(Q1, Q2, U,
                   levels=np.linspace(-3.0, -1.2, 40),
                   colors = [LINIEN])
                   
    ax.scatter(pos_sonne[0], pos_sonne[1], color = SONNE, zorder=2)
    ax.scatter(pos_erde[0], pos_erde[1], color = ERDE, zorder=2)
    
    for point, name in zip(l_punkte, LAGRANGE_LABELS):
        if xlim[0] <= point[0] <= xlim[1] and ylim[0] <= point[1] <= ylim[1]:
            ax.plot(point[0], point[1], 'x', markersize=12, zorder=3)
            ax.text(point[0] + 0.0001, point[1] + 0.0001, name, fontsize=fontsize)

def draw_hintergrund_3d(ax, xlim, ylim, zlim, potential = True):
    ax.set_facecolor(WEISS)
    if potential:
        xg = np.linspace(*xlim, 350)
        yg = np.linspace(*ylim, 350)
        Q1, Q2 = np.meshgrid(xg, yg) 
        U = U_mu(Q1, Q2, 0.0, mu)    
        ax.contour(Q1, Q2, U,
                   levels=np.linspace(-3.0, -1.2, 40),
                   colors = [LINIEN],
                   zdir='z',
                   offset=0.0)
    ax.scatter(pos_sonne[0], pos_sonne[1], pos_sonne[2], color = SONNE)
    ax.scatter(pos_erde[0], pos_erde[1], pos_erde[2], color = ERDE)
    for point, name in zip(l_punkte, LAGRANGE_LABELS):
        if xlim[0] <= point[0] <= xlim[1] and ylim[0] <= point[1] <= ylim[1]:
            ax.plot([point[0]], [point[1]], [point[2]], 'x', markersize=12)
            ax.text(point[0] + 0.0001, point[1] + 0.0001, point[2], name, fontsize=fontsize)
    ax.view_init(elev=40, azim=-110)

def Axenbeschriftung(ax, xlabel=LABEL_Q1, ylabel=LABEL_Q2):
    ax.set_xlabel(xlabel, color=SCHWARZ, fontsize=fontsize)
    ax.set_ylabel(ylabel, color=SCHWARZ, fontsize=fontsize)
    if hasattr(ax, 'set_zlabel'):
        ax.set_zlabel(LABEL_Q3, color=SCHWARZ, fontsize=fontsize)        
    ax.tick_params(colors=SCHWARZ, labelsize=8)
    if hasattr(ax, 'spines') and ax.spines:
        for spine in ax.spines.values():
            spine.set_edgecolor(WEISS)
    ax.grid(color=WEISS, linewidth=0.4, alpha=0.4)