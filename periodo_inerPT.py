##############################################
### author: A. Astoul, mod. by..           ###                
### Plot periodogram from iner_P or T .tag ###
##############################################

from astropy.timeseries import LombScargle
from numpy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from matplotlib import cm

# style
pgf_with_latex = { # setup matplotlib to use LaTeX for output
    # 'pgf.texsystem': 'pdflatex', # change this if using XeTeX or LuaTeX
    # 'text.usetex': True, # use LaTeX to write all text # often not working 
    'mathtext.fontset': 'stix',
    'xtick.top':True,
    'ytick.right':True,
    'xtick.direction':'in',
    'ytick.direction':'in',
    'axes.labelsize': 20, # x and y label fontsize; laTeX default is 10pt font                                                                  
    'font.size': 16,       # Legend title fontsize                                                                                              
    'legend.fontsize': 16, # Legend fontsize; make the legend/label fonts a little smaller                                                      
    'xtick.labelsize': 14, # values fontsize                                                                                                   
    'ytick.labelsize': 14,
    'lines.linewidth': 3
}
mpl.rcParams.update(pgf_with_latex)

# modes in inerPT.tag, sum l+m even or odd
even = {1:r'$(1,1)$',3:r'$(2,2)$',4:r'$(3,1)$',6:r'$(3,3)$',8:r'$(4,2)$',10:r'$(4,4)$',11:r'$(5,1)$',13:r'$(5,3)$',15:r'$(5,5)$',17:r'$(6,2)$',19:r'$(6,4)$',21:r'$(6,6)$'}
odd  = {2:r'$(2,1)$',5:r'$(3,2)$',7:r'$(4,1)$',9:r'$(4,3)$',12:r'$(5,2)$',14:r'$(5,4)$',16:r'$(6,1)$',18:r'$(6,3)$',20:r'$(6,5)$'}
tot  = {key: value for d in (even, odd) for key, value in d.items()}

# parameters to change
tab  = tot     # usually even for P, odd for T
pot  = 'P'     # poloidal or toroidal potential
pots = {'P':'W','T':'Z'}  
inif_str = '0p98'              # initial frequency (1 per simulation)
inif_num = 0.98
CT    = '5em2'            # tidal amplitude forcing     1em2 OU 5em2
CT_num = 5                # 1 ou 5
tag  = '_alpha0p7'        # shell aspect ratio

# select colors for != modes
n_colors = len(tab)
colours  = cm.nipy_spectral(np.linspace(0, 1, n_colors))

# directories and extension (to adapt)
nam  = 'om'+inif_str+'_A'+CT+tag+'/'
do   = {nam:['test']}              # 'test', 'hiPhiRes', 'test_BIS', 'hiPhiRes_BIS', 'hiSpaTimRes'
wdir = '/home/manuelf/PIRS8/PIR/manu_red/'  # '/home/manuelf/PIRS8/PIR/inerP_A1em2_alpha0p7/' ou '/home/manuelf/PIRS8/PIR/manu_red/'
tagi = do[nam][0]
fdom = open(wdir+nam+'iner'+pot+'.'+tagi,'r')
la   = np.loadtxt(fdom)
    
# sampling rate
itend = len(la[:,0]) 
t     = la[:itend,0]     # you might want to change itend (and istart) to analyse a smaller interval of the simu
sr = len(t)
dt = max(t)/sr

# Window functions
w1 = np.zeros(sr)
for n in range(sr):
    w1[n] = 1./2-1./2*np.cos(2*np.pi*n/(sr-1))        # Hanning function

# Début de la figure
plt.figure(figsize=(9, 6))     # PENSER A MODIFIER

# compute Fourier transform and frequencies
freqs   = np.fft.fftfreq(sr, d=dt)*2*np.pi
lim     = 1e4
for i,lm in enumerate(tab):
    fft = np.fft.fft(la[:itend,lm]*w1)
    # select frequencies with highest power
    # if np.any(abs(fft)>lim):
    #     plt.plot(freqs, abs(fft),label=tab[lm],color=colours[i])
    # TOUS LES MODES
    plt.plot(freqs, abs(fft),label=tab[lm],color=colours[i],linewidth=1.5)

# # =========================================================================
# # compute Lomb-Scargle periodogram and frequencies
# # =========================================================================

# # 1. Définir la grille de fréquences (f = omega / 2*pi)
# # Vu que ton xlim est à [0, 2] pour omega, on calcule f pour omega allant jusqu'à 3
# f_min = 0.001
# f_max = 3.0 / (2 * np.pi) 
# freqs_hz = np.linspace(f_min, f_max, 5000) # Grille de 5000 points pour une belle résolution
# freqs_omega = freqs_hz * 2 * np.pi           # Conversion en omega pour ton axe X

# for i, lm in enumerate(tab):
#     # Appliquer la fenêtre de Hanning au signal
#     signal_fenetre = la[:itend, lm] * w1
    
#     # 2. Initialiser Lomb-Scargle avec ton vecteur temps (t) EXACT
#     ls = LombScargle(t, signal_fenetre)
    
#     # 3. Calculer la puissance (Power Spectral Density pour se rapprocher de l'échelle de la FFT)
#     puissance = ls.power(freqs_hz, normalization='psd')
    
#     # TOUS LES MODES
#     plt.plot(freqs_omega, puissance, label=tab[lm], color=colours[i], linewidth=1.5)

# Fréquene de forçage
plt.axvline(x=abs(inif_num), color='lightgray', linestyle='--', linewidth=1.5, zorder=50)

# # Fréquences secondaires !!! POUR 0.98 !!!
# omegas = [ 0.09, 0.13, 0.24, 0.33, 0.41, 0.57, 0.65, 0.81, 0.84, 0.98, 1.07, 1.48, 1.96 ]
# for om in omegas :
#     plt.axvline( x=om, color='lightgray', linestyle='--', linewidth=1.5, zorder=50 )
#     plt.text(
#         x=om,                # Texte exactement sur la ligne
#         y=6e+5,                 
#         s=fr'$\omega \sim {om}$', 
#         rotation=90,         
#         color='black',        
#         fontsize=12,         
#         ha='center',         # Centre le texte horizontalement par rapport à la ligne
#         va='center',         # Centre le texte verticalement
#         zorder=51,           # zorder PLUS GRAND que celui de la ligne (50) pour passer devant
#         bbox=dict(facecolor='white', edgecolor='none', pad=1) # pad gère la marge de la boîte
#     )
# # pour le 0.79 car empiète
# plt.axvline( x=0.79, color='lightgray', linestyle='--', linewidth=1.5, zorder=50 )
# plt.text(
#     x=0.79-0.02,                # Texte exactement sur la ligne
#     y=6e+5,                 
#     s=fr'$\omega \sim 0.79$', 
#     rotation=90,         
#     color='black',        
#     fontsize=12,         
#     ha='center',         # Centre le texte horizontalement par rapport à la ligne
#     va='center',         # Centre le texte verticalement
#     zorder=51,           # zorder PLUS GRAND que celui de la ligne (50) pour passer devant
# )

# # optional: find frequencies associated with highest FFT values
# lmi = 1  # mode you want to look at
# fft = np.fft.fft(la[:itend,lmi]*w1)
# coeftab, freqtab = [], []
# for coef, freq in zip(fft, freqs):
#     if freq>0.05 and freq<0.1:
#         coeftab.append(coef)
#         freqtab.append(freq)
# idx = np.where(coeftab == max(coeftab))[0][0]
# print(int(coeftab[idx]),freqtab[idx],tab[lmi])

# # to adapt: identify frequency excited; clever ways can be used with arrows or embeddded numbers
# omegas = [inif_num,inif_num/2]  # etc ...
# ax     = plt.gca()
# ylims  = ax.get_ylim()

# ps = []
# ls = []

# for omi in omegas:
#     p = plt.vlines(omi,ylims[0],ylims[1],linestyle='--',label=rf'$\omega={omi}$')
#     ps.append(p)

# # gather the legends
# for li in ps:
#     ls.append(li.get_label())
# plt.legend(ps,ls,loc='upper right',ncol=2)
#plt.gca().add_artist(l0)

# Plot Fourier analysis 
plt.xlim([0, 2])
# plt.ylim([1., 14.0e6])     # A MODIFIER A CHAQUE FOIS
plt.ylim(bottom=0)
plt.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
plt.title(fr'$\omega = {inif_num}, C_t = {CT_num} \cdot 10^{{-2}}$')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\mathrm{power\ of\ '+pots[pot]+'}$')
plt.legend(title=r'$(l,m)$', fontsize=12, ncol=2)
## JUSTE POUR 0.98 pour pas que ça empiète
# plt.legend(
#     title=r'$(l,m)$', 
#     fontsize=10,
#     title_fontsize=12,    
#     ncol=2,              
#     loc='upper right',
#     bbox_to_anchor=(0.96, 1.0)
# )
# plt.savefig( 'fft_om'+inif_str+'_CT'+CT+tag+'_iner'+pot+'.pdf', bbox_inches='tight' )    # rajouter '_annote.pdf' si besoin

# # Vérification du vecteur temps
# plt.figure()
# plt.plot(t, lw=1.5)
# plt.title(fr'Vecteur temps de $\omega = {inif_num}, C_t = {CT_num} \cdot 10^{{-2}}$, extention = {tagi}', fontsize=12)
# # plt.savefig( 'timetab_om'+inif_str+'_CT'+CT+tag+'_iner'+pot+'.pdf', bbox_inches='tight' )
plt.show()
