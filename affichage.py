import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm   # Colormap
from astropy.timeseries import LombScargle
from pathlib import Path    # Pour parcourir les fichiers


# ========================= Style des graphiques ========================= #

pgf_with_latex = {      # Dictionnaire pour utiliser LaTeX dans matplotlib
    'mathtext.fontset' : 'stix',
    'xtick.top'   : True,
    'ytick.right' : True,
    'xtick.direction' : 'in',
    'ytick.direction' : 'in',
    'font.size'       : 16,     # Taille de police globale du graphique
    'axes.labelsize'  : 20,     # Taille de police x et y ; LaTeX par défaut : 10pt
    'legend.fontsize' : 16,     # Taille de police à l'intérieur de la légende                                
    'xtick.labelsize' : 14,     # Taille de police des valeurs sur les axes
    'ytick.labelsize' : 14,
    'lines.linewidth' : 1.5     # Épaisseur des traits
}
mpl.rcParams.update(pgf_with_latex)     # Injecter le dictionnaire dans matplotlib


# ==================== Dictionnaire des modes (l, m) ==================== #

# Dictionnaires des modes dans fichiers inerP..., l+m pair ou impair
pair   = {    # Modes pairs correspondent souvent au potentiel Poloïdal P
    1 : r'$(1,1)$', 3 : r'$(2,2)$', 4 : r'$(3,1)$', 6 : r'$(3,3)$',
    8 : r'$(4,2)$', 10: r'$(4,4)$', 11: r'$(5,1)$', 13: r'$(5,3)$',
    15: r'$(5,5)$', 17: r'$(6,2)$', 19: r'$(6,4)$', 21: r'$(6,6)$'
}

impair = {    # Modes impairs correspondent souvent au potentiel Toroïdal T
    2 : r'$(2,1)$', 5 : r'$(3,2)$', 7 : r'$(4,1)$',
    9 : r'$(4,3)$', 12: r'$(5,2)$', 14: r'$(5,4)$',
    16: r'$(6,1)$', 18: r'$(6,3)$', 20:r'$(6,5)$'
}

# Fusion des dictionnaires de modes
tot = {**pair, **impair}   # ** "vide" le dictionnaire dans l'accolade
tab = tot   # Souvent modes pairs pour P et les impairs pour T

# Couleur de chaque mode (colormap)
n_couleurs = len(tab)
couleurs = cm.get_cmap('tab20', n_couleurs).colors
# palette = cm.get_cmap('tab20').colors


# =============== Choix de la simulation + Lecture fichier =============== #

# Paramètres de la simualtion à étudier
omega_str = '0p98'
Ct_str    = '5em2'

# Conversion en float : 'p' -> 'virgule' et 'm' -> 'négatif'
omega = float( omega_str.replace('m', '-').replace('p', '.') )
Ct    = float( Ct_str.replace('m', '-') )

# Répertoire du dossier principal correspondant à Ct
dossier = Path( f'/home/manuelf/PIRS8/PIR/inerP_A{Ct_str}_alpha0p7/om{omega_str}_A{Ct_str}_alpha0p7/' )

fichier = None

for f in dossier.glob('inerP.*') :
    if ':Zone.Identifier' not in f.name : 
        fichier = f  # Afin d'éviter les fichiers "fantômes"
        break

# Lecture des données du fichier
if fichier is not None :
    data = np.loadtxt(fichier)
    print("Fichier lu avec succès.")
else:
    print("Attention : Aucun fichier valide trouvé dans ce dossier.")


# =================== Calcul et stockage des spectres =================== #

# Échantillonnage
itend = len( data[:, 0] )   # Temps max
t     = data[:itend, 0]   
sr = len(t)
dt = max(t) / sr  # Moyenne grossière de dt

# Tableau de fréquences pour LombScargle
f_min = 0.001
f_max = 3.0 / (2 * np.pi)   # Calcul jusqu'à omega = 3 pour "avoir de la marge"
freq_hz    = np.linspace(f_min, f_max, 5000)  # 5000 pts --> résolution = 0.0006
freq_omega = freq_hz * 2 * np.pi

# Fenêtre spectrale (fonction d'Hanning)
w1 = np.zeros(sr)
for n in range(sr) :
    w1[n] = 1./2 - (1./2) * np.cos( (2*np.pi*n)/(sr-1) )

# w1 = 1  # Pour teste sans Hanning

# Stockage des spectres et calcul du max pour ensuite normaliser
spectres_list = {}
max_global = 0.0

for i, lm in enumerate(tab):
    # Multiplication par la fenêtre d'Hanning
    signal_fenetre = data[:itend, lm] * w1
    
    # Lombscargle
    ls = LombScargle(t, signal_fenetre)
    
    # Calcul de la puissance (proportionnel au carré de vitesse poloïdal)
    puissance = ls.power(freq_hz, normalization='psd')     # 'psd' -> Pas normalisé
    amp = np.sqrt(puissance)

    # Stockage du mode (l,m)
    spectres_list[lm] = amp

    # Recherche du maximum de la simulation
    max_simu = np.max(amp)
    if max_simu > max_global : 
        max_global = max_simu


# =================== Affichage du spectre fréquentiel =================== #

plt.figure(figsize=(9, 6))

lim = 0.05          # Affichage des modes > lim
modes_retenus = []  # Liste des modes supérieurs > lim pour spectre temporel
# n_courbes = 0     # Autre manière de faire distribution des couleurs (ne pas utiliser)

for i, (lm, amp) in enumerate(spectres_list.items()) :
    
    # Normalisation par l'amplitude maximale
    amp_normalisee = amp / max_global

    # Plot du mode (l,m) si > lim
    if np.any(amp_normalisee > lim) :

        # couleur_mode = palette[n_courbes % 20]  # %20 car parlette contient 10 couleurs contrastées

        plt.plot(freq_omega, amp_normalisee, label=tab[lm], color=couleurs[i])
        # plt.plot(freq_omega, amp_normalisee, label=tab[lm], color=couleur_mode)

        # Sauvegarde du mode
        modes_retenus.append((i, lm))

        # n_courbes += 1

# Fréquence de forçage et sub-harmoniques
plt.axvline(x=abs(omega)  , color='lightgray', alpha=0.7, linestyle='--', zorder=50)
plt.axvline(x=abs(omega/2), color='lightgray', alpha=0.7, linestyle='--', zorder=50)
plt.axvline(x=abs(omega/3), color='lightgray', alpha=0.7, linestyle='--', zorder=50)
plt.axvline(x=abs(omega/4), color='lightgray', alpha=0.7, linestyle='--', zorder=50)

# Paramètres des axes
plt.xlim([0, 2])
plt.ylim([0, 1])     # Si on veut bien voir pics secondaires (A MOFIDIER)
plt.title(fr'$\omega = {omega}, C_t = {Ct}$')
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\mathrm{power\ of\ W}$')
plt.legend(title=r'$(l,m)$', fontsize=12, ncol=2)

# plt.savefig( f'ls_om{omega_str}_CT{Ct_str}_alpha0p7_inerP.pdf', bbox_inches='tight' )


# ==================== Affichage du spectre temporel ==================== #

plt.figure(figsize=(9, 6))

# Affichage des modes > lim
for i, lm in modes_retenus :
    
    serie_temporelle = data[1:itend, lm]
    log_temporel     = np.log( np.abs(serie_temporelle) )
    
    plt.plot(t[1:], log_temporel, label=tab[lm], linewidth=1, color=couleurs[i])

# Finitions du graphique temporel
plt.title(f"Évolution temporelle")
plt.xlabel(r'$t$')
plt.ylabel(r'$\mathrm{log(W)}$')
plt.legend(title=r'$(l,m)$', fontsize=12, ncol=2)


# ======================================================================= #
plt.show()