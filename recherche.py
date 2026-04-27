import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
from matplotlib import cm           # Couleurs
from astropy.timeseries import LombScargle
from pathlib import Path    # Pour parcourir les fichiers


def extraire_frequence(nom_dossier):
    """
    Extrait la fréquence de forçage depuis le nom du dossier.
    Exemples: 'om0p8_A1...' -> 0.8 | 'omm0p6_A1...' -> -0.6
    """
    # 1. On cherche ce qui se trouve entre "om" et "_A"
    recherche = re.search(r'om(.*?)_A', nom_dossier)
    
    if recherche:
        # 2. On récupère la chaîne trouvée (ex: "m0p6")
        freq_texte = recherche.group(1)     # group(1) prend ce qu'il y a dans la première parenthèse
        
        # 3. On remplace le 'm' par '-' et le 'p' par '.'
        freq_propre = freq_texte.replace('m', '-').replace('p', '.')
        
        # 4. On convertit le texte en vrai nombre décimal
        return float(freq_propre)
    else:
        print(f"⚠️ Impossible de trouver la fréquence dans : {nom_dossier}")
        return None


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
    'lines.linewidth' : 3
}
mpl.rcParams.update(pgf_with_latex)     # Injecter le dictionnaire dans matplotlib


# ==================== Dictionnaire des modes (l, m) ==================== #
# Dictionnaires des modes dans fichiers inerP..., l+m pair ou impair
even = {    # Modes pairs correspondent souvent au potentiel Poloïdal P
    1 : r'$(1,1)$', 3 : r'$(2,2)$', 4 : r'$(3,1)$', 6 : r'$(3,3)$',
    8 : r'$(4,2)$', 10: r'$(4,4)$', 11: r'$(5,1)$', 13: r'$(5,3)$',
    15: r'$(5,5)$', 17: r'$(6,2)$', 19: r'$(6,4)$', 21: r'$(6,6)$'
}

odd  = {    # Modes impairs correspondent souvent au potentiel Toroïdal T
    2 : r'$(2,1)$', 5 : r'$(3,2)$', 7 : r'$(4,1)$',
    9 : r'$(4,3)$', 12: r'$(5,2)$', 14: r'$(5,4)$',
    16: r'$(6,1)$', 18: r'$(6,3)$', 20:r'$(6,5)$'
}

# Fusion des dictionnaires de modes
tot = {**even, **odd}   # ** "vide" le dictionnaire dans l'accolade
tab = tot     # usually even for P, odd for T

# Couleur de chaque mode
n_colors = len(tab)
colours  = cm.nipy_spectral( np.linspace(0, 1, n_colors) )


# =============== Recherche des simulations intéressantes =============== #
main_folder = Path( '/home/manuelf/PIRS8/PIR/inerP_A1em2_alpha0p7/' )

for file in main_folder.glob('*/inerP.*'):   # Cible inerP.* de chaque dossier (glob() prend fichiers type 'inerP.*')

    # Si le nom du fichier se termine par :Zone.Identifier, on l'ignore et on passe au suivant
    if file.name.endswith(':Zone.Identifier'):
        continue    

    folder_name  = file.parent.name   # Nom du dossier 'om...p..._A1em2_alpha0p7'
    freq_forcage = extraire_frequence(folder_name)

    # Lecture des données du fichier
    try:
        data = np.loadtxt(file)

        # Sampling rate
        itend = len(data[:,0]) 
        t     = data[:itend,0]     # you might want to change itend (and istart) to analyse a smaller interval of the simu
        sr = len(t)
        dt = max(t)/sr
        #dt = np.abs( np.diff(t) )

        # Window functions
        w1 = np.zeros(sr)
        for n in range(sr):
            w1[n] = 1./2-1./2*np.cos(2*np.pi*n/(sr-1))  # Hanning function
        
        # Compute Fourier transform and frequencies
        freqs   = np.fft.fftfreq(sr, d=dt)*2*np.pi

        # Paramètres pour "cacher" les fréquences pas intéressantes
        peak_width = 0.01 # Demi-largeur moyenne des pics

        # ==============================================================
        # ÉTAPE 1 : Calculer toutes les FFT et trouver le Maximum Global
        # ==============================================================
        spectres_par_mode = {}
        max_global_simulation = 0.0
        infos_mode_principal = {}

        for i, lm in enumerate(tab):
            # ### FOURIER
            # fft = np.fft.fft( data[:itend,lm]*w1 )
            # amp = np.abs(fft)
            
            ### LOMBSCARGLE
            f_min = 0.001
            f_max = 3.0 / (2 * np.pi) 
            freqs_hz = np.linspace(f_min, f_max, 5000) # Grille de 5000 points pour une belle résolution
            freqs = freqs_hz * 2 * np.pi           # Conversion en omega pour ton axe X
            ls = LombScargle(t, data[:itend, lm] * w1)
            puissance = ls.power(freqs_hz, normalization='psd')
            amp = np.sqrt( puissance )

            # On stocke l'amplitude pour l'étape 2 (pour ne pas recalculer la FFT)
            spectres_par_mode[lm] = amp 

            # On cherche si le max de ce mode est le nouveau record de la simulation
            max_du_mode = np.max(amp[freqs > 0]) 
            if max_du_mode > max_global_simulation:
                max_global_simulation = max_du_mode

                indice_max = np.argmax(amp)
                infos_mode_principal = {
                    'mode': tab[lm],
                    'omega': freqs[indice_max],
                    'amplitude': max_global_simulation
                }

        # ==============================================================
        # ÉTAPE 2 : Appliquer le filtre et chercher les pics > 10%
        # ==============================================================
        simulation_retenue = False

        # On crée une liste pour stocker tous les pics trouvés pour CE dossier
        pics_trouves = []

        for lm, amp in spectres_par_mode.items():
            amp_bis = np.copy(amp)  # ET PAS JUSTE = !!!
            amp_bis[freqs <= 0] = 0     # Pas besoin avec LombScargle

            # On "efface" la fréquence de forçage et ses harmoniques
            for n in range(1, 3): # Exemple: 1 et 2
                freq_harmonique = n * abs(freq_forcage)
                zone_a_effacer = (freqs >= freq_harmonique - peak_width) & (freqs <= freq_harmonique + peak_width)
                amp_bis[zone_a_effacer] = 0

            # Le pic secondaire de CE mode est le max du spectre masqué
            pic_secondaire = np.max(amp_bis)

            # Ta condition finale avec le max GLOBAL !
            if pic_secondaire > 0.10 * max_global_simulation:
                simulation_retenue = True
                # 1. On cherche la position (l'indice) de ce maximum dans le tableau
                indice_du_pic = np.argmax(amp_bis)
                
                # 2. On récupère le omega correspondant à cette position
                omega_du_pic = freqs[indice_du_pic]
                
                # 3. On sauvegarde les infos de ce mode
                pics_trouves.append({
                    'mode': tab[lm],
                    'omega': omega_du_pic,
                    'amplitude': pic_secondaire
                })

        if simulation_retenue:
            print(f"✅ Retenu : {folder_name}")
            # On affiche chaque pic trouvé joliment
            for pic in pics_trouves:
                print(f"   -> Mode {pic['mode']:<6} | Omega = {pic['omega']:.4f} | Amplitude = {pic['amplitude']:.2e}")
            print("-" * 50) # Petite ligne de séparation pour faire propre

    except Exception as e:
        print(f"Erreur de lecture sur {folder_name} : {e}")
        continue # Si le fichier est corrompu, on passe au suivant