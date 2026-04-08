import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import os
from pathlib import Path
import library
from operator import itemgetter
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed # Import pour le multithreading

# --- 1. CONFIGURATION DU STYLE ---
pgf_with_latex = {
    'mathtext.fontset': 'stix',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.labelsize': 20,
    'font.size': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2
}
mpl.rcParams.update(pgf_with_latex)

# --- 2. PARAMÈTRES ET MODES ---
even = {1:r'$(1,1)$', 3:r'$(2,2)$', 4:r'$(3,1)$', 6:r'$(3,3)$', 8:r'$(4,2)$', 10:r'$(4,4)$', 11:r'$(5,1)$', 13:r'$(5,3)$', 15:r'$(5,5)$', 17:r'$(6,2)$', 19:r'$(6,4)$', 21:r'$(6,6)$'}
odd  = {2:r'$(2,1)$', 5:r'$(3,2)$', 7:r'$(4,1)$', 9:r'$(4,3)$', 12:r'$(5,2)$', 14:r'$(5,4)$', 16:r'$(6,1)$', 18:r'$(6,3)$', 20:r'$(6,5)$'}

pot = 'P'             # 'P' ou 'T'
symbole = 'W' if pot == 'P' else 'Z'
tab = even if pot == 'P' else odd

colours = cm.nipy_spectral(np.linspace(0, 1, len(tab)))

# --- PARAMÈTRES FIXES ---
repertoire_base = './inerP_A1em2_alpha0p7/'
type_potentiel = 'P' 
tag_fichier = 'test'
nom_fichier_donnees = f"iner{type_potentiel}.{tag_fichier}"


def recherche_pics(data,freqs,hanning):
    liste_puissance_Cnm = []
    for i, (colonne, label_Cnm) in enumerate(tab.items()):
        if colonne < data.shape[1]:
            # Calcul de la FFT réelle
            spectre = np.fft.rfft(data.iloc[:, colonne] * hanning)
            puissance = np.abs(spectre)
            
            idx_max = np.argmax(puissance)
            puissance_max = np.max(puissance)
            freq_max = freqs[idx_max]
            
            if puissance_max > 1e4:
                liste_puissance_Cnm.append((freq_max,puissance_max,label_Cnm,spectre,puissance))
    return liste_puissance_Cnm
        

def recherche_simulation_interressante(chemin_fichier):
    data = pd.read_csv(chemin_fichier, sep='\s+', header = None)
    temps =  data.iloc[:,0]
    len_temps = len(temps)
    dt = temps.iloc[1] - temps.iloc[0]
    hanning = np.hanning(len_temps)
    freqs = np.fft.rfftfreq(len_temps,d=dt) * 2 * np.pi
    liste_puissance_Cnm = recherche_pics(data,freqs,hanning)
    
    if not liste_puissance_Cnm: # Sécurité au cas où la liste est vide
        return None
        
    mode_dominant = max(liste_puissance_Cnm, key=itemgetter(1))    #pour trouver le mode dominant
    
    for i,element in enumerate(liste_puissance_Cnm):
        if element[2] != mode_dominant[2]:
            if element[1] >= 0.2*mode_dominant[1] and element[0] != mode_dominant[0]:
                print(chemin_fichier)
                # Correction du guillemet manquant ici avant freq_max :
                print("chemin = ", chemin_fichier, "freq_max = ", element[0], "puissance_max = ", element[1], "label = ", element[2], "\n")
                return (chemin_fichier,liste_puissance_Cnm)
    return None


# --- MULTITHREADING ---

def traiter_dossier(racine):
    """Fonction wrapper pour exécuter la recherche sur un seul thread."""
    chemin_fichier = racine / nom_fichier_donnees
    if chemin_fichier.exists():
        return recherche_simulation_interressante(chemin_fichier)
    return None

if __name__ == '__main__':
    liste_chemins_dossier = library.registre_chemin_dossier("./inerP_A1em2_alpha0p7")
    liste_simulations_interressante = []

    # Détermine le nombre de threads optimal (généralement le nombre de coeurs de ton processeur)
    nb_threads = min(32, os.cpu_count() + 4)

    # Initialisation du ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=nb_threads) as executor:
        # On soumet toutes les tâches à l'exécuteur
        futures = {executor.submit(traiter_dossier, racine): racine for racine in liste_chemins_dossier}
        
        # On utilise tqdm combiné avec as_completed pour mettre à jour la barre de progression au fur et à mesure
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyse des simulations", unit="fichier"):
            resultat = future.result()
            # Si la fonction a retourné un tuple (et non None), on l'ajoute à la liste finale
            if resultat is not None:
                liste_simulations_interressante.append(resultat)

    print(f"\nTerminé ! {len(liste_simulations_interressante)} simulations intéressantes trouvées.")