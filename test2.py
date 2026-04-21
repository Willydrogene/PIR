import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from operator import itemgetter
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import library # Assurez-vous que ce module est accessible par les workers

# --- 1. CONFIGURATION (Identique) ---
pgf_with_latex = {
    'mathtext.fontset': 'stix',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.labelsize': 20,
    'font.size': 16,
}
mpl.rcParams.update(pgf_with_latex)

even = {1:r'$(1,1)$', 3:r'$(2,2)$', 4:r'$(3,1)$', 6:r'$(3,3)$', 8:r'$(4,2)$', 10:r'$(4,4)$', 11:r'$(5,1)$', 13:r'$(5,3)$', 15:r'$(5,5)$', 17:r'$(6,2)$', 19:r'$(6,4)$', 21:r'$(6,6)$'}
odd  = {2:r'$(2,1)$', 5:r'$(3,2)$', 7:r'$(4,1)$', 9:r'$(4,3)$', 12:r'$(5,2)$', 14:r'$(5,4)$', 16:r'$(6,1)$', 18:r'$(6,3)$', 20:r'$(6,5)$'}

pot = 'P' 
tab = even if pot == 'P' else odd
type_potentiel = 'P'



def analyseSimu(chemin_fichier):
    try:
        if not chemin_fichier.exists():
            return None
    
        """Fonction exécutée par chaque processeur"""
    #-----------------------------on extrait les données----------------------------------------
        df = pd.read_csv(chemin_fichier, sep=r'\s+', header=None)            #j'utilise panda qui prends d'une traite les colonnes
        data = df.values                                                    # Conversion en array NumPy pour la vitesse
        temps = data[:, 0]
        """dt_list = np.diff(temps)
        est_constant = np.allclose(dt_list, dt_list[0])"""
        dt = temps[1] - temps[0]                                            #je modifierais après que les codes sois harmonisé
        n_samples = len(temps)
        freqs = np.fft.rfftfreq(n_samples, d=dt) * 2 * np.pi                #on calcul les fréquences

        epsilon =  2 * np.pi/(np.max(temps)-np.min(temps))                       
        # Préparation de la fenêtre Hanning (vectorisée)
        hanning = np.hanning(n_samples)[:, np.newaxis]                      #vectorisation pour aller plus vite
        # Sélection des colonnes cibles présentes dans le fichier
        cols_presentes = [c for c in tab.keys() if c < data.shape[1]]
        # On calcule la FFT sur toutes les colonnes d'un coup (axis=0)
        subset_data = data[:, cols_presentes] * hanning                     #on applique la fenêtre de hanning sur les spectres
        spectres = np.fft.rfft(subset_data, axis=0)
        puissances = np.abs(spectres)                                       #on calcul la puissance

        #------------------------segment décisions--------------------------

        liste_puissance_Cnm = []                                            #ici je crée une liste dans laquel je retiens les Cnm suffisement élevé

        for i, col_idx in enumerate(cols_presentes):
            
            puissance_col = puissances[:,i]
            idx_max = np.argmax(puissance_col)
            p_max = puissance_col[idx_max]
            f_max = freqs[idx_max]
            
            if p_max > 1e4:

                label = tab[col_idx]                                        #le label c'est le char 'Cnm'
                liste_puissance_Cnm.append((f_max, p_max, label))           #je garde seulement les informations intéressente

        
            mode_dominant = max(liste_puissance_Cnm, key=itemgetter(1))         #on extrait le mode dominant qui va nous servir à la comparaison
            
            c1, c2 = 0, 0                                                       #c1 compte les modes qui sont superposé avec le pic de forçage
                                                                                #c2 compte les pics qui ont été sélectionné (suffisament fort)
            for element in liste_puissance_Cnm:                                 #je boucle pour analyse les caractéristique des spectre
                if element[1] >= 0.2 * mode_dominant[1]:    
                    c2 += 1
                    #print("c2 = ",c2,"\n")
                    if element[2] != mode_dominant[2]:
                        if abs(element[0] - mode_dominant[0]) < epsilon:
                            #print("df = ",df,"\n")
                            #print("diff = ",abs(element[0] - mode_dominant[0]),"\n")
                            c1 += 1
                            #print("c1 = ",c1,"\n")
            if c1 == c2:                                                       #si on a tout les pics qui sont superposé au pic de forçage, c'est un cas inintérressant
                #print("égale c1 = ",c1,"c2 = ",c2,"\n")
                return None
            if c1 != c2:
                #print("pas égale c1 = ",c1,"c2 = ",c2,"\n")
                return chemin_fichier
            
    except Exception as e:                                                      #fermeture du bloc try except
        return f"Erreur sur {chemin_fichier}:{e}"





repertoire = "./inerP_A5em2_alpha0p7"
liste_chemins_dossier = library.registre_chemin_dossier(str(repertoire))    #j'utilise un bibliothèque que j'ai crée pour créer une liste de chemins
liste_tag_fichier = ['test', 'hiPhiRes', 'test_BIS', 'hiPhiRes_BIS']
list_simu_inter = []

for racine in tqdm(liste_chemins_dossier):
    for tag in liste_tag_fichier:                                           #je parcours tout les tag existant, peut être qu'ici il y aune couille dans le paté?
        resSimu = analyseSimu(Path(racine)/f"iner{type_potentiel}.{tag}")           
        if resSimu != None:                                                 #conditions pour voir si la simu était bonne et on la stocke si c le cas
            list_simu_inter.append(resSimu)

print(f"Lancement de l'analyse sur {len(list_simu_inter)} fichiers...")
    


nom_csv = "test.csv"                                                           #tu met ici le nom du ficher CSV


df_resultats = pd.DataFrame(list_simu_inter, columns=['chemin_fichier'])      #je crée un fichier avec une colonne contenant les chemins
df_resultats.to_csv(nom_csv, index=False, sep=';', encoding='utf-8')          #on y sauvegarde les données

print(f"\nTerminé ! {len(list_simu_inter)} simulations trouvées.")
print(f"Fichier CSV créé : {nom_csv}")