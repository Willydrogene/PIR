import numpy as np
import pandas as pd
import matplotlib as mpl
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

# --- 2. FONCTIONS DE CALCUL OPTIMISÉES ---

def analyser_un_fichier(chemin_fichier):
    """Fonction exécutée par chaque processeur"""
    try:
        if not chemin_fichier.exists():
            return None

        # Lecture rapide avec NumPy ou Pandas (engine='c' est très performant)
        df = pd.read_csv(chemin_fichier, sep='\s+', header=None)
        data = df.values # Conversion en array NumPy pour la vitesse
        
        temps = data[:, 0]
        dt = temps[1] - temps[0]
        n_samples = len(temps)
        df =  2 * np.pi/(np.max(temps)-np.min(temps))
        
        # Préparation de la fenêtre Hanning (vectorisée)
        hanning = np.hanning(n_samples)[:, np.newaxis]
        
        # Sélection des colonnes cibles présentes dans le fichier
        cols_presentes = [c for c in tab.keys() if c < data.shape[1]]
        if not cols_presentes:
            return None
            
        # --- OPTIMISATION NUMPY ---
        # On calcule la FFT sur toutes les colonnes d'un coup (axis=0)
        subset_data = data[:, cols_presentes] * hanning
        spectres = np.fft.rfft(subset_data, axis=0)
        puissances = np.abs(spectres)
        freqs = np.fft.rfftfreq(n_samples, d=dt) * 2 * np.pi
        
        liste_puissance_Cnm = []
        for i, col_idx in enumerate(cols_presentes):
            puissance_col = puissances[:,i]
            idx_max = np.argmax(puissance_col)
            p_max = puissance_col[idx_max]
            f_max = freqs[idx_max]
            
            if p_max > 1e4:
                label = tab[col_idx]
                # On ne garde que les infos nécessaires pour le filtrage
                liste_puissance_Cnm.append((f_max, p_max, label))

        if not liste_puissance_Cnm:
            return None

        # Logique de sélection "intéressante"
        mode_dominant = max(liste_puissance_Cnm, key=itemgetter(1))
        
        c1, c2 = 0, 0
        for element in liste_puissance_Cnm:       
            if element[1] >= 0.2 * mode_dominant[1]:    
                c2 += 1
                #print("c2 = ",c2,"\n")
                if element[2] != mode_dominant[2]:
                    if abs(element[0] - mode_dominant[0]) < df:
                        #print("df = ",df,"\n")
                        #print("diff = ",abs(element[0] - mode_dominant[0]),"\n")
                        c1 += 1
                        #print("c1 = ",c1,"\n")
        if c1 == c2:
            #print("égale c1 = ",c1,"c2 = ",c2,"\n")
            return None
        if c1 != c2:
            #print("pas égale c1 = ",c1,"c2 = ",c2,"\n")
            return chemin_fichier

    except Exception as e:
        return f"Erreur sur {chemin_fichier}: {e}"

# --- 3. EXÉCUTION PARALLÈLE ---

if __name__ == '__main__':
    repertoire = Path("./inerP_A5em2_alpha0p7")
    liste_chemins_dossier = library.registre_chemin_dossier(str(repertoire))
    liste_tag_fichier = ['test', 'hiPhiRes', 'test_BIS', 'hiPhiRes_BIS']
    
    tous_les_fichiers = []
    for racine in liste_chemins_dossier:
        for tag in liste_tag_fichier:
            tous_les_fichiers.append(Path(racine) / f"iner{type_potentiel}.{tag}")

    print(f"Lancement de l'analyse sur {len(tous_les_fichiers)} fichiers...")
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        list_map = list(tqdm(executor.map(analyser_un_fichier, tous_les_fichiers), total=len(tous_les_fichiers),desc="Analyse"))

    # 1. Filtrage pour ne garder que les chemins valides
    liste_simulations_interressante = [res for res in list_map if res is not None]

    # 2. SAUVEGARDE EN CSV
    nom_csv = "resultats_simulationspool2.csv"
    
    # On crée un DataFrame avec une colonne 'chemin'
    df_resultats = pd.DataFrame(liste_simulations_interressante, columns=['chemin_fichier'])
    
    # Sauvegarde (index=False permet de ne pas écrire la colonne d'index 0, 1, 2...)
    df_resultats.to_csv(nom_csv, index=False, sep=';', encoding='utf-8')

    print(f"\nTerminé ! {len(liste_simulations_interressante)} simulations trouvées.")
    print(f"Fichier CSV créé : {nom_csv}")