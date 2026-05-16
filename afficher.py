import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Bibliothèque pour la barre de chargement

# --- 1. CONFIGURATION DU STYLE ---
mpl.use('Agg') # Backend non-interactif pour éviter les conflits en parallèle
pgf_with_latex = {
    'mathtext.fontset': 'stix',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.labelsize': 20,
    'font.size': 16,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5
}
mpl.rcParams.update(pgf_with_latex)

# Définition globale des modes
EVEN = {1:r'$(1,1)$', 3:r'$(2,2)$', 4:r'$(3,1)$', 6:r'$(3,3)$', 8:r'$(4,2)$', 10:r'$(4,4)$', 11:r'$(5,1)$', 13:r'$(5,3)$', 15:r'$(5,5)$', 17:r'$(6,2)$', 19:r'$(6,4)$', 21:r'$(6,6)$'}
ODD  = {2:r'$(2,1)$', 5:r'$(3,2)$', 7:r'$(4,1)$', 9:r'$(4,3)$', 12:r'$(5,2)$', 14:r'$(5,4)$', 16:r'$(6,1)$', 18:r'$(6,3)$', 20:r'$(6,5)$'}
TAB_MODES = {**EVEN, **ODD}
POT_LABEL = {'P': 'W', 'T': 'Z'}
OUTPUT_DIR = Path("./test")

# --- 2. FONCTION DE TRAITEMENT UNITAIRE ---

def traiter_fichier(chemin_str):
    """Fonction qui sera distribuée sur les cœurs du processeur."""
    try:
        chemin = Path(chemin_str)
        if not chemin.exists(): return None

        # Chargement et calculs
        data = pd.read_csv(chemin, sep='\s+', header=None).values
        t = data[:, 0]
        sr = len(t)
        dt = t[1] - t[0]
        w1 = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(sr) / (sr - 1))
        freqs = np.fft.fftfreq(sr, d=dt) * 2 * np.pi

        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 8))
        colours = cm.nipy_spectral(np.linspace(0, 1, len(TAB_MODES)))

        for i, (lm_idx, label) in enumerate(sorted(TAB_MODES.items())):
            if lm_idx < data.shape[1]:
                amp = np.abs(np.fft.fft(data[:, lm_idx] * w1))
                if np.max(amp) > 1e4: # Filtre de bruit
                    ax.plot(freqs, amp, label=label, color=colours[i], alpha=0.8)

        # Style
        ax.set_xlim([0, 3])
        ax.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$\mathrm{power\ of\ P}$')
        ax.set_title(f"Simulation: {chemin.parent.name}")
        ax.legend(loc='upper right', ncol=3, fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Sauvegarde
        nom_img = f"{chemin.parent.name}_{chemin.suffix[1:]}.png"
        fig.savefig(OUTPUT_DIR / nom_img, dpi=150)
        plt.close(fig) # Libère la RAM
        return True
    except Exception as e:
        return f"Erreur {chemin_str}: {e}"

# --- 3. EXÉCUTION MULTI-PROCESSUS ---

if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Récupération des chemins depuis ton CSV
    df = pd.read_csv("test.csv", sep=';')
    chemins = df['chemin_fichier'].tolist()

    print(f"🚀 Lancement du traitement de {len(chemins)} fichiers...")

    # Utilisation de ProcessPoolExecutor pour diviser le travail
    with ProcessPoolExecutor() as executor:
        # tqdm encapsule l'itérateur executor.map pour afficher la barre
        list(tqdm(
            executor.map(traiter_fichier, chemins), 
            total=len(chemins), 
            unit="plot",
            desc="Progression"
        ))

    print(f"\n✅ Terminé ! Les images sont ici : {OUTPUT_DIR.absolute()}")