from pathlib import Path

def registre_chemin_dossier(nom_chemin_dossier):
    chemin_racine = Path(nom_chemin_dossier) 
    
    # 1. On prépare notre "panier" (une liste vide)
    sous_dossiers = []

    # 2. On vérifie si le dossier existe avant de commencer
    if chemin_racine.exists() and chemin_racine.is_dir():
        
        # 3. On commence à regarder chaque élément (d) à l'intérieur
        for d in chemin_racine.iterdir():
            
            # 4. On pose la question : "Est-ce   que c'est un dossier ?"
            if d.is_dir() and not d.name.startswith("__") and not d.name.startswith("."):
                # 5. Si oui, on l'ajoute à notre panier
                sous_dossiers.append(d)
        sous_dossiers.sort()
    return sous_dossiers

