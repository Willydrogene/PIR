# PIR
Projet d'Initiation à la Recherche

Branche MANU

Les commandes de bases:

###################################################### Au début ######################################################
git init // Moi je l'ai oublié et ça a fonctionné
git clone CLESSH // Bon si t'es là c'est que tu l'as déjà fait
git config --global user.name "Ton Nom" // Tu dois faire les deux sinon ça va pas capter la 1ere fois
git config --global user.email "ton@email.com"

######################################################## H24 #########################################################
git branch // Pour savoir dans quelle branche tu es
git branch "nom de branche" // Pour créer une branche, mais logiquement pas besoin vu que t'as manu
git switch manu // Très important, te permet de naviguer entre les branches

git pull // Pour récupérer les fichiers du distant, surtout si tu viens sur les branches des autres
git add fichier.py fichier2.py // Pour préparer ton colis (autant que tu veux)
git commit -m "J'écris une description" // Pour mettre une phrase obligatoire sur ton colis final
git push // (git push -u origin manu) la première fois pour qu'il capte le lien de la branche du local au git, pour envoyer le colis

git rm <fichier> // Pour supp un fichier du git, faut le add commit push après 
git log // Affiche l’historique des commits
git log --oneline // Version courte de l’historique

Après il y a des histoires de git fetch, git reset etc, mais je te laisse regarder avec chatgpt suivant ce que tu veux faire, parce qu'il y a plusieurs subtilités difficiles à expliquer. Peace and love ;)