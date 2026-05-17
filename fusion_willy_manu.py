import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from astropy.timeseries import LombScargle
from pathlib import Path
from scipy.signal import find_peaks


# ====================================================================== #
# OPTIONS TERMINAL
# ====================================================================== #

AFFICHER_PICS = False
AFFICHER_COMBINAISONS = True
AFFICHER_PICS_SECONDAIRES = True


# ====================================================================== #
# STYLE DES GRAPHIQUES
# ====================================================================== #

pgf_with_latex = {
    'mathtext.fontset': 'stix',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'font.size': 16,
    'axes.labelsize': 20,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 1.5
}
mpl.rcParams.update(pgf_with_latex)


# ====================================================================== #
# MODES (l,m)
# ====================================================================== #

pair = {
    1: r'$(1,1)$', 3: r'$(2,2)$', 4: r'$(3,1)$', 6: r'$(3,3)$',
    8: r'$(4,2)$', 10: r'$(4,4)$', 11: r'$(5,1)$',
    13: r'$(5,3)$', 15: r'$(5,5)$', 17: r'$(6,2)$',
    19: r'$(6,4)$', 21: r'$(6,6)$'
}

impair = {
    2: r'$(2,1)$', 5: r'$(3,2)$', 7: r'$(4,1)$',
    9: r'$(4,3)$', 12: r'$(5,2)$', 14: r'$(5,4)$',
    16: r'$(6,1)$', 18: r'$(6,3)$', 20: r'$(6,5)$'
}

tot = {**pair, **impair}
tab = dict(sorted(tot.items()))


# ====================================================================== #
# COULEURS
# ====================================================================== #

palette = plt.get_cmap('tab20_r', 20).colors
couleurs = ['black', *palette]


# ====================================================================== #
# PARAMÈTRES UTILISATEUR
# ====================================================================== #

omega_str = 'm1p96' #0.1, 0.9, 0.14, -1.24(absolument rien), -0.62, -0.66, -1.64(découverte)
alpha_str = 'alpha0p7'

potential_type = 'P'  # P ou T
potential_label = {'P': 'W', 'T': 'Z'}

omega = float(omega_str.replace('m', '-').replace('p', '.'))

num_A1 = '1'
num_A5 = '5'

Ct_A1 = num_A1 + 'em2'
Ct_A5 = num_A5 + 'em2'

dossier_A1 = Path(
    f'/home/willy/code/PIR/inerP_A{num_A1}em2_{alpha_str}/'
    f'om{omega_str}_A{Ct_A1}_{alpha_str}/'
)

dossier_A5 = Path(
    f'/home/willy/code/PIR/inerP_A{num_A5}em2_{alpha_str}/'
    f'om{omega_str}_A{Ct_A5}_{alpha_str}/'
)


# ====================================================================== #
# PARAMÈTRES LOMB-SCARGLE
# ====================================================================== #

f_min = 0.001
f_max = 3.0 / (2 * np.pi)
freq_hz = np.linspace(f_min, f_max, 5000)
freq_omega = freq_hz * 2 * np.pi

lim = 0.02


# ====================================================================== #
# FONCTIONS
# ====================================================================== #

def lire_fichier(dossier, potential_type='P'):
    """
    Lit automatiquement un fichier inerP.* ou inerT.*.
    """

    fichiers_possibles = [
        'test',
        'hiPhiRes',
        'test_BIS',
        'hiPhiRes_BIS',
        'hiSpaTimRes'
    ]

    for ext in fichiers_possibles:
        fichier = dossier / f'iner{potential_type}.{ext}'
        if fichier.exists():
            data = np.loadtxt(fichier)
            print(f"Fichier lu : {fichier}")
            return data, fichier

    for fichier in dossier.glob(f'iner{potential_type}.*'):
        if ':Zone.Identifier' not in fichier.name:
            data = np.loadtxt(fichier)
            print(f"Fichier lu : {fichier}")
            return data, fichier

    raise FileNotFoundError(f"Aucun fichier iner{potential_type}.* trouvé dans {dossier}")


def fenetre_hanning(n):
    """
    Fenêtre de Hanning.
    """
    indices = np.arange(n)
    return 0.5 - 0.5 * np.cos((2 * np.pi * indices) / (n - 1))


def afficher_pics_secondaires(res, n_pics=3, distance=30, hauteur_relative=0.03):
    """
    Affiche les n plus grands pics des modes visibles,
    avec le décalage par rapport au pic principal.
    """

    print("\n" + "#" * 70)
    print(f"Pics secondaires pour {res['titre']}")
    print("#" * 70)

    for i, lm in res["modes_retenus"]:

        amp = res["spectres"][lm]
        amp_norm = amp / res["max_global"]

        peaks, props = find_peaks(
            amp_norm,
            height=hauteur_relative,
            distance=distance
        )

        if len(peaks) == 0:
            continue

        # On trie les pics par amplitude décroissante
        peaks_tries = sorted(
            peaks,
            key=lambda k: amp_norm[k],
            reverse=True
        )

        peaks_tries = peaks_tries[:n_pics]

        w_principal = freq_omega[peaks_tries[0]]

        print(f"\nMode {tab[lm]}")
        print(f"  pic principal : w = {w_principal:.4f}")

        for k in peaks_tries[1:]:

            w_pic = freq_omega[k]
            delta_w = w_pic - w_principal

            print(
                f"  pic secondaire : w = {w_pic:.4f} "
                f"| delta = {delta_w:+.4f}"
            )


def afficher_combinaisons_lineaires(freqs, amps, indice_ref, modes, titre, modes_retenus, delta=0.005):
    """
    Affiche les combinaisons linéaires uniquement pour les modes visibles.
    """

    modes_visibles = [lm for _, lm in modes_retenus]

    freq_ref = freqs[indice_ref]
    mode_ref = modes[indice_ref]

    print("\n" + "#" * 70)
    print(f"Combinaisons linéaires pour {titre}")
    print(f"Mode dominant : {tab[mode_ref]} (w = {freq_ref:.3f})")
    print("#" * 70)

    for i, freq in enumerate(freqs):

        mode_test = modes[i]

        if mode_test not in modes_visibles:
            continue

        if i == indice_ref:
            continue

        for a in range(-10, 11):

            if a == 0:
                continue

            if abs(freq - a * freq_ref) <= delta:
                print(
                    f"{tab[mode_test]} : "
                    f"w = {freq:.3f} ≈ {a} × "
                    f"w{tab[mode_ref]} = {freq_ref:.3f}"
                )
                break

            if abs(freq - (1 / a) * freq_ref) <= delta:
                print(
                    f"{tab[mode_test]} : "
                    f"w = {freq:.3f} ≈ 1/{a} × "
                    f"w{tab[mode_ref]} = {freq_ref:.3f}"
                )
                break


def analyser_simulation(dossier, titre):
    """
    Calcule les spectres Lomb-Scargle et garde les infos utiles.
    """

    data, fichier = lire_fichier(dossier, potential_type)

    itend = len(data[:, 0])
    t = data[:itend, 0]
    sr = len(t)

    window = fenetre_hanning(sr)

    spectres = {}
    max_global = 0.0

    peak_amp_values = []
    peak_freq_values = []

    liste_modes = []

    for i, lm in enumerate(tab):

        signal_fenetre = data[:itend, lm] * window

        ls = LombScargle(t, signal_fenetre)
        puissance = ls.power(freq_hz, normalization='psd')
        amp = np.sqrt(puissance)

        spectres[lm] = amp

        max_mode = np.max(amp)

        if max_mode > max_global:
            max_global = max_mode

        indice_pic = np.argmax(amp)
        peak_amp = amp[indice_pic]
        peak_freq = freq_omega[indice_pic]

        peak_amp_values.append(peak_amp)
        peak_freq_values.append(peak_freq)
        liste_modes.append(lm)

        if AFFICHER_PICS:
            print(
                f"{titre} | pic = {peak_amp:.2e}, "
                f"w = {peak_freq:.3f}, mode = {tab[lm]}"
            )

    indice_max_peak = np.argmax(peak_amp_values)

    if AFFICHER_PICS:
        print("#" * 70)
        print(
            f"{titre} | pic maximum = "
            f"{peak_amp_values[indice_max_peak]:.2e}, "
            f"w = {peak_freq_values[indice_max_peak]:.3f}"
        )

    # if AFFICHER_COMBINAISONS:
    #     afficher_combinaisons_lineaires(
    #         peak_freq_values,
    #         peak_amp_values,
    #         indice_max_peak,
    #         liste_modes,
    #         titre
    #     )

    return {
        "titre": titre,
        "data": data,
        "t": t,
        "spectres": spectres,
        "max_global": max_global,
        "modes_retenus": [],
        "peak_freq_values": peak_freq_values,
        "peak_amp_values": peak_amp_values,
        "indice_max_peak": indice_max_peak,
        "liste_modes": liste_modes
    }


# ====================================================================== #
# ANALYSE A1 ET A5
# ====================================================================== #

res_A1 = analyser_simulation(dossier_A1, titre='A1')
res_A5 = analyser_simulation(dossier_A5, titre='A5')

resultats = [res_A1, res_A5]


# ====================================================================== #
# GRAPHE 1 : SPECTRES FRÉQUENTIELS A1 / A5
# ====================================================================== #

fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

for ax, res in zip(axes, resultats):

    modes_retenus = []

    for i, (lm, amp) in enumerate(res["spectres"].items()):

        amp_normalisee = amp / res["max_global"]

        if np.any(amp_normalisee > lim):

            ax.plot(
                freq_omega,
                amp_normalisee,
                label=tab[lm],
                color=couleurs[i]
            )

            modes_retenus.append((i, lm))

    res["modes_retenus"] = modes_retenus
    
    if AFFICHER_COMBINAISONS:
        afficher_combinaisons_lineaires(
            res["peak_freq_values"],
            res["peak_amp_values"],
            res["indice_max_peak"],
            res["liste_modes"],
            res["titre"],
            res["modes_retenus"]
        )

    for facteur in [4, 3, 2, 1, 1 / 2, 1 / 3, 1 / 4]:
        ax.axvline(
            x=abs(omega * facteur),
            color='lightgray',
            alpha=0.7,
            linestyle='--',
            zorder=50
        )

    ax.set_xlim([0, 2])
    ax.set_ylim([0, 1])
    ax.set_title(fr'{res["titre"]} : $\omega = {omega}$')
    ax.set_ylabel(rf'$\mathrm{{power\ of\ {potential_label[potential_type]}}}$')
    ax.legend(title=r'$(l,m)$', fontsize=10, ncol=2, loc='upper right')

if AFFICHER_PICS_SECONDAIRES:
    afficher_pics_secondaires(res_A1)
    afficher_pics_secondaires(res_A5)

axes[-1].set_xlabel(r'$\omega$')

plt.tight_layout()


# ====================================================================== #
# GRAPHE 2 : TEMPOREL A1 / A5
# ====================================================================== #

fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

for ax, res in zip(axes, resultats):

    data = res["data"]
    t = res["t"]

    for i, lm in res["modes_retenus"]:

        serie_temporelle = data[1:, lm]

        # Petite sécurité contre log(0)
        log_temporel = np.log(np.abs(serie_temporelle) + 1e-300)

        ax.plot(
            t[1:],
            log_temporel,
            label=tab[lm],
            linewidth=1,
            color=couleurs[i]
        )

    ax.set_title(fr'{res["titre"]} : $\omega = {omega}$')
    ax.set_ylabel(rf'$\log({potential_label[potential_type]})$')
    ax.legend(title=r'$(l,m)$', fontsize=10, ncol=2, loc='upper right')

axes[-1].set_xlabel(r'$t$')

plt.tight_layout()


# ====================================================================== #
# AFFICHAGE FINAL
# ====================================================================== #

plt.show()