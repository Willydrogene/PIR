##############################################
# Auteur: A. Astoul (modifié et simplifié)
# Analyse FFT et plot du spectre de puissance
##############################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib import cm

# =========================
# CONFIGURATION PLOT
# =========================
plot_style = {
    'mathtext.fontset': 'stix',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.labelsize': 20,
    'font.size': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 3
}
mpl.rcParams.update(plot_style)

# =========================
# MODES (l, m)
# =========================
even_modes = {
    1: r'$(1,1)$', 3: r'$(2,2)$', 4: r'$(3,1)$', 6: r'$(3,3)$',
    8: r'$(4,2)$', 10: r'$(4,4)$', 11: r'$(5,1)$',
    13: r'$(5,3)$', 15: r'$(5,5)$', 17: r'$(6,2)$',
    19: r'$(6,4)$', 21: r'$(6,6)$'
}

odd_modes = {
    2: r'$(2,1)$', 5: r'$(3,2)$', 7: r'$(4,1)$',
    9: r'$(4,3)$', 12: r'$(5,2)$', 14: r'$(5,4)$',
    16: r'$(6,1)$', 18: r'$(6,3)$', 20: r'$(6,5)$'
}

# Fusion des modes
all_modes = {**even_modes, **odd_modes}

# =========================
# PARAMÈTRES UTILISATEUR
# =========================
selected_modes = all_modes   # choisir even_modes / odd_modes si besoin
potential_type = 'P'         # 'P' (poloidal) ou 'T' (toroidal)
potential_label = {'P': 'W', 'T': 'Z'}

initial_freq = '0p1'
forcing_amplitude = '1em2'
simulation_tag = '_alpha0p7'

# Dossier de travail
mode_A1 = "20pourcent_mode_A1/"
mode_A5 = "20pourcent_mode_A5/"
base_folder = f'/home/willy/code/PIR/{mode_A1}'
subfolder_name = f'om{initial_freq}_A{forcing_amplitude}{simulation_tag}/'

# Extensions possibles des fichiers
file_extensions = ['test', 'hiPhiRes', 'test_BIS', 'hiPhiRes_BIS']

# =========================
# COULEURS POUR LES MODES
# =========================
colors = cm.nipy_spectral(np.linspace(0, 1, len(selected_modes)))

# =========================
# FONCTION: fenêtre de Hanning
# =========================
def hanning_window(n_points):
    """Retourne une fenêtre de Hanning."""
    n = np.arange(n_points)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (n_points - 1))

# =========================
# BOUCLE PRINCIPALE
# =========================
folders = os.listdir(base_folder)

for folder in folders[:1]:  # limiter à 3 dossiers

    # =========================
    # CHARGEMENT DES DONNÉES
    # =========================
    data = None
    for ext in file_extensions:
        file_path = f"{base_folder}{folder}/iner{potential_type}.{ext}"
        try:
            data = np.loadtxt(file_path)
            break
        except OSError:
            continue

    if data is None:
        print(f"Impossible de charger les données pour {folder}")
        continue
    
    print(file_path)
    # =========================
    # PRÉPARATION TEMPS
    # =========================
    time = data[:, 0]
    n_samples = len(time)
    dt = max(time) / n_samples

    # Fenêtre de Hanning
    window = hanning_window(n_samples)

    # =========================
    # FFT
    # =========================
    frequencies = np.fft.fftfreq(n_samples, d=dt) * 2 * np.pi

    plt.figure(figsize=(13, 9))

    threshold = 1e4  # seuil pour afficher une courbe

    for i, mode_index in enumerate(selected_modes):
        signal = data[:, mode_index] * window
        fft_values = np.fft.fft(signal)

        # Afficher seulement si signal significatif
        if np.any(np.abs(fft_values) > threshold):
            print(mode_index)
            plt.plot(
                frequencies,
                np.abs(fft_values),
                label=selected_modes[mode_index],
                color=colors[i]
            )

    # =========================
    # DÉTECTION DES PICS
    # =========================
    peak_amp_values = []
    peak_freq_values = []
    max_peak = 0
    for mode_index in range(1, 22):
        signal = data[:, mode_index] * window
        fft_values = np.fft.fft(signal)

        max_index = np.argmax(np.abs(fft_values))
        peak_amp = np.abs(fft_values[max_index])
        peak_freq = frequencies[max_index]

        peak_amp_values.append(peak_amp)
        peak_freq_values.append(peak_freq)

        print(f"pic = {peak_amp:.2e}, w = {peak_freq:.3f}, mode = {selected_modes.get(mode_index, '?')}")

        if peak_amp > max_peak:
            max_peak = peak_amp
            indice_max_peak = mode_index

    print("#" * 70)

    print(f"pic maximum = {max_peak:.2e}")

    # =========================
    # AFFICHAGE FINAL
    # =========================
    delta = 0.1
    for i, peak in enumerate(peak_amp_values):
        if peak != max_peak:
            for a in range(-10, 10):
                if abs(peak_freq_values[i] - a*peak_freq_values[indice_max_peak]) <= delta:
                    print(f"w3 est bien une combinaison linéaire de w: {peak_freq_values[i]} = {a} * {peak_freq_values[indice_max_peak]}")
                    break
                if a!=0 and abs(peak_freq_values[i] - 1/a *peak_freq_values[indice_max_peak]) <= delta:
                    if a>0: print(f"w3 est bien une combinaison linéaire de w: {peak_freq_values[i]} = 1/{a}* { peak_freq_values[indice_max_peak]}")
                    else: print(f"w3 est bien une combinaison linéaire de w: {peak_freq_values[i]} = -1/{abs(a)}* { peak_freq_values[indice_max_peak]}")
                    break


    plt.xlim([0, 3])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
    plt.xlabel(r'$\omega$')
    plt.ylabel(rf'$\mathrm{{power\ of\ {potential_label[potential_type]}}}$')
    plt.legend(title=r'$(l,m)$')

    plt.show()

    print("#" * 70 + "\n")