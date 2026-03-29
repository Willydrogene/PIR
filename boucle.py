##############################################
### author: A. Astoul, mod. by..           ###                
### Plot periodogram from iner_P or T .tag ###
##############################################

# from astropy.timeseries import LombScargle
from numpy.fft import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from matplotlib import cm

# style
pgf_with_latex = { # setup matplotlib to use LaTeX for output
    # 'pgf.texsystem': 'pdflatex', # change this if using XeTeX or LuaTeX
    # 'text.usetex': True, # use LaTeX to write all text # often not working 
    'mathtext.fontset': 'stix',
    'xtick.top':True,
    'ytick.right':True,
    'xtick.direction':'in',
    'ytick.direction':'in',
    'axes.labelsize': 20, # x and y label fontsize; laTeX default is 10pt font                                                                  
    'font.size': 16,       # Legend title fontsize                                                                                              
    'legend.fontsize': 16, # Legend fontsize; make the legend/label fonts a little smaller                                                      
    'xtick.labelsize': 14, # values fontsize                                                                                                   
    'ytick.labelsize': 14,
    'lines.linewidth': 3
}
mpl.rcParams.update(pgf_with_latex)

# modes in inerPT.tag, sum l+m even or odd
even = {1:r'$(1,1)$',3:r'$(2,2)$',4:r'$(3,1)$',6:r'$(3,3)$',8:r'$(4,2)$',10:r'$(4,4)$',11:r'$(5,1)$',13:r'$(5,3)$',15:r'$(5,5)$',17:r'$(6,2)$',19:r'$(6,4)$',21:r'$(6,6)$'}
odd  = {2:r'$(2,1)$',5:r'$(3,2)$',7:r'$(4,1)$',9:r'$(4,3)$',12:r'$(5,2)$',14:r'$(5,4)$',16:r'$(6,1)$',18:r'$(6,3)$',20:r'$(6,5)$'}
tot  = {key: value for d in (even, odd) for key, value in d.items()}

# parameters to change
tab  = tot     # usually even for P, odd for T
pot  = 'P'     # poloidal or toroidal potential
pots = {'P':'W','T':'Z'}  
inif = '0p1'              # initial frequency (1 per simulation)
CT    = '1em2'            # tidal amplitude forcing
tag  = '_alpha0p7'        # shell aspect ratio

# select colors for != modes
n_colors = len(tab)
colours  = cm.nipy_spectral(np.linspace(0, 1, n_colors))

# directories and extension (to adapt)
nam  = 'om'+inif+'_A'+CT+tag+'/'
do   = {nam:['test']}              # other extensions exist
wdir = '/home/willy/code/PIR/inerP_A1em2_alpha0p7/' 

liste_fichiers = os.listdir(wdir)
#print(liste_fichiers)

ext_fichier = ['test', 'hiPhiRed', 'test_BIS', 'hiPhiRed_BIS']

for elem in liste_fichiers[0:3]:
    tagi = do[nam][0]
    try:
        fdom = open(wdir+elem+'/iner'+pot+'.'+ext_fichier[0],'r')
    except:
        try:
            fdom = open(wdir+elem+'/iner'+pot+'.'+ext_fichier[1],'r')
        except:
            try:
                fdom = open(wdir+elem+'/iner'+pot+'.'+ext_fichier[2],'r')
            except:
                try:
                    fdom = open(wdir+elem+'/iner'+pot+'.'+ext_fichier[3],'r')
                except:
                    print("J'peux plus rien faire pour toi")

    la = np.loadtxt(fdom)

    # sampling rate
    itend = len(la[:,0]) 
    t     = la[:itend,0]     # you might want to change itend (and istart) to analyse a smaller interval of the simu
    sr = len(t)
    dt = max(t)/sr

    # Window functions
    w1 = np.zeros(sr)
    for n in range(sr):
        w1[n] = 1./2-1./2*np.cos(2*np.pi*n/(sr-1))        # Hanning function

    # compute Fourier transform and frequencies
    freqs   = np.fft.fftfreq(sr, d=dt)*2*np.pi
    lim     = 1e4
    for i,lm in enumerate(tab):
        fft = np.fft.fft(la[:itend,lm]*w1)
        # select frequencies with highest power
        if np.any(abs(fft)>lim):
            plt.plot(freqs, abs(fft),label=tab[lm],color=colours[i])

    # optional: find frequencies associated with highest FFT values
    lmi = 3  # mode you want to look at
    fft = np.fft.fft(la[:itend,lmi]*w1)
    coeftab, freqtab = [], []
    for coef, freq in zip(fft, freqs):
        if freq>0.05 and freq<0.1:
            coeftab.append(coef)
            freqtab.append(freq)
    idx = np.where(coeftab == max(coeftab))[0][0]
    print(int(coeftab[idx]),freqtab[idx],tab[lmi])

    # # to adapt: identify frequency excited; clever ways can be used with arrows or embeddded numbers
    # omegas = [inif,inif/2]  # etc ...
    # ax     = plt.gca()
    # ylims  = ax.get_ylim()
    # for omi in omegas:
    # p4,    = plt.vlines(omega,ylims[0],ylims[1],'--',label=r'$\omega=0.2$')
    # # gather the legends
    # ps = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]
    # ls = []
    # for li in ps:
    #     ls.append(li.get_label())
    # plt.legend(ps,ls,loc='upper right',ncol=2)
    # plt.gca().add_artist(l0)

    # Plot Fourier analysis 
    plt.xlim([0,3])
    # plt.ylim([1.,4e6])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(5,5))
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\mathrm{power\ of\ '+pots[pot]+'}$')
    plt.legend(title=r'$(l,m)$')
    plt.show()
    # plt.savefig('fft_om'+inif+'_CT'+CT+tag+'_iner'+pot+'.pdf')

