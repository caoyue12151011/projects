import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.append('/Users/yuecao/Documents/coding/module')
import phunction 


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


m = np.logspace(np.log10(.08), np.log10(120), 300)

R = phunction.m2R_MS(m)
L = phunction.m2L_MS(m)
T = phunction.m2T_MS(m)

''' mass vs radius 
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(m, R, color='k')
plt.grid()
plt.xlabel(r'$M_{\rm main\ sequence}\ (M_\odot)$')
plt.ylabel(r'$R_{\rm main\ sequence}\ (R_\odot)$')
plt.tight_layout()
plt.savefig('image/relation/m_vs_R.pdf')
plt.close()
#'''

''' mass vs luminosity 
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(m, L, color='k')
plt.grid()
plt.xlabel(r'$M_{\rm main\ sequence}\ (M_\odot)$')
plt.ylabel(r'$L_{\rm main\ sequence}\ (L_\odot)$')
plt.tight_layout()
plt.savefig('image/relation/m_vs_L.pdf')
plt.close()
#'''

''' mass vs temperature 
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(m, T, color='k')
plt.grid()
plt.xlabel(r'$M_{\rm main\ sequence}\ (M_\odot)$')
plt.ylabel(r'$T_{\rm main\ sequence}\ (\rm K)$')
plt.tight_layout()
plt.savefig('image/relation/m_vs_T.pdf')
plt.close()
#'''

