import numpy as np 
import matplotlib.pyplot as plt 


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


# parameters
density_inf = np.pi/2/3**.5 # infinite hexagonal packing
Kind = ['square','circle','IRT','semicircle','quarter_circle']
Prop = ['density','loose']


# load data (from http://hydra.nat.uni-magdeburg.de/packing/)
Data = {}
for kind in Kind:
    Data[kind] = {}
    for prop in Prop:
        Data[kind][prop] = np.loadtxt('data/%s/%s.txt'%(kind,prop))[:,1]


# demo -------------------------------------------------------------------------

''' properties vs numbers
for prop in Prop:
    plt.figure(figsize=(7,4))
    plt.xscale('log')
    plt.yscale('log')

    i = 0
    for kind in Kind:
        data = Data[kind][prop]
        x = np.arange(len(data))+1
        if prop=='loose':
            data += 1
        plt.scatter(x,data,s=2,ec='none',fc='C%s'%i,alpha=.5,zorder=45)
        plt.scatter(-1,0,s=20,ec='none',fc='C%s'%i,alpha=.5,label=kind)
        i += 1

    if prop=='density':
        plt.axhline(density_inf,ls='--',lw=1,color='k',label='inf hexagonal')

    plt.legend()
    plt.grid()
    plt.xlabel('No. of circles')

    if prop=='loose':
        plt.ylabel(prop.capitalize()+'+1')
    else:
        plt.ylabel(prop.capitalize())

    plt.tight_layout()
    plt.savefig('image/%s.pdf'%prop)
    plt.close()
#'''






