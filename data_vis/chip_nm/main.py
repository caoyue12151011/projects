import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


# load data
nm, yr = np.loadtxt('data.txt').transpose()

xticks = yr 
xlabels = [int(i) for i in yr]
yticks = nm 
ylabels = []
for nm1 in nm:
    if nm1 >= 1000:
        ylabels.append(f'{nm1/1000:g}'+r'$\rm\mu m$')
    else:
        ylabels.append(f'{nm1:g}nm')


# demo
# halved every 4.2 yr, 1 order every 14 yr
plt.figure(figsize=(9,6))
plt.yscale('log')
ax = plt.gca()

plt.step(yr,nm,color='k',lw=2,where='post')
for ylabel, nm1, yr1 in zip(ylabels, nm, yr):
    plt.text(yr1+.1,1.05*nm1,ylabel,fontsize=14,rotation=0)

plt.grid()
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
plt.xticks(rotation=45)
# ax.set_yticks(yticks)
# ax.set_yticklabels(ylabels)
plt.xlabel('Year')
plt.ylabel('Gate width (nm)')
plt.tight_layout()
plt.savefig('chip_nm.pdf')
plt.show()





