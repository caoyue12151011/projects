import numpy as np 
import matplotlib.pyplot as plt 
from collections import OrderedDict


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 'medium'
plt.rcParams['ytick.labelsize'] = 'medium'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'medium'


# load dynasty data 
f = open('docs/dynasty.txt','r')
f.readline()
f.readline()
data = f.readlines()
f.close()

Dynasty = OrderedDict()
for line in data:
    name, start, end = line.split()
    start = float(start)
    end = float(end)
    Dynasty[name] = {'start': start, 'end': end}


# load temperature data 
Yr_T, T = np.loadtxt('docs/Greenland_T.txt',skiprows=2).transpose()

# load population data 
Yr_Cp, Cp = np.loadtxt('docs/china_pop.txt',skiprows=2).transpose()
Yr_Ep, Ep = np.loadtxt('docs/euro_pop.txt',skiprows=2).transpose()


# demo -------------------------------------------------------------------------

plt.figure(figsize=(9,4.5))
ax1 = plt.gca()

# population
plt.scatter(Yr_Cp,Cp,fc='r',s=20,ec=None,label='China population')
plt.plot(Yr_Cp,Cp,lw=1,color='r')
plt.scatter(Yr_Ep,Ep,fc='blue',s=20,ec=None,label='Europe population')
plt.plot(Yr_Ep,Ep,lw=1,color='blue')

# China's dynasties
i = 0
for name in Dynasty:
    start = Dynasty[name]['start']
    end = Dynasty[name]['end']
    c = 'C'+str(i%10)
    plt.fill_between([start,end],1.45,1.6,fc=c,ec=None,alpha=.4)
    plt.text(start,1.65,name,color=c,weight='bold')
    i += 1

plt.legend()
plt.grid(axis='x')
plt.xlabel('Year')
plt.ylabel('Population (100 million)')

# temperature in new axis
ax2 = ax1.twinx()
plt.plot(Yr_T,T,lw=1,color='g',ls='--')
plt.plot(0,-29.2)

ax2.set_ylabel('Greenland temperature '+r'$\rm(^\circ C)$',color='green')
ax2.spines['right'].set_color('green')
ax2.tick_params(axis='y',colors='green')

plt.tight_layout()
plt.savefig('graph.pdf')
plt.close()















