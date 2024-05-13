# To calculate the time needed for a point to climb on top of a circle.

import numpy as np 
import astropy.units as u 
import matplotlib.pyplot as plt 


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


# parameters
R = .5 # [m]
g = 9.8 # [m/s^2]
c_u = [1*u.fm,1*u.pm,1*u.nm,1*u.um,1*u.mm,1*u.cm,1*u.dm] # arc length

c_s = [''.join(i.to_string(precision=0).split('. ')) for i in c_u]
c = np.array([i.to_value(u.m) for i in c_u]) # [m]
t = (R/g)**.5*np.log(1/np.sin(c/2/R) + 1/np.tan(c/2/R)) # [s]


text = r'$R=%s{\rm\ m}$'%R+'\n'+r'$g=%s{\rm\ m\ s^{-2}}$'%g

plt.figure(figsize=(6,4))
plt.xscale('log')
plt.scatter(c,t,color='k',s=30)
ax = plt.gca()
plt.text(.05,.05,text,fontsize=12,transform=ax.transAxes)
plt.grid()
ax.xaxis.set_ticks(c)
ax.xaxis.set_ticklabels(c_s)
plt.ylabel('Time (s)')
plt.tight_layout()
plt.savefig('time_vs_arc_length.pdf')
plt.close()



