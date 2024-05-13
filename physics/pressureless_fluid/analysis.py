# To calculate the flow of a fluid under gravity.

import numpy as np 
import matplotlib.pyplot as plt 
import astropy.units as u 
import astropy.constants as ct


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


''' spherical collapse to a central mass, pressureless, self-G ----------------

# dimensionless distance of a free-fall point mass to a gravitational center
f = np.linspace(0,1,500) 

term = (1/f-1)**.5
tau = 2/np.pi*(f*term + np.arctan(term)) # dimensionless time
mu = -np.pi/2/f*term # dimensionless velocity


plt.figure(figsize=(4,3.5))
plt.plot(tau,f,lw=1,color='k')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$f(\tau)$')
plt.grid()
plt.tight_layout()
plt.savefig('sphere_collapse/f_vs_tau.pdf')
plt.close()


plt.figure(figsize=(4,3.5))
plt.ylim(-.1,5)
plt.plot(tau,-mu,lw=1,color='k')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$-mu(\tau)/xi$') 
plt.grid()
plt.tight_layout()
plt.savefig('sphere_collapse/mu_vs_tau.pdf')
plt.close()
#'''


#''' hyperbolic scheme of Bondi-Hoyle accretion (pressureless, no self-G) ------

# some constants
A_y0 = 2*ct.G*u.M_sun*u.km**-2*u.s**2
print(A_y0.to(u.AU))
A_dm = 4*np.pi*(ct.G*u.M_sun)**2*2.8*ct.m_n*1e6/u.cm**3/(u.km/u.s)**3
print(A_dm.to(u.M_sun/u.yr))











#'''








