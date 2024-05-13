'''
To derive the density profile of a isothermal, self-gravitating gas in 
equilibrium in the 1D, 2D, 3D cases. See knowledge/BE_gas.md for details.
'''

import num2tex
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# change default matplotlib fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['font.size'] = 14


def eq_3D(y,xi):
    ''' 
    Same as eq_den_profile_2D but for the spherically symmetric case.
    '''

    y0, y1 = y
    dy0 = y1
    dy1 = np.exp(-y0)-2*y1/xi

    return [dy0,dy1]


# global parameters 
ic = [0.,0.] # initial condition [psi,d(psi)/d(xi)] 


# demo =========================================================================

#''' density profile 

# parameters
Xi = np.logspace(-6,10,10000) # i.e. r/r_hat, Xi[0] is small (should be 0)  

# density profiles
Eta1 = 4*np.exp(2**.5*Xi)/(1+np.exp(2**.5*Xi))**2 # 1D BE sheet
Eta2 = (1+(Xi/2**1.5)**2)**-2 

Psi3 = odeint(eq_3D,ic,Xi)[:,0] # 3D BE sphere 
Eta3 = np.exp(-Psi3)
Eta3a = 1/(1+(Xi/2)**2) # approx. BE profile

# HWHM radius
xi_hwhm1 = np.log(3+2**1.5)/2**.5 
xi_hwhm2 = 2*(2*(2**.5-1))**.5
xi_hwhm3 = Xi[np.nanargmin((Eta3-.5)**2)]

print(xi_hwhm1,xi_hwhm2,xi_hwhm3)

# density profiles .............................................................

plt.figure(figsize=(5,4))
plt.plot(Xi,Eta1,color='r',label='BE sheet')
plt.plot(Xi,Eta2,color='g',label='BE cylinder')
plt.plot(Xi,Eta3,color='b',label='BE sphere')
plt.plot(Xi,Eta3a,color='b',linestyle='--',label='Approx. BE sphere')
plt.scatter(xi_hwhm1,.5,s=20,fc='w',ec='r',zorder=49)
plt.scatter(xi_hwhm2,.5,s=20,fc='w',ec='g',zorder=49)
plt.scatter(xi_hwhm3,.5,s=20,fc='w',ec='b',zorder=49)
plt.scatter(-1,0,s=20,fc='w',ec='k',zorder=49,label='FWHM')
plt.legend()
plt.grid()
plt.xlabel(r'$\xi=r\left(\frac{4\pi G\rho_0m}{k_{\rm B}T}\right)^{1/2}$')
plt.ylabel(r'$\rho/\rho_0$')
plt.tight_layout()

# save linear
plt.xlim(0,10)
plt.savefig('image/density_profile/linear.pdf')

# save log
plt.xlim(1e-1,1e3)
plt.ylim(1e-6,1.2)
plt.gca().get_legend().remove()
plt.xscale('log')
plt.yscale('log')
plt.savefig('image/density_profile/log.pdf')
plt.close()

# Eta3/Eta3a vs xi .............................................................

plt.figure()
plt.xscale('log')
plt.xlim(1e-1,1e7)
plt.plot(Xi,Eta3/Eta3a,color='b')
plt.grid()
plt.xlabel(r'$\xi=r\left(\frac{4\pi G\rho_0m}{k_{\rm B}T}\right)^{1/2}$')
plt.ylabel(r'$\eta_{\rho,\rm 3D}/\tilde\eta_{\rho,\rm 3D}$')
plt.tight_layout()
plt.savefig('image/density_profile/density_ratio.pdf')
plt.close()

# The oscillation pattern in Eta3  .............................................

text=r'$\sqrt{\xi}{\rm ln}\frac{2\eta_{\rho,\rm 3D}}{\tilde\eta_{\rho,\rm 3D}}$'

plt.figure()
plt.xscale('log')
plt.xlim(1e-3,1e7)
plt.ylim(-2,1.6)
plt.plot(Xi,np.log(2*Eta3/Eta3a)*Xi**.5,color='b',
    label=r'$\eta_{\rho,\rm 3D}$')
plt.plot(Xi,np.sin(7**.5/2*np.log(Xi)),color='k',
    label=r'$\tilde\eta_{\rho,\rm 3D,s}$')
plt.legend()
plt.grid()
plt.xlabel(r'$\xi=r\left(\frac{4\pi G\rho_0m}{k_{\rm B}T}\right)^{1/2}$')
plt.ylabel(text)
plt.tight_layout()
plt.savefig('image/density_profile/BE_sphere_oscillation.pdf')
plt.close()
#'''


