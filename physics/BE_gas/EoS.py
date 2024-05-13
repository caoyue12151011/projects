'''
To derive the EoS of an ideal, isothermal, self-gravitating gas in the 
equilibrium. See knowledge/BE_gas.md for details. See also Bonnor 1956, 
"Boyle's Law and Gravitational Instability".
'''

import sys
import scipy
import num2tex
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import odeint
from scipy.interpolate import interp1d

sys.path.append('/Users/yuecao/Documents/issues/coding/module')
from clausen import clausen



# change default matplotlib fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['font.size'] = 14


def Theta_func(x):
    # A function used in the p-V curve of the approximated BE gas.

    A = x - (.5*np.log((x**2+1)/4) + 1)*np.arctan(x) 
    B = np.array([.5*clausen(2*np.arctan(xx)-np.pi) for xx in x])

    return (A+B)/(x-np.arctan(x))


def eq_3D(y,xi):
    ''' 
    Same as eq_den_profile_2D but for the spherically symmetric case.
    '''

    y0, y1 = y
    dy0 = y1
    dy1 = np.exp(-y0)-2*y1/xi

    return [dy0,dy1]


def cumtrapz(y,x):
    # To calculate the cumulative trapezoidal integral. y, x are 1d arrays. 
    # Returns the integral as 1d array

    dx = np.gradient(x)
    dy = np.gradient(y)

    return np.cumsum(.5*(2*y+dy)*dx)


def eq_PI(Xi,Psi):
    # The PI function in the EoS. Xi and Psi are 1D arrays and Xi[0] must be 
    # small. Returns 1D array of PI.

    I1 = cumtrapz(Xi**2*np.exp(-Psi),Xi)
    I2 = cumtrapz(Xi*np.exp(-Psi)*I1,Xi)

    return I2/I1/3



# global parameters 
ic = [0.,0.] # initial condition [psi,d(psi)/d(xi)] 


# demo =========================================================================

#''' p-V curve of BE spheres ---------------------------------------------------

# parameters
Xi = np.logspace(-4,6,10000) # i.e. r/r_hat, Xi0[0] is small (should be 0)  


# solve the differential equation (real BE gas)
Psi = odeint(eq_3D,ic,Xi)[:,0]
dPsi = np.gradient(Psi)/np.gradient(Xi)

# p-V ..........................................................................

# parameters
lg_xi_bar = np.array([-1,0,1,2,3]) # ticks of the colorbar 

xi_bar = 10**lg_xi_bar.astype(float)
xi_min, xi_max = xi_bar.min(), xi_bar.max() 


# color values
Color = np.log10(Xi/xi_min)/np.log10(xi_max/xi_min)
Color[Color<0] = 0.
Color[Color>1] = 1.


# normed p and V of BE sphere
V_BE = (Xi*dPsi)**-3/3
p_BE = Xi**4*dPsi**2*np.exp(-Psi)

# ieal gas without self-G
p_nG = 1/V_BE 

# approx. BE gas
p_aBE = 64*(Xi/2-np.arctan(Xi/2))**2/(1+(Xi/2)**2)
V_aBE = (1-4/3*Theta_func(Xi/2))/p_aBE

# approx. spiral BE gas
theta_p = np.arcsin(7**.5/4)
theta_v = np.arcsin(14**.5/4)
p_sBE = 8*(1 + 2/Xi**.5*np.sin(7**.5/2*np.log(Xi)-theta_p))
V_sBE = 1/24*(1 - 3/2**.5/Xi**.5*np.sin(7**.5/2*np.log(Xi)-theta_v))
p_sBE = p_sBE[Xi>=1] # truncate p_sBE & V_sBE
V_sBE = V_sBE[Xi>=1]

# integrated form of p-V curve (it has cumulative errors!)
PI = eq_PI(Xi,Psi)
p_BE2 = np.exp(-Psi)*np.cumsum(Xi**2*np.exp(-Psi)*np.gradient(Xi))**2 
V_BE2 = (1-PI)/p_BE2


# p-V 
plt.figure(figsize=(6,4.5))
plt.xlim(0,.3)
plt.ylim(0,1.1*np.nanmax(p_BE))
ax = plt.gca()

# other solutions
lw = 1
plt.plot(V_BE,p_nG,color='k',label='Ideal gas',linewidth=lw,linestyle='--')
# plt.plot(V_BE2,p_BE2,color='r',label='BE sphere2',linewidth=lw,
    # linestyle='--')
plt.plot(V_aBE,p_aBE,color='b',label='Approx. BE solution',linewidth=lw,
    linestyle='--')
plt.plot(V_sBE,p_sBE,color='r',label='Approx. BE spiral',linewidth=lw,
    linestyle='--')
plt.scatter(1/24,8,fc='none',ec='k',marker='o',s=40,label='Singular solution',
    zorder=49)

# The curve
points = np.array([V_BE,p_BE]).T.reshape(-1,1,2)
segments = np.concatenate([points[:-1],points[1:]],axis=1)
lc = LineCollection(segments,cmap='plasma',linewidth=2)
lc.set_array(Color)
# lc.set_zorder(49)
line = ax.add_collection(lc)

# curve label
plt.plot(-1,-1,color=matplotlib.cm.get_cmap('plasma')(.5),linewidth=2,
    label='Exact BE sphere')

# curve colorbar
ticks = (lg_xi_bar-lg_xi_bar.min())/(lg_xi_bar.max()-lg_xi_bar.min())
labels = [r'$10^{%d}$'%lg_xi for lg_xi in lg_xi_bar]
cb = plt.colorbar(line)
cb.set_ticks(ticks)
cb.set_ticklabels(labels)
cb.set_label(r'$\xi$')

plt.legend()
plt.grid()
plt.xlabel(r'$\hat{V}=V\frac{(Nk_{\rm B}T)^3}{4\pi G^3M^6}$')
plt.ylabel(r'$\hat{p}=p\frac{4\pi G^3M^6}{(Nk_{\rm B}T)^4}$')
plt.tight_layout()
plt.savefig('image/EoS/p_V.pdf')
plt.close()


# PI vs xi
plt.figure()
plt.xscale('log')
plt.plot(Xi,PI,color='k')
plt.axhline(2/3,color='k',linestyle='--')
plt.text(Xi.min(),.7,r'$\frac{2}{3}$',fontsize=20)
plt.grid()
plt.xlabel(r'$\xi=r\left(\frac{4\pi G\rho_0m}{k_{\rm B}T}\right)^{1/2}$')
plt.ylabel(r'$\Pi(\xi)$')
plt.tight_layout()
plt.savefig('image/EoS/PI_xi.pdf')
plt.close()

# find max p_BE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

Xi_f = np.linspace(5,7,10000) # finer Xi 

# find max p_BE
p_BE_f = interp1d(Xi,p_BE,'cubic')(Xi_f)
ind = np.argmax(p_BE_f)
print('Max p_BE=%s, xi=%s.'%(p_BE_f[ind],Xi_f[ind]))

# check
# plt.figure()
# plt.xscale('log')
# plt.scatter(Xi,p_BE,s=20)
# plt.scatter(Xi_f,p_BE_f,s=1)
# plt.show()

# find min V_BE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

Xi_f = np.linspace(22,23,10000) # finer Xi 

# find 
V_BE_f = interp1d(Xi,V_BE,'cubic')(Xi_f)
ind = np.argmin(V_BE_f)
print('Min V_BE=%s, xi=%s.'%(V_BE_f[ind],Xi_f[ind]))

# check
# plt.figure()
# plt.xscale('log')
# plt.yscale('log')
# plt.scatter(Xi,V_BE,s=20)
# plt.scatter(Xi_f,V_BE_f,s=1)
# plt.show()

# V-M ..........................................................................

# parameters
lg_xi_bar = np.array([-1,0,1,2,3]) # ticks of the colorbar 

xi_bar = 10**lg_xi_bar.astype(float)
xi_min, xi_max = xi_bar.min(), xi_bar.max() 


# normed V, M
V_BE = Xi/dPsi/np.exp(Psi)
M_BE = Xi**(2/3)*dPsi**(1/3)/np.exp(Psi/6)

# color values
Color = np.log10(Xi/xi_min)/np.log10(xi_max/xi_min)
Color[Color<0] = 0.
Color[Color>1] = 1.


plt.figure(figsize=(6,4.5))
plt.xlim(0,1.05*M_BE.max())
plt.ylim(0,3.2)
ax = plt.gca()

# The curve
points = np.array([M_BE,V_BE]).T.reshape(-1,1,2)
segments = np.concatenate([points[:-1],points[1:]],axis=1)
lc = LineCollection(segments,cmap='plasma',linewidth=2)
lc.set_array(Color)
line = ax.add_collection(lc)

# curve label
plt.plot(-1,-1,color=matplotlib.cm.get_cmap('plasma')(.5),linewidth=2,
    label='Exact BE sphere')

# curve colorbar
ticks = (lg_xi_bar-lg_xi_bar.min())/(lg_xi_bar.max()-lg_xi_bar.min())
labels = [r'$10^{%d}$'%lg_xi for lg_xi in lg_xi_bar]
cb = plt.colorbar(line)
cb.set_ticks(ticks)
cb.set_ticklabels(labels)
cb.set_label(r'$\xi$')

# singular values
plt.scatter(2**.5,1,fc='none',ec='k',marker='o',s=60,label='Singular solution')

plt.legend()
plt.grid()
plt.xlabel(r'$\hat{M}=M\frac{(Nk_{\rm B}T)^{2/3}}{(4\pi p)^{1/6}G^{1/2}}$')
plt.ylabel(r'$\hat{V}=V\frac{Nk_{\rm B}T}{3p}$')
plt.tight_layout()
plt.savefig('image/EoS/V_M.pdf')
plt.close()

# find max M_BE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

Xi_f = np.linspace(6,7,10000) # finer Xi 

# find 
M_BE_f = interp1d(Xi,M_BE,'cubic')(Xi_f)
ind = np.argmax(M_BE_f)
print('Max M_BE=%s, xi=%s.'%(M_BE_f[ind],Xi_f[ind]))

# check
# plt.figure()
# plt.xscale('log')
# plt.yscale('log')
# plt.scatter(Xi,M_BE,s=20)
# plt.scatter(Xi_f,M_BE_f,s=1)
# plt.show()
#'''








