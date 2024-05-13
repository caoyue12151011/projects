'''
To calculate the probability of a velocity interval for the Maxwell-Boltzmann
distribution. 
Conclusion: for 1 mol of gas, the fastest particle has v ~ 6.5*(2kT/m)^.5.
For 15°C air, that is 2.7 km/s.
'''
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as ct 
import astropy.units as u
from numpy import exp, pi, log10
from scipy.special import erf

# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13

def MB_ccdf(x, option):
    '''
    To calculate the complementary CDF (1-CDF) of the Maxwell-Boltzmann 
    distribution.
    x: = v/M = v/(2kT/m)^.5, where M is the mode of velocity
    option: 'exact' or 'approx'

    Test results: when option='exact', the output is essentially 1 for x ~> 6.
        Last difference from 1 is ~1e-16.
    '''
    res = None
    if option == 'exact':
        res = 1 - erf(x) + 2*x/pi**.5 * exp(-x**2)
    elif option == 'approx':
        res = 2*x/pi**.5 * exp(-x**2)
    elif option == 'approx2':
        res = 2/pi**.5 * exp(-x**2)
    return res


''' Complementary CDF
# parameters
x = np.linspace(1, 15, 300)

plt.figure()
# plt.xscale('log')
plt.yscale('log')
for option in ['exact', 'approx', 'approx2']:
    plt.plot(x, MB_ccdf(x,option), lw=.5, label=option)
plt.legend()
plt.grid()
plt.xlabel(r'$x=\frac{v}{\sqrt{2kT/m}}$')
plt.ylabel(r'$P(>x)$')
plt.tight_layout()
plt.savefig('x_fraction.pdf')
plt.close()
#'''

''' highest particle velocity
# constants
k_B = 1.38e-23  # [J/K]

# parameters
Tc = 15  # [°C]
p = 1.013e5  # [Pa]
d = 4e-10  # [m], particle diameter
m = 29*1.67e-27  # [kg], particle mass
x = np.linspace(1, 17, 300)

T = Tc + 273.15  # [K]
Z = p*d**2 * (8*pi/(m*k_B*T))**.5  # [Hz]
N1 = p*1 / (k_B*T)  # 1 m^3 of air
N2 = 5.15e18 / m  # total earth atmosphere 
v = (2*k_B*T/m)**.5 * x / 1e3  # [km/s]

ccdf = MB_ccdf(x, 'exact')
t_ob1 = 1/(N1*Z*ccdf)
t_ob2 = 1/(N2*Z*ccdf)

Ind1 = (1e-9 < t_ob1)*(t_ob1 < 1e17)
Ind2 = (1e-9 < t_ob2)*(t_ob2 < 1e17)

yticks = [1e-9, 1e-6, 1e-3, 1, 3600, 3600*24*365, 3.15e10, 3.15e13, 3.15e16]
ylabels = ['ns', r'$\rm\mu s$', 'ms', 's', 'hr', 'yr', 'kyr', 'Myr', 'Gyr']

plt.figure()
plt.yscale('log')
ax = plt.gca()
plt.plot(v[Ind1], t_ob1[Ind1], lw=2, label=r'$1\rm\ m^3$'+' of air')
plt.plot(v[Ind2], t_ob2[Ind2], lw=2, label='Total earth atmosphere')
plt.legend()
plt.grid()
plt.text(.7, .1, r'$Z=%.2e\rm\ Hz$'%Z, fontsize=14, transform=ax.transAxes)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
plt.xlabel(r'$v_{\rm max}\rm\ (km\ s^{-1})$')
plt.ylabel('Observing time')
plt.tight_layout()
plt.savefig('t_ob_vs_v.pdf')
plt.close()
#'''






