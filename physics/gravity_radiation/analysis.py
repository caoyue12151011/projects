'''
To demo to gravitational radiation of two-body systems. See 
knowledge/gravity_radiation.md for the theories.
'''

import numpy as np 
import matplotlib.pyplot as plt 
import astropy.units as u
import astropy.constants as ct


# parameters
System = {
    'Sun-earth': {
        'm1': 1*u.M_sun,
        'm2': 1*u.M_earth,
        'r': 1*u.AU, # separation
    },

    'earth-moon': {
        'm1': 1*u.M_earth,
        'm2': 7.35e22*u.kg,
        'r': 385e3*u.km,
    },

    'Sun-Sun': {
        'm1': 1*u.M_sun,
        'm2': 1*u.M_sun,
        'r': 1*u.AU,
    },

    'neutron-neutron': {
        'm1': 1*u.M_sun,
        'm2': 1*u.M_sun,
        'r': 1*u.R_earth,
    },
}


for name in sorted(System):
    tmp = System[name]

    m1 = tmp['m1']
    m2 = tmp['m2']
    r = tmp['r']

    # calculation
    omega = (ct.G*(m1+m2)/r**3)**.5 # orbital frequency
    T = 2*np.pi/omega # orbital period
    P = 32*ct.G**4/(5*ct.c**5)*(m1*m2)**2*(m1+m2)/r**5
    drdt = -64*ct.G**3/(5*ct.c**5)*m1*m2*(m1+m2)/r**3
    tau = -r/drdt/4

    # demo
    print('The %s system:'%name)
    print('G-radiation power: %.2e W.'%P.to_value(u.W))
    print('dr/dt: %.2e m/s.'%drdt.to_value(u.m/u.s))
    print('Lifetime: %.2e yr.'%tau.to_value(u.yr))
    # print('tau/T = %.2e'%(tau/T*u.K).to_value(u.K))
    print()



# demo -------------------------------------------------------------------------

# change default fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"


#''' r vs t

t = np.linspace(0,1,1000) # t/tau
r = (1-t)**.25 # r/r0 

plt.figure(figsize=(5,3.5))
plt.plot(t,r,'k')
plt.text(.1,.1,r'$\tau=\frac{5c^5r_0^4}{256G^3m_1m_2(m_1+m_2)}$',fontsize=16,
    transform=plt.gca().transAxes)
plt.grid()
plt.xlabel(r'$t/\tau$')
plt.ylabel(r'$r/r_0$')
plt.tight_layout()
plt.savefig('image/r_t.pdf')
plt.close()
#'''








