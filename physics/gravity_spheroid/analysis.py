'''
To illustrate the Maclaurin spheriod & the Jacobi ellipsoid. See 
knowledge/rotating_fluid.md for more information.
'''

import numpy as np 
import matplotlib.pyplot as plt 
import astropy.constants as ct  
import astropy.units as u
from numpy import sin, cos, tan, arcsin, arccos, arctan
from matplotlib.patches import Ellipse
from scipy.special import ellipkinc, ellipeinc
from scipy.optimize import fsolve


def Jc_eq(beta,gamma):
    '''
    The geometry equation of the Jacobi ellipsoid. See Eq 2.139 at 
    https://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node36.html.
    Some definitions in the equation: e12=sin(beta), e13=sin(gamma), 
    alpha = arcsin(sin(beta)/sin(gamma)). The equation has a trivial solution 
    beta=0, which implies a Maclaurin spheriod. It also has a non-zero solution
    when e > e_bf = 0.812670. This function returns res such that res=0.
    '''

    alpha = arcsin(sin(beta)/sin(gamma))
    E = ellipeinc(gamma,sin(alpha)**2)
    F = ellipkinc(gamma,sin(alpha)**2)
    res = (E - 2*F + 
        ((1+(sin(alpha)*tan(beta)*cos(gamma))**2)/cos(alpha)**2) * E -
        sin(alpha)*tan(beta)*cos(gamma)*(1+sin(beta)**2)/cos(alpha)**2 )
    # res = np.sign(res)*np.abs(res)**4

    return res



def solver(e,name):
    '''
    To calculate the parameters of the Maclaurin spheriod or Jacobi ellipsoid 
    given e_13. 

    Inputs
    ------
    e: eccentricity of the ellipse in the x-z plane (e_13)
    name: 'Maclaurin' or 'Jacobi'

    Returns
    -------
    a, b, c: semi-axes such that abc = 1. For Maclaurin spheriod b is omitted
    e12: (Jacobi ellipsoid only) eccentricity of the ellipse in the x-y plane
    omega_h: reduced angular v, = omega/sqrt(pi*G*rho)
    L_h: reduced angular momentum, = L/sqrt(GM^3*a_bar)
    T_h: reduced kinetic energy, = T/(GM*rho*a_bar^2)
    U_h: reduced potential energy, = U/(GM*rho*a_bar^2)
    E_h: reduced total energy, = T_h+U_h 

    Notes
    -----
    * Some definitions for the Jacobi ellipsoid: e12=sin(beta), e13=sin(gamma), 
      alpha = arcsin(sin(beta)/sin(gamma)).
    * When e<=e_bf, the Jacobian results become Maclaurin ones (see 
      image/trivia/e_vs_beta.pdf).
    '''

    a, b, c = None, None, None
    e12 = None
    omega_h = None
    L_h = None
    T_h = None
    U_h = None
    E_h = None

    if name=='Maclaurin':

        fe = 2*(1-e**2)**.5/e**3*(3-2*e**2)*arcsin(e) - 6/e**2*(1-e**2)

        a = (1-e**2)**(-1/6)
        c = a*(1-e**2)**.5
        omega_h = fe**.5
        L_h = 3**.5/5*fe**.5/(1-e**2)**(1/3) 
        T_h = np.pi/5*fe/(1-e**2)**(1/3) 
        U_h = -.6*arcsin(e)/e*(1-e**2)**(1/6)
        E_h = T_h + U_h 

        return a, c, omega_h, L_h, T_h, U_h, E_h


    elif name=='Jacobi':

        # Warn if e<=e_bf
        if np.any(e<=e_bf):
            print('Warning: some e<=e_bf=%s when solving the Jacobi ellipsoid.'%
                e_bf)

        # find gamma
        gamma = arcsin(e)

        # find beta 
        # the non-trivial solution of Jc_eq=0 for e>e_bf, or the trivial 
        # solution (0) otherwise
        Beta0 = np.linspace(0,np.pi/2,100) # beta sequence 
        Eq0 = Jc_eq(Beta0[:,np.newaxis],gamma) # shape: (len(Beta0),len(e))
        beta_max = Beta0[np.nanargmax(Eq0,axis=0)] # that maximizes Jc_eq

        beta = None
        if isinstance(e,np.ndarray):
            beta = np.full_like(e,np.nan)
            for i in range(len(e)): 
                Beta = np.linspace(beta_max[i],np.pi/2,100) # beta sequence
                Eq = Jc_eq(Beta[:,np.newaxis],gamma[i])
                beta_guess = Beta[np.nanargmin(Eq**2,axis=0)] # blind solver
                beta[i] = fsolve(Jc_eq,beta_guess,args=gamma[i])
        else:
            Beta = np.linspace(beta_max,np.pi/2,100) # beta sequence
            Eq = Jc_eq(Beta[:,np.newaxis],gamma)
            beta_guess = Beta[np.nanargmin(Eq**2,axis=0)] # blind solver
            beta = fsolve(Jc_eq,beta_guess,args=gamma)

        # find alpha
        alpha = arcsin(sin(beta)/sin(gamma))

        # elliptical integrals
        E = ellipeinc(gamma,sin(alpha)**2)
        F = ellipkinc(gamma,sin(alpha)**2)
        

        # physical parameters 
        a = (cos(beta)*cos(gamma))**(-1/3)
        b = a*cos(beta)
        c = a*cos(gamma)
        e12 = sin(beta)
        omega_h = (4*( (F-E)/(tan(beta)*sin(beta)*tan(gamma)) +
            cos(beta)*E/tan(gamma)**3/cos(alpha)**2 - 
            (cos(beta)/tan(gamma)/cos(alpha))**2  ))**.5
        L_h = 3**.5/10*(1+cos(beta)**2)/(cos(beta)*cos(gamma))**(2/3)*omega_h 
        T_h = 3/10*(1+cos(beta)**2)/(cos(beta)*cos(gamma))**(2/3)*omega_h**2
        U_h = -3/5*(cos(beta)*cos(gamma))**(1/3)/sin(gamma)*F 
        E_h = T_h+U_h

        return a, b, c, e12, omega_h, L_h, T_h, U_h, E_h



# parameters
e_bf = .812670 # eccentricity at bifurcation from Mc to Jc
e_ft_Mc = .92996 # eccentricity at largest possible angular velocity
e_st_Mc = .952887 # critical eccentricity for dynamical stability 
e_st_Jc = .93858 # critical eccentricity for dynamical stability of Jc
q = .5 # power index to control the grid spacing of e, smaller q causes grids
    # to crowd around e = 0 and 1


# eccentricity of the ellipse in the x-z plane (e_13)
e0 = np.linspace(1e-3,1-1e-2,1000) 
e = .5*np.sign(sin(np.pi*(e0-.5)))*np.abs(sin(np.pi*(e0-.5)))**q + .5
e_Jc = e[e>e_bf] # e13 for Jacobi ellipsoid


# solve the shape of the rotating mass
# 0     1       2          3      4       5       6
a_Mc, c_Mc, omega_h_Mc, L_h_Mc, T_h_Mc, U_h_Mc, E_h_Mc = solver(e,'Maclaurin')
# 0    1    2    3       4          5    6       7      8
a_Jc,b_Jc,c_Jc,e12_Jc,omega_h_Jc,L_h_Jc,T_h_Jc,U_h_Jc,E_h_Jc = solver(e_Jc,
    'Jacobi')


# parameters at critical points
para_bf_Mc = solver(e_bf,'Maclaurin') # Mc-Jc bifurcation point
para_ft_Mc = solver(e_ft_Mc,'Maclaurin') # fastest rotating Mc spheriod 
para_st_Mc = solver(e_st_Mc,'Maclaurin') # Mc critical stable point
para_st_Jc = solver(e_st_Jc,'Jacobi') # Jc critical stable point


# dynamically unstable part
Ind_st_Mc = np.where(e>e_st_Mc)[0]
Ind_st_Jc = np.where(e_Jc>e_st_Jc)[0]


# demo -------------------------------------------------------------------------

# change default fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

# global parameters
cl_Mc = 'blue' # color for Maclaurin spheriod
cl_st_Mc = 'violet' # color for dynamically unstable Maclaurin spheriod 
cl_Jc = 'red' # color for Jacobi ellipsoid
cl_st_Jc = 'orange'
lw = 1 # line width for general plot
lw_st = 3 # line width for dynamically unstable part


''' e grid
plt.figure()
plt.scatter(e0,e,s=1,color='k')
plt.grid()
plt.xlabel(r'$e_{13,\rm orig}$')
plt.ylabel(r'$e_{13}$')
plt.tight_layout()
plt.savefig('image/trivia/e_grid.pdf')
plt.close()
#'''


''' Plots for solving the equation Jc_eq(beta)=0

# parameters
ee = np.linspace(.001,.999,1000) # e_13
beta = np.linspace(0,np.pi/2,1000) # sin(beta)=e_12

# calculate Jc_eq(beta)
gamma = arcsin(ee)
Eq = Jc_eq(beta[:,np.newaxis],gamma) # shape: (len(beta),len(ee))
beta_max = beta[np.nanargmax(Eq,axis=0)] # that maximizes Jc_eq

# solve Jc_eq=0
beta_sol = arcsin(solver(ee,'Jacobi')[3])


# beta vs Jc_eq with varying e 
plt.figure()
plt.plot(beta,Eq,linewidth=.2)
plt.axhline(0,color='k')
plt.grid()
plt.title('e=%.4f~%.4f'%(ee.min(),ee.max()))
plt.xlabel('beta')
plt.ylabel('Jc_eq')
plt.tight_layout()
plt.savefig('image/trivia/beta_vs_Jc_eq.pdf')
plt.close()


# e vs beta
plt.figure()
plt.plot(ee,beta_max,label='beta_max')
plt.plot(ee,beta_sol,label='beta_sol')
plt.legend()
plt.grid()
plt.xlabel('e')
plt.ylabel('beta')
plt.tight_layout()
plt.savefig('image/trivia/e_vs_beta.pdf')
plt.close()
#'''


#''' e vs quantities 
plt.figure(figsize=(4,8))

# e vs omega_h_Mc
plt.axes([0.16,0.07,0.8,0.3])
plt.plot(e,omega_h_Mc,color=cl_Mc,linewidth=lw,zorder=20,
    label='Maclaurin (Mc) spheriod')
plt.plot(e[Ind_st_Mc],omega_h_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19,label='Mc spheriod (unstable)')
plt.plot(e_Jc,omega_h_Jc,color=cl_Jc,linewidth=lw,zorder=20,
    label='Jacobi (Jc) ellipsoid')
plt.plot(e_Jc[Ind_st_Jc],omega_h_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19,label='Jc ellipsoid (unstable)')
plt.legend()
plt.grid()
plt.xlabel(r'$e_{13}$')
plt.ylabel(r'$\omega/\sqrt{\pi G\rho}$')

# e vs L_h_Mc
plt.axes([0.16,0.37,0.8,0.3])
plt.ylim(-.1,1)
plt.plot(e,L_h_Mc,color=cl_Mc,linewidth=lw,zorder=20)
plt.plot(e[Ind_st_Mc],L_h_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19)
plt.plot(e_Jc,L_h_Jc,color=cl_Jc,linewidth=lw,zorder=20)
plt.plot(e_Jc[Ind_st_Jc],L_h_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19)
plt.grid()
plt.gca().xaxis.set_ticklabels([])
plt.gca().xaxis.set_ticks_position('none')
plt.ylabel(r'$L/\sqrt{GM^3\bar{a}}$')


# e vs T_h_Mc
plt.axes([0.16,0.67,0.8,0.3])
plt.plot(e,T_h_Mc,color=cl_Mc,linewidth=lw,linestyle='--',zorder=20)
plt.plot(e[Ind_st_Mc],T_h_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19)
plt.plot(e,U_h_Mc,color=cl_Mc,linewidth=lw,linestyle=':',zorder=20)
plt.plot(e[Ind_st_Mc],U_h_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19)
plt.plot(e,E_h_Mc,color=cl_Mc,linewidth=lw,zorder=20)
plt.plot(e[Ind_st_Mc],E_h_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19)

plt.plot(e_Jc,T_h_Jc,color=cl_Jc,linewidth=lw,linestyle='--',zorder=20)
plt.plot(e_Jc[Ind_st_Jc],T_h_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19)
plt.plot(e_Jc,U_h_Jc,color=cl_Jc,linewidth=lw,linestyle=':',zorder=20)
plt.plot(e_Jc[Ind_st_Jc],U_h_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19)
plt.plot(e_Jc,E_h_Jc,color=cl_Jc,linewidth=lw,zorder=20)
plt.plot(e_Jc[Ind_st_Jc],E_h_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19)

# freeze axis limits 
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

# labels
plt.plot(10,10,color='k',linewidth=lw,linestyle='--',label='Kinetic')
plt.plot(10,10,color='k',linewidth=lw,linestyle=':',label='Potential')
plt.plot(10,10,color='k',linewidth=lw,label='Total')

plt.legend()
plt.grid()
plt.gca().xaxis.set_ticklabels([])
plt.gca().xaxis.set_ticks_position('none')
plt.ylabel(r'Energy$/(GM^2/\bar{a})$')
plt.savefig('image/e_vs_quantity.pdf')
plt.close()
#'''


#''' L_h vs omega_h
plt.figure(figsize=(7,5.5))

plt.plot(L_h_Mc,omega_h_Mc,color=cl_Mc,linewidth=lw,zorder=20,
    label='Maclaurin (Mc) spheriod')
plt.plot(L_h_Mc[Ind_st_Mc],omega_h_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19,label='Mc spheriod (unstable)')

plt.plot(L_h_Jc,omega_h_Jc,color=cl_Jc,linewidth=lw,zorder=20,
    label='Jacobi (Jc) ellipsoid')
plt.plot(L_h_Jc[Ind_st_Jc],omega_h_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19,label='Jc ellipsoid (unstable)')

plt.legend()
plt.grid()
plt.xlabel(r'$L/\sqrt{GM^3\bar{a}}$')
plt.ylabel(r'$\omega/\sqrt{\pi G\rho}$')
plt.tight_layout()
plt.savefig('image/L_h_vs_omega_h.pdf')
plt.close()
#'''


#''' L_h vs energies
plt.figure(figsize=(7,5.5))

plt.plot(L_h_Mc,T_h_Mc,color=cl_Mc,linewidth=lw,linestyle='--',zorder=20)
plt.plot(L_h_Mc,U_h_Mc,color=cl_Mc,linewidth=lw,linestyle=':',zorder=20)
plt.plot(L_h_Mc,E_h_Mc,color=cl_Mc,linewidth=lw,zorder=20,
    label='Maclaurin (Mc) spheriod')
plt.plot(L_h_Mc[Ind_st_Mc],T_h_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19,label='Mc spheriod (unstable)')
plt.plot(L_h_Mc[Ind_st_Mc],U_h_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19)
plt.plot(L_h_Mc[Ind_st_Mc],E_h_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19)

plt.plot(L_h_Jc,T_h_Jc,color=cl_Jc,linewidth=lw,linestyle='--',zorder=20)
plt.plot(L_h_Jc,U_h_Jc,color=cl_Jc,linewidth=lw,linestyle=':',zorder=20)
plt.plot(L_h_Jc,E_h_Jc,color=cl_Jc,linewidth=lw,zorder=20,
    label='Jacobi (Jc) ellipsoid')
plt.plot(L_h_Jc[Ind_st_Jc],T_h_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19,label='Jc ellipsoid (unstable)')
plt.plot(L_h_Jc[Ind_st_Jc],U_h_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19)
plt.plot(L_h_Jc[Ind_st_Jc],E_h_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19)

# freeze axis limits & create labels
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot(10,10,color='k',linewidth=lw,linestyle='--',label='Kinetic')
plt.plot(10,10,color='k',linewidth=lw,linestyle=':',label='Potential')
plt.plot(10,10,color='k',linewidth=lw,label='Total')

plt.legend()
plt.grid()
plt.xlabel(r'$L/\sqrt{GM^3\bar{a}}$')
plt.ylabel(r'Energy$/(GM^2/\bar{a})$')
plt.tight_layout()
plt.savefig('image/L_h_vs_T_h.pdf')
plt.close()
#'''


#''' L_h vs a,b,c
plt.figure(figsize=(7,5.5))
plt.xlim(0,1)
plt.ylim(0,3)

plt.plot(L_h_Mc,a_Mc,color=cl_Mc,linewidth=lw,zorder=20,
    label='Maclaurin (Mc) spheriod')
plt.plot(L_h_Mc,c_Mc,color=cl_Mc,linestyle=':',linewidth=lw,zorder=20)
plt.plot(L_h_Mc[Ind_st_Mc],a_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19,label='Mc spheriod (unstable)')
plt.plot(L_h_Mc[Ind_st_Mc],c_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,
    zorder=19)

plt.plot(L_h_Jc,a_Jc,color=cl_Jc,linewidth=lw,zorder=20,
    label='Jacobi (Jc) ellipsoid')
plt.plot(L_h_Jc,b_Jc,color=cl_Jc,linestyle='--',linewidth=lw,zorder=20)
plt.plot(L_h_Jc,c_Jc,color=cl_Jc,linestyle=':',linewidth=lw,zorder=20)

plt.plot(L_h_Jc[Ind_st_Jc],a_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19,label='Jc ellipsoid (unstable)')
plt.plot(L_h_Jc[Ind_st_Jc],c_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19)
plt.plot(L_h_Jc[Ind_st_Jc],b_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19)

# freeze axis limits & create labels
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot(10,10,color='k',linewidth=lw,label=r'$a$')
plt.plot(10,10,color='k',linewidth=lw,linestyle='--',label=r'$b$')
plt.plot(10,10,color='k',linewidth=lw,linestyle=':',label=r'$c$')

plt.legend()
plt.grid()
plt.xlabel(r'$L/\sqrt{GM^3\bar{a}}$')
plt.ylabel('Axis lengths')
plt.tight_layout()
plt.savefig('image/L_h_vs_lengths.pdf')
plt.close()
#'''


#''' e13-e12 diagram 

# parameters
marker_size = 5
Para = { # for demo of shapes
    'bf': { 
        'e_13': e_bf,
        'e_12': 0,
        'x1': e_bf, # coords of the critical point
        'y1': 0,
        'x2': .77, # coords of the demo shape
        'y2': .03,
        'width': .07, # fraction of the figure width
        'color': cl_Mc, 
        'lw': .5*lw,
        'label': None,
    },

    'ft_Mc': { 
        'e_13': e_ft_Mc,
        'e_12': 0,
        'x1': e_ft_Mc, # coords of the critical point
        'y1': 0,
        'x2': .86, # coords of the demo shape
        'y2': .03,
        'width': .07, # fraction of the figure width
        'color': cl_Mc, 
        'lw': .5*lw,
        'label': r'$\omega_{\rm max}$',
    },

    'st_Mc': { 
        'e_13': e_st_Mc,
        'e_12': 0,
        'x1': e_st_Mc, # coords of the critical point
        'y1': 0,
        'x2': .96, # coords of the demo shape
        'y2': .03,
        'width': .07, # fraction of the figure width
        'color': cl_Mc, 
        'lw': .5*lw,
        'label': None,
    },

    'st_Jc': { 
        'e_13': e_st_Jc,
        'e_12': para_st_Jc[3],
        'x1': e_st_Jc, # coords of the critical point
        'y1': para_st_Jc[3],
        'x2': .87, # coords of the demo shape
        'y2': para_st_Jc[3],
        'width': .07, # fraction of the figure width
        'color': cl_Jc, 
        'lw': .5*lw,
        'label': None,
    },
}


plt.figure(figsize=(7,7))
plt.axis('equal')

# x=y line
plt.plot([0,1],[0,1],color='gray',linewidth=lw)

# Mc spheriod
plt.plot(e,np.zeros_like(e),color=cl_Mc,linewidth=lw)
plt.plot(e[Ind_st_Mc],np.zeros_like(e[Ind_st_Mc]),color=cl_st_Mc,
    linewidth=lw_st)

# Jc ellipsoid
e_Jc1 = np.concatenate(([e_bf],e_Jc)) # to connect lines
e12_Jc1 = np.concatenate(([0],e12_Jc))
plt.plot(e_Jc1,e12_Jc1,color=cl_Jc,linewidth=lw)
plt.plot(e_Jc1[Ind_st_Jc+1],e12_Jc1[Ind_st_Jc+1],color=cl_st_Jc,linewidth=lw_st)


# shape illustration of some critical points
for name in Para:
    e_13 = Para[name]['e_13']
    e_12 = Para[name]['e_12']
    x1 = Para[name]['x1']
    y1 = Para[name]['y1']
    x2 = Para[name]['x2']
    y2 = Para[name]['y2']
    width = Para[name]['width']
    color = Para[name]['color']
    lw = Para[name]['lw']
    label = Para[name]['label']

    height = width*(1-e_13**2)**.5
    b = width/2*(1-e_12**2)**.5

    ell = Ellipse(xy=(x2,y2),width=width,height=height,ec=color,fc='w',
        linewidth=lw,zorder=20)

    plt.gca().add_patch(ell)
    plt.plot([x1,x2],[y1,y2],color='k',linewidth=lw,zorder=19)
    plt.plot([x2,x2+b],[y2,y2],color=color,linewidth=lw,zorder=21)
    plt.text(x2,y2+height/1.5,label,zorder=22)

plt.grid()
plt.xlabel(r'$e_{13}$')
plt.ylabel(r'$e_{12}$')
plt.tight_layout()
plt.savefig('image/e_diagram.pdf')
plt.close()
#'''


#''' Rotation period vs e of the Earth

# parameters
M = 6e24 # [kg], mass
a_bar = 6371 # [km], average radius, =(abc)^(1/3)

# calculation
rho = 3*(M*u.kg)/(4*np.pi*(a_bar*u.km)**3) # density
omega_Mc = omega_h_Mc*(np.pi*ct.G*rho)**.5 # angular v
P_Mc = (2*np.pi/omega_Mc).to_value(u.h) # [hr]
omega_Jc = omega_h_Jc*(np.pi*ct.G*rho)**.5 # angular v
P_Jc = (2*np.pi/omega_Jc).to_value(u.h) # [hr]


plt.figure(figsize=(5,4))
plt.ylim(0,25)

plt.plot(e,P_Mc,color=cl_Mc,linewidth=lw,zorder=20,
    label='Maclaurin (Mc) spheriod')
plt.plot(e[Ind_st_Mc],P_Mc[Ind_st_Mc],color=cl_st_Mc,linewidth=lw_st,zorder=19,
    label='Mc spheriod (unstable)')

plt.plot(e_Jc,P_Jc,color=cl_Jc,linewidth=lw,zorder=20,
    label='Jacobi (Jc) ellipsoid')
plt.plot(e_Jc[Ind_st_Jc],P_Jc[Ind_st_Jc],color=cl_st_Jc,linewidth=lw_st,
    zorder=19,label='Jc ellipsoid (unstable)')

plt.legend()
plt.grid()
plt.xlabel("Earth's "+r'$e_{13}$')
plt.ylabel("Earth's rotation period (hr)")
plt.tight_layout()
plt.savefig('image/e_vs_period_Earth.pdf')
plt.close()
#'''








