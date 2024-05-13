'''
This file contains:
 * Physics model of a smooth ball moving in a fluid at a constant speed. 
   Its dependence on Re and Ma is considered. The analytical expression of 
   drag are derived from the experimental data of firing cannon balls in the 
   air. So strictly speaking the calculations are valid for such conditions 
   only.
 * ICAO standard atmosphere model (1993) 
'''

import dill
import num2tex
import matplotlib
import numpy as np
import matplotlib.pyplot as plt 
from ambiance import Atmosphere as Atm 


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def Cd_vs_Re(Re):
    '''
    To calculate drag coefficient given Re. Do not recommend using it for 
    Re > 1e6. Ref: 
    https://pages.mtu.edu/~fmorriso/DataCorrelationForSphereDrag2016.pdf

    Inputs
    ------
    Re: scalar or np.ndarray, Reynolds number, NaN is allowed

    Returns
    -------
    Cd: scalar or np.ndarray, drag coefficient
    '''
    if np.any(Re > 1e6):
        print('Warning (Cd_vs_Re): calculating Cd with Re > 1e6 is not '
              'recommended.')

    a = 2.6 * (Re/5) / (1 + (Re/5)**1.52)
    b = .411 * (Re/2.63e5)**-7.94 / (1 + (Re/2.63e5)**-8)
    c = .25 * (Re/1e6) / (1+Re/1e6)

    return 24/Re + a + b + c


def Cd_st_vs_Re(Re):
    '''
    To calculate the Stokes drag coefficient given Re. Re should be < 1 to 
    apply the Stokes law. 
    '''
    if np.any(Re > 1):
        print('Warning (Cd_st_vs_Re): Re > 1 while using Stokes law.')
    return 24/Re    


def spec_func(Ma, func_type):
    '''
    To calculate functions k(Ma), t(Ma) in Cd(Re, Ma). Ref: 
    https://www.arc.id.au/CannonballDrag.html

    Inputs
    ------
    Ma: scalar or np.ndarray
    func_type: 'k', 't' or 's', function type

    Returns
    -------
    y: scalar or np.ndarray, function outputs
    '''

    if func_type == 's':
        y = 0.78 + 0.22*np.arctan(-12*(Ma-0.23))
    
    else:
        Ma = np.array([Ma])

        if func_type == 'k':
            Ma0,y0,x1,x2,x3,x4,y1,y2,y3,y4 = 1.5,1,.1,.95,.55,1.5,0,0,.95,1
        elif func_type == 't':
            Ma0,y0,x1,x2,x3,x4,y1,y2,y3,y4 = 1,0,0,.85,.57,1,1.1,1.1,.05,0

        # au^3 + bu^2 + cu + d = 0
        a = -x1 + 3*x2 - 3*x3 + x4 
        b = 3*x1 - 6*x2 + 3*x3 
        c = -3*x1 + 3*x2 
        d = x1 - Ma

        p = (3*a*c - b**2) / (3*a**2)
        q = (27*a**2*d - 9*a*b*c + 2*b**3) / (27*a**3)
        delta = (q/2)**2 + (p/3)**3

        # calculate the function
        m = -q/2 + delta**.5
        n = -q/2 - delta**.5
        u = -b/(3*a) + np.sign(m)*abs(m)**(1/3) + np.sign(n)*abs(n)**(1/3)
        y = y1*(1-u)**3 + y2*3*u*(1-u)**2 + y3*3*u**2*(1-u) + y4*u**3
        y[Ma >= Ma0] = y0

        y = y[0]

    return y


def Cd_vs_ReMa(Re, Ma, debug=False, broadcast=True):
    ''' 
    To calculate the drag coefficient given Re and Ma. See 
    https://www.arc.id.au/CannonballDrag.html.

    Inputs
    ------
    Re, Ma: scalar or 1d array
    debug: whether to return func k(Ma), t(Ma), s(Ma)
    broadcast: whether to broadcast axes of Re & Ma for output values

    Returns
    -------
    Cd: its type determined by 

               | Re scalar | Re 1darray 
    ------------------------------------------
    Ma scalar  | scalar    | 1darray 
    ------------------------------------------
    Ma 1darray | 1darray   | 2darray of (Ma,Re) (if broadcast=True)
               |           | 1darray (Re and Ma of same length,broadcast=F)  

    k, t, s: if debug=True
    '''
    k, t, s = [spec_func(Ma, func_type) for func_type in ['k', 't', 's']]
    if isinstance(Re,np.ndarray) and isinstance(Ma,np.ndarray) and broadcast:
        k = k[...,np.newaxis]
        t = t[...,np.newaxis]
        s = s[...,np.newaxis]

    Cd = k + t*Cd_vs_Re(s*Re)  # (Ma, Re)

    if debug:
        return Cd, k, t, s 
    else:
        return Cd


def drag(v, d, rho, mu, cs):
    '''
    To calculate the drag force.

    Inputs
    ------
    v: [m/s], velocity of the sphere, >=0
    d: [m], diameter of the sphere, >0
    rho: [kg/m^3], fluid density. = nan if in space
    mu: [kg/m/s], dynamic viscosity. = nan if in space
    cs: [m/s], sound speed. = nan if in space

    Notes: one of the inputs can be array, others must be scalars

    Returns
    -------
    Fd: [N], drag force 
    '''

    Re = rho*v*d/mu
    Ma = v/cs 
    Cd = Cd_vs_ReMa(Re, Ma, debug=False, broadcast=False) 
    Fd = np.pi/8 * rho * v**2 * Cd * d**2 

    return Fd


def calc_atm(h, prop, Atm_data):
    '''
    To calculate the atmosphere properties given height. ICAO standard 
    atmosphere (1993) is used.

    Inputs
    ------
    h: [m], scalar or ndarray, altitude
    prop: string or list of strings, names of the properties to be calculated
        can be 'p', 'rho', 'T', 'mu', 'cs'
    Atm_data: dict, the atmosphere model

    Returns
    -------
    res: [SI units], scalar/ndarray or list of scalars/ndarrays
    '''
    if isinstance(prop, str):
        res = np.interp(h, Atm_data['h'], Atm_data[prop], left=np.nan,
                        right=np.nan)
    else:
        res = [np.interp(h, Atm_data['h'], Atm_data[i], left=np.nan, 
               right=np.nan) for i in prop]

    return res


# ICAO 1993 atmosphere data ---------------------------------------------------
# To store the atmosphere data. It's faster to interpolate the data than using
# the Atm tool for massive computations.

h_min = -5004 # [m]
h_max = 81020

h = np.linspace(h_min,h_max,300)
atm = Atm(h,check_bounds=False)
Atm_data = {
    'h': h, 
    'p': atm.pressure,
    'rho': atm.density,
    'T': atm.temperature_in_celsius,
    'mu': atm.dynamic_viscosity,
    'cs': atm.speed_of_sound,
}


# outputs
dill.dump(spec_func, open('variable/spec_func.p','wb'))
dill.dump(Cd_vs_Re, open('variable/Cd_vs_Re.p','wb'))
dill.dump(Cd_vs_ReMa, open('variable/Cd_vs_ReMa.p','wb'))
dill.dump(drag, open('variable/drag.p','wb'))
dill.dump(Atm_data,open('variable/Atm_data.p','wb'))
dill.dump(calc_atm, open('variable/calc_atm.p','wb'))


# demo =========================================================================

''' special functions

Ma = np.logspace(-1.5, .5, 300)
Cd, k, t, s = Cd_vs_ReMa(1e4, Ma, debug=True)

plt.figure(figsize=(5,4))
plt.xscale('log')
plt.plot(Ma,k,lw=2,c='C0')
plt.plot(Ma,t,lw=2,c='C1')
plt.plot(Ma,s,lw=2,c='C2')
plt.text(1.05,1.02,r'$k\rm(Ma)$',color='C0',fontsize=14)
plt.text(.3,.8,r'$t\rm(Ma)$',color='C1',fontsize=14)
plt.text(.13,.6,r'$s\rm(Ma)$',color='C2',fontsize=14)
plt.grid()
plt.xlabel('Ma')
plt.tight_layout()
plt.savefig('image/drag_model/spec_func.pdf')
plt.close()
#'''


''' Cd vs Re, constant Ma

# parameters
Re = np.logspace(.5,7,1000)
Re_st =  np.logspace(.5,2.3,1000)
Ma = np.array([0,.4,.6,.7,.8,.9,1,1.1,1.3])

Cd = Cd_vs_ReMa(Re,Ma)
Cd0 = Cd_vs_Re(Re)
Cd_st = Cd_st_vs_Re(Re_st)

cmap = matplotlib.cm.get_cmap('rainbow_r')
c_value = np.linspace(0,1,len(Ma))

# demo
plt.figure(figsize=(6,4))
plt.xscale('log')
plt.yscale('log')
plt.plot(Re,Cd0,lw=1,color='k',label='Miller&Bailey 1979')
plt.plot(Re_st,Cd_st,ls='--',lw=1,color='k',label='Stokes law')
for i in range(len(Ma)):
    c = cmap(c_value[i])
    plt.plot(Re,Cd[i],lw=1,color=c)
    plt.text(1.01,.9-.05*i,'Ma=%s'%Ma[i],color=c,fontsize=10,
        transform=plt.gca().transAxes)
plt.legend()
plt.grid()
plt.xlabel('Re')
plt.ylabel(r'$C_{\rm d,sphere}$')
plt.tight_layout()
plt.savefig('image/drag_model/Cd_Re_iso-Ma.pdf')
plt.close()
#'''


''' Cd vs Ma, constant Re

# parameters
Re = np.array([10,100,1e3,1e4,1e5,2e5,3e5,4e5,5e5,6e5,7e5,8e5,9e5,1e6,1e7]) 
Ma = np.linspace(0,1.5,300)

Cd = Cd_vs_ReMa(Re,Ma)
cmap = matplotlib.cm.get_cmap('rainbow_r')
c_value = np.linspace(0,1,len(Re))

# demo
plt.figure(figsize=(6,4))
plt.yscale('log')
for i in range(len(Re)):
    c = cmap(c_value[i])
    plt.plot(Ma,Cd[:,i],lw=1,color=c)
    plt.text(1.01,.9-.05*i,r'$\rm Re=%s$'%num2tex.num2tex('%.1e'%Re[i]),color=c,
        fontsize=10,transform=plt.gca().transAxes)
plt.grid()
plt.xlabel('Ma')
plt.ylabel(r'$C_{\rm d,sphere}$')
plt.tight_layout()
plt.savefig('image/drag_model/Cd_Ma_iso-Re.pdf')
plt.close()
#'''


''' Cd vs v, others remain constant

# parameters
Ma = np.logspace(-2.5,.4,300) 
lg_r_RM = np.linspace(4,7,7) # Re/Ma ratio

# parameters for demo
cmap = matplotlib.cm.get_cmap('rainbow_r')
c_value = np.linspace(0,1,len(lg_r_RM))

Re = 10**lg_r_RM[...,np.newaxis]*Ma 

# demo
plt.figure(figsize=(6,4))
plt.xscale('log')
for i in range(len(lg_r_RM)):
    Cd = Cd_vs_ReMa(Re[i],Ma,broadcast=False)
    c = cmap(c_value[i])

    plt.plot(Ma,Cd,color=c,lw=1)
    plt.text(1.01,.9-.05*i,
        r'$\rm Re/Ma=10^{%s}$'%lg_r_RM[i],
        color=c,fontsize=10,transform=plt.gca().transAxes)
plt.grid()
plt.xlabel(r'$v/c_{\rm s}$')
plt.ylabel(r'$C_{\rm d}$')
plt.tight_layout()
plt.savefig('image/drag_model/Cd_v.pdf')
plt.close()
#'''


''' Re-Ma diagram 

# parameters
Re = np.logspace(2,7,500) 
Ma = np.linspace(0,1.5,300)
lg_r_RM = np.arange(1,9) # Re/Ma ratio

lgRe = np.log10(Re)
Cd = Cd_vs_ReMa(Re,Ma)

# demo
plt.figure(figsize=(6,4))
plt.imshow(Cd,origin='lower',extent=[lgRe.min(),lgRe.max(),Ma.min(),
    Ma.max()],cmap='coolwarm',aspect="auto")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
ax = plt.gca()

# v-curve 
for i in range(len(lg_r_RM)):
    plt.plot(lgRe,Re/10**lg_r_RM[i],color='k',ls='--',lw=.5)


# annotations
plt.text(5.5,.18,'Drag crisis',fontsize=14,weight='bold',color='w',
    path_effects=[path_effects.withStroke(linewidth=1,foreground='k')])
plt.text(4,1,'Transonic',fontsize=14,weight='bold',color='w',
    path_effects=[path_effects.withStroke(linewidth=1,foreground='k')])
plt.text(2.05,.4,'Stokes drag',fontsize=14,weight='bold',color='w',
    path_effects=[path_effects.withStroke(linewidth=1,foreground='k')])

# colorbar
cb = plt.colorbar()
cb.ax.set_ylabel(r'$C_{\rm d,sphere}$')
plt.contour(lgRe,Ma,Cd,30,colors='w',linewidths=.5)

# x/ylabels
plt.xlabel(r'$\rm Re$')
ticks = ax.get_xticks()
labels = [r'$10^%d$'%i for i in ticks]
ax.set_xticklabels(labels)
plt.ylabel('Ma')

plt.tight_layout()
plt.savefig('image/drag_model/Cd_map.pdf')
plt.close()
#'''


''' atmosphere profiles

# parameters 
h = np.linspace(-6e3,85e3,1000) # [m]
xlabel = ['Pressure (bar)',r'$\rm Density\ (kg\ m^{-3})$',
    r'$\rm Temperature\ (^\circ C)$',
    r'$\rm Dynamic\ viscosity\ (kg\ m^{-1}s^{-1})$',
    r'$\rm Sound\ speed\ (m\ s^{-1})$']
name = ['p','rho','T','mu','cs']
xticks = np.arange(-5,83,5) # [km]

xlabels = [str(i) for i in xticks]


for i in range(len(name)):
    # if not i==0:
    #     continue

    res = calc_atm(h, [name[i]], Atm_data)[0]
    if name[i]=='p':
        res /= 1e5 # Pa to bar


    plt.figure(figsize=(8,4))
    # if name[i] in ['p','rho']:
    #     plt.xscale('log')

    plt.plot(h/1e3,res,color='k')
    ax = plt.gca()
    plt.text(.6,.9,'ICAO standard atmosphere (1993)',fontsize=12,
        transform=ax.transAxes)

    plt.axvline(0,color='k',ls='--',lw=1,alpha=.7)
    plt.axvline(h_tp/1e3,color='r',ls='--',lw=1,alpha=.7)
    plt.axvline(h_sp/1e3,color='b',ls='--',lw=1,alpha=.7)
    plt.axvline(h_mp/1e3,color='g',ls='--',lw=1,alpha=.7)

    _min = np.nanmin(res)
    plt.text(-3,_min,'See level',color='k',fontsize=12,rotation=90)
    plt.text(h_tp/1e3-3,_min,'Tropopause',color='r',fontsize=12,rotation=90)
    plt.text(h_sp/1e3-3,_min,'Stratopause',color='b',fontsize=12,rotation=90)
    plt.text(h_mp/1e3-3,_min,'Mesopause',color='g',fontsize=12,rotation=90)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    plt.xlabel('Altitude (km)')
    plt.ylabel(xlabel[i])
    plt.grid()
    plt.tight_layout()
    plt.savefig('image/atm_model/%s.pdf'%name[i])
    plt.close()
#'''


''' drag vs velocity for different ball sizes

# parameters
V = np.logspace(0,3,300) # [m/s]
D = np.array([3,10,30,100,300,1000,3e3])/1e3 # [m]
rho = 1.225 # [kg/m^3], 1.225=air at 15 Cel sea level 
mu = 1.81e-5 # [kg/m/s], dynamic viscosity, 1.81e-5=air at 15 Cel 
cs = 340 # [m/s], 340=air at 15 Cel sea level

cmap = matplotlib.cm.get_cmap('rainbow_r')
c_value = np.linspace(0,1,len(D))

# parameters for demo
text_drag = ('Sphere in fluid\n'+
    r'$\rho=%s\rm\ kg\ m^{-3}$'%rho+'\n'+
    r'$\mu=%s\rm\ kg\ m^{-1}s^{-1}$'%num2tex.num2tex('%.2e'%mu)+'\n'+
    r'$c_{\rm s}=%s\rm\ m\ s^{-1}$'%cs)

# demo
plt.figure(figsize=(7,5))
plt.xscale('log')
plt.yscale('log')

for i in range(len(D)):
    Fd = drag(V,D[i],rho,mu,cs)
    c = cmap(c_value[i])
    plt.plot(V,Fd,color=c,lw=1)
    plt.text(1.01,.9-.05*i,r'$d=%d\rm\ mm$'%(1e3*D[i]),color=c,
        fontsize=12,transform=plt.gca().transAxes)

plt.axvline(cs,color='k',ls=':')
plt.grid()
plt.annotate(text_drag,xy=(.05,.95),va='top',xytext=(12,-12),fontsize=14,
    xycoords='axes fraction',textcoords='offset points')
plt.xlabel(r'$v\rm\ (m\ s^{-1})$')
plt.ylabel(r'$F_{\rm d}\rm\ (N)$')
plt.tight_layout()
plt.savefig('image/drag_model/Fd_v.pdf')
plt.close()
#'''


''' drag vs velocity example: basketball

# parameters
v_kmh = np.linspace(0,180,300) # [km/h]
d = .24 # [m]
rho = 1.225 # [kg/m^3], 1.225=air at 15 Cel sea level 
mu = 1.81e-5 # [kg/m/s], dynamic viscosity, 1.81e-5=air at 15 Cel 
cs = 340 # [m/s], 340=air at 15 Cel sea level

v = v_kmh/3.6 
Fd = drag(v,d,rho,mu,cs)

# parameters for demo
text_drag = ('Sphere in fluid\n'+
    r'$\rho=%s\rm\ kg\ m^{-3}$'%rho+'\n'+
    r'$\mu=%s\rm\ kg\ m^{-1}s^{-1}$'%num2tex.num2tex('%.2e'%mu)+'\n'+
    r'$c_{\rm s}=%s\rm\ m\ s^{-1}$'%cs+'\n'+
    r'$d=%s\rm\ m$'%d)

# demo
plt.figure(figsize=(6,4.5))
plt.xlim(v_kmh.min(),v_kmh.max())
plt.ylim(np.nanmin(Fd),np.nanmax(Fd))
plt.plot(v_kmh,Fd,color='k',lw=2)
plt.grid()
plt.annotate(text_drag,xy=(.05,.95),va='top',xytext=(12,-12),fontsize=14,
    xycoords='axes fraction',textcoords='offset points')
plt.xlabel(r'$v\rm\ (km\ h^{-1})$')
plt.ylabel(r'$F_{\rm d}\rm\ (N)$')
plt.tight_layout()
plt.savefig('image/drag_model/Fd_v_basketball.pdf')
plt.close()
#'''


''' fall time using Stokes law
def sec_to_str(t):
    """
    Return string 's' of time 't' (in seconds) such that:
    s in seconds if t < 1 min,
    s in minutes if 1 min <= t < 1 hr,
    s in hours if 1 hr <= t < 1 d,
    s in days if 1 d <= t < 1 mon (1 mon = 30 d), 
    s in months if 1 mon <= t < 1 yr, (1 yr = 365 d)
    s in years if 1 yr <= t, 
    s keeps to one decimal place.
    """
    if t < 60:
        s = f'{t:.1f} s'
    elif t < 3600:
        s = f'{t/60:.1f} min'
    elif t < 3600*24:
        s = f'{t/3600:.1f} hr'
    elif t < 3600*24*30:
        s = f'{t/3600/24:.1f} d'
    elif t < 3600*24*365:
        s = f'{t/3600/24/30:.1f} mon'
    else:
        s = f'{t/3600/24/365:.1f} yr'
    return s

# parameters 
l = 1  # [m], falling distance
d = np.logspace(-7, -4, 300)  # [m]
rho_s = 2e3  # [kg/m^3]
rho = 1.225  # [kg/m^3]
g = 9.8  # [kg m/s^2]
mu = 1.81e-5  # [kg/m/s]
d0_list = [1, 2.5, 10]  # [um]
xticks = [1e-7, 1e-6, 1e-5, 1e-4]
xlabels = [r'$0.1\rm\mu m$', r'$1\rm\mu m$', r'$10\rm\mu m$', '0.1mm']
yticks = [1, 60, 3600, 3600*24, 3600*24*30]
ylabels = ['1s', '1min', '1hr', '1d', '1mon']

# calculation
v = abs(rho_s - rho) * g * d**2 / (18 * mu)  # [m/s]
t = l/v  # [s]
Re = rho*v*d/mu
print(f'max(Re): {max(Re): .2f}. Should be < ~1.')
d0 = np.array(d0_list)*1e-6  # [m]
t0 = np.interp(d0, d, t)

# text info
text = ('Sphere in fluid\nStokes drag considered\n'
        'Brownian motion not considered\n'
        r'$\rho_{\rm s}=%s\rm\ kg\ m^{-3}$'%rho_s + '\n'
        r'$g=%s\rm\ kg\ m\ s^{-2}$'%g + '\n'
        r'$\rho=%s\rm\ kg\ m^{-3}$'%rho + '\n'
        r'$\mu=%s\rm\ kg\ m^{-1}s^{-1}$'%num2tex.num2tex('%.2e'%mu)
)

plt.figure()
plt.xscale('log')
plt.yscale('log')
ax = plt.gca()
plt.plot(d, t, color='k')
plt.scatter(d0, t0, fc='w', ec='k', s=40, zorder=48)
for i in range(len(d0)):
    plt.text(1.1*d0[i], 1.1*t0[i], f'PM{d0_list[i]}: {sec_to_str(t0[i])}', 
             fontsize=14)
plt.annotate(text,xy=(.03,.5),va='top',xytext=(12,-12),fontsize=14,
    xycoords='axes fraction',textcoords='offset points')
plt.grid()
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
plt.xlabel('Diameter')
plt.ylabel(f'Time needed to descend {l} m')
plt.tight_layout()
plt.savefig('image/drag_model/fall_time.pdf')
plt.close()
#'''