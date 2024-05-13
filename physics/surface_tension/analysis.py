import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, sign, pi
from numpy import log as ln


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def surface_curve(y,theta):
    ''' 
    Return the curve function x(y) of the liquid surface. 
    y: sqrt(a)*y
    x: sqrt(a)*x, dimensionless x, a=rho*g/gamma
    theta: [rad], contact angle
    '''

    s = sin(theta)
    C = (2*(1+s))**.5 - ln(((1+s)**.5+2**.5)/(1-s)**.5)
    print(C)

    return C + ln(((4-y**2)**.5+2)/(sign(pi/2-theta)*y)) - (4-y**2)**.5 


# demo -------------------------------------------------------------------------

''' sketch plot

# parameters
c_wall = 'silver'
c_liquid = 'azure'
theta = 20 # [deg]

theta_r = theta*pi/180
y0 = (2*(1-sin(theta_r)))**.5 * sign(pi/2-theta_r)
y = np.linspace(0,y0,1000)
x = surface_curve(y,theta_r)

# demo
plt.figure(figsize=(7,3))
plt.axis('equal')
plt.plot(x,y,color='k')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.fill_between(x,-1e5*np.ones_like(x),y,fc=c_liquid,ec='none')
plt.axvspan(-10,0,fc=c_wall,ec='k')
plt.axhline(0,c='k',lw=1)
plt.gca().axis('off')
plt.tight_layout()
plt.savefig('image/sketch_plot.pdf')
plt.close()
#'''

''' surface_curve

y = np.linspace(-2,2,1000)
x = surface_curve(y,0)

plt.figure(figsize=(7,3))
plt.axis('equal')
plt.plot(x,y,color='k')
text = r'$\hat{x}={\rm ln}\frac{\sqrt{4-\hat{y}^2}+2}{|\hat{y}|}$' 
text += r'$-\sqrt{4-\hat{y}^2}+C$'
plt.text(.4,.5,text,fontsize=16,transform=plt.gca().transAxes)
plt.xlabel(r'$\hat{x}=\sqrt{a}x$')
plt.ylabel(r'$\hat{y}=\sqrt{a}y$')
plt.grid()
plt.tight_layout()
plt.savefig('image/surface_curve.pdf')
plt.close()
#'''


''' surface_curve cases

# parameters
g = 9.81 # [m/s^2]
c_wall = 'turquoise'
Cases = {
    'water_glass': {
        'rho': 998, # [kg/m^3]
        'gamma': .0728, # [N/m] 
        'theta': 10, # [deg]
        'color': 'azure',
    },

    'mercury_glass': {
        'rho': 13500, # [kg/m^3]
        'gamma': .4865, # [N/m] 
        'theta': 135,
        'color': 'silver',
    },
}

for case in Cases:
    liquid, solid = case.split('_')
    liquid = liquid.capitalize()
    solid = solid.capitalize()

    rho = Cases[case]['rho']
    gamma = Cases[case]['gamma']
    theta = Cases[case]['theta']
    color = Cases[case]['color']

    theta_r = theta*pi/180
    a = rho*g/gamma 
    y0 = (2*(1-sin(theta_r))/a)**.5 * sign(pi/2-theta_r) # [m]

    y = np.linspace(0,y0,1000) # [m]
    x = surface_curve(a**.5*y,theta_r)/a**.5 # [m]


    # demo
    plt.figure(figsize=(7,3))
    plt.axis('equal')
    plt.plot(x*1e3,y*1e3,color='k')
    ax = plt.gca()

    text = r'$\sqrt{a}x={\rm ln}\frac{\sqrt{4-ay^2}+2}{\sqrt{a}|y|}$' 
    text += r'$-\sqrt{4-ay^2}+C$'+'\n'
    text += r'$a=\frac{\rho g}{\gamma}$'+'\n'
    text += r'$\rho=%.1f\rm\ g\ cm^{-3}$'%(rho/1e3)+'\n'
    text += r'$\gamma=%.1f\rm\ mN\ m^{-1}$'%(gamma*1e3)+'\n'
    text += 'Contact angle=%d°'%theta
    plt.text(.5,.4,text,fontsize=12,transform=ax.transAxes)

    plt.xlim(-.5,10)
    plt.ylim(plt.ylim())

    plt.fill_between(x*1e3,-1e5*np.ones_like(x),y*1e3,fc=color,ec='none')
    plt.axvspan(-10,0,fc=c_wall,ec='none')
    plt.axhline(0,ls='--',c='k',lw=1)

    plt.text(.05,.4,solid,fontsize=16,weight='bold',transform=ax.transAxes,
        rotation='vertical')
    plt.text(.2,.2,liquid,fontsize=16,weight='bold',transform=ax.transAxes)


    plt.xlabel(r'$x$'+' (mm)')
    plt.ylabel(r'$y$'+' (mm)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('image/surface_curve_%s.pdf'%case)
    plt.close()
#'''

















