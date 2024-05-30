"""
To calculate the motion of a rolling ellipse.

TODO
1. kinematics: v vs t (vc0, e), v_bar vs e
2. animation: rolling ellipse of diff e's
3. Force analysis
    N/f vs t (vc0, e)
    jump limit: v_bar_max vs e
    mu limit: mu_min vs e
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from importlib import reload
from matplotlib.patches import Circle, Arc
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp, odeint
from scipy.special import ellipeinc
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def calc_alpha(theta, e):
    # theta, alpha in [0, pi/2]
    return np.arctan((1-e**2) * np.tan(theta))


def calc_d(theta, e):
    t = np.tan(theta)
    return ((1 + (1-e**2)**2*t**2) / (1 + (1-e**2)*t**2))**.5


def calc_E(theta, omega, e, J):
    # To calculate total energy
    alpha = calc_alpha(theta, e)
    d = calc_d(theta, e)
    return .5*(d**2+J)*omega**2 + d*np.cos(theta-alpha)


def eq(theta, e, J, E0):
    """
    The equation of the motion of a rolling ellipse.

    Inputs
    ------
    t: time
    theta: [0, pi/2], position angle of the ellipse

    Returns
    -------
    dtheta_dt: time derivative of theta
    """
    alpha = calc_alpha(theta, e)
    d = calc_d(theta, e)
    return (2*(E0 - d*np.cos(theta-alpha)) / (d**2+J)) **.5


def solve(theta0, omega0, e, J):
    E0 = calc_E(theta0, omega0, e, J)
    th = theta0
    theta = [th]
    while th < np.pi/2:
        th += eq(th, e, J, E0) * dt
        theta.append(th)
    theta.pop()  # pop the last value that > pi/2
    return np.array(theta)


def data_packing(t, E, theta, omega, beta, x, y, vcx, f, N, mu):
    return {
        't': {
            'data': t,
            'label': r'$\hat t$',
        },
        'E': {
            'data': E,
            'label': r'$\hat E$',
        },
        'theta': {
            'data': theta * 180/np.pi,
            'label': r'$\theta\rm (^\circ)$',
        },
        'omega': {
            'data': omega,
            'label': r'$\hat\omega$',
        },
        'beta': {
            'data': beta,
            'label': r'$\hat\beta$',
        },
        'x': {
            'data': x,
            'label': r'$\hat x$',
        },
        'y': {
            'data': y,
            'label': r'$\hat y$',
        },
        'vcx': {
            'data': vcx,
            'label': r'$\hat v_{cx}$',
        },
        'f': {
            'data': f,
            'label': r'$\hat f$',
        },
        'N': {
            'data': N,
            'label': r'$\hat N$',
        },
        'mu': {
            'data': mu,
            'label': r'$\mu$',
        },
    }


# parameters
e = .8
J = (2-e**2)/4  # = J/(ma^2), for homogeneous disc: m(a^2+b^2)/4 = (2-e^2)/4

# initial condition
theta0 = 0
omega0 = .1  # = vcx0 bc a=1 

# t step
dt = .0001

# derived parameters
b = (1-e**2)**.5
E0 = calc_E(theta0, omega0, e, J)

# solve
theta = solve(theta0, omega0, e, J)
print(f'len(theta)={len(theta)}.')

t = np.arange(len(theta)) * dt
omega = np.gradient(theta) / dt
beta = np.gradient(omega) / dt
alpha = calc_alpha(theta, e)
E = calc_E(theta, omega, e, J)
d = calc_d(theta, e)
s = b * ellipeinc(np.arctan(np.tan(alpha)/b), 1-b**-2)
x = s + d*np.sin(theta-alpha)
y = d*np.cos(theta-alpha)
vc = d*omega
vcx = vc*np.cos(theta-alpha)
vcy = -vc*np.sin(theta-alpha)
acx = np.gradient(vcx) / dt
acy = np.gradient(vcy) / dt
f = acx
N = acy + 1
mu = f/N

Data = data_packing(t, E, theta, omega, beta, x, y, vcx, f, N, mu)

# demo ========================================================================

#''' a vs b plot
pairs = [['t', 'E'],  # [x, y]
         ['t', 'theta'],
         ['t', 'omega'],
         ['t', 'beta'],
         ['t', 'f'],
         ['t', 'N'],
         ['t', 'mu'],
         ['N', 'mu'],
         ['x', 'y']]

for name_x, name_y in pairs:
    data_x = Data[name_x]['data']
    data_y = Data[name_y]['data']
    label_x = Data[name_x]['label']
    label_y = Data[name_y]['label']

    plt.figure()
    if name_x=='x' and name_y=='y':
        plt.axis('equal')
    plt.plot(data_x, data_y, color='k')
    plt.text(.1, .1, r'$e=$'+f'{e:.2}', fontsize=14, 
             transform=plt.gca().transAxes)
    plt.grid()
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.tight_layout()
    plt.savefig(f'image/relation/{name_x}_vs_{name_y}.pdf')
    plt.close()
#''''
