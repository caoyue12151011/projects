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
from scipy.integrate import solve_ivp
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


def calc_E(theta0, vc0, e, J):
    # To calculate total energy given i.c.
    omega0 = vc0
    alpha0 = calc_alpha(theta0, e)
    d0 = calc_d(theta0, e)
    return .5*(d0**2+J)*omega0**2 + d0*np.cos(theta0-alpha0)


def eq(t, theta, e, J, E):
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
    return (2*(E - d*np.cos(theta-alpha)) / (d**2+J)) **.5


def event(t, theta, e, J, E):
    return theta[0] - np.pi/2

event.terminal = True
event.direction = 1


# constants
e = .8
J = (2-e**2)/4  # = J/(ma^2), for homogeneous disc: m(a^2+b^2)/4 = (2-e^2)/4

# initial condition
theta0 = 0
vc0 = .1

# time range
t_f = np.pi/4 / vc0
t_eval = np.linspace(0, t_f, 1000)

# solve
E = calc_E(theta0, vc0, e, J)
sol = solve_ivp(eq, (0,np.inf), [theta0], args=(e,J,E), events=event, 
                t_eval=t_eval)
if not sol.success:
    print('[Error] Integration not successful.')

t = sol.t
theta = sol.y[0]
omega = np.gradient(theta) / np.gradient(t)
beta = np.gradient(omega) / np.gradient(t)


plt.figure()
plt.plot(t, theta)
plt.savefig('theta.pdf')
plt.close()

plt.figure()
plt.plot(t, omega)
plt.savefig('omega.pdf')
plt.close()

plt.figure()
plt.plot(t, beta)
plt.savefig('beta.pdf')
plt.close()


