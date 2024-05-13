'''
To calculate the Maxwell-Juttner distribution, relativistic version of 
the Maxwell-Boltzmann distribution.
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, exp, pi
from matplotlib.widgets import Button, Slider
from scipy.special import kn  # modified Bessel function of the 2nd kind


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def MJ(beta, theta, broadcast=False):
    '''
    The Maxwell-Juttner distribution. beta=v/c, theta=kT/(mc^2). 
    beta & theta can be scalars or 1darrays.
    broadcast: true or false, whether to broadcast the axes of the output.
    '''
    if broadcast:
        theta = theta[..., np.newaxis]  # of shape (theta, beta)
    gamma = (1-beta**2)**-.5
    return 1/(theta * kn(2,1/theta)) * beta**2 * gamma**5 * exp(-gamma/theta)


def MB(beta, theta, broadcast=False):
    '''Same as MJ() but for the Maxwell-Boltzmann distribution.'''
    if broadcast:
        theta = theta[..., np.newaxis]
    return (2/pi)**.5 * beta**2/theta**1.5 * exp(-beta**2/(2*theta))



# parameters 
beta = np.linspace(0, 1, 300)
lg_theta0 = -.11
fct_y_max = 1.15

y0 = MJ(beta, 10**lg_theta0)
y10 = MB(beta, 10**lg_theta0)

fig, ax = plt.subplots()
fig.subplots_adjust(left=.25)
l, = plt.plot(beta, y0, color='k', label='Maxwell-Juttner')
plt.xlim(plt.xlim())
l1, = plt.plot(beta, y10, color='k', ls='--', label='Maxwell-Boltzmann')

y_max = max(np.nanmax(y0), np.nanmax(y10))
ax.set_ylim(0, fct_y_max*y_max)
plt.legend()
plt.text(.05, .5, r'$\theta=\frac{k_{\rm B}T}{mc^2}$', fontsize=16,
         transform=ax.transAxes)
plt.grid()
plt.xlabel(r'$\beta=v/c$')
plt.ylabel('Probability density')

# theta slider
slider = Slider(ax=fig.add_axes([0.07, 0.15, 0.03, 0.75]),
                label=r'$\rm lg\theta$',
                valmin=-2,
                valmax=1,
                valinit=lg_theta0,
                orientation="vertical")

def update(val):
    y = MJ(beta, 10**slider.val)
    y1 = MB(beta, 10**slider.val)
    y_max = max(np.nanmax(y), np.nanmax(y1))

    ax.set_ylim(0, fct_y_max*y_max)
    l.set_ydata(y)
    l1.set_ydata(y1)
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.savefig('MJ.pdf')
plt.show()
