# See knowledge/projectile_with_drag.md. 
import matplotlib   
import numpy as np 
import matplotlib.pyplot as plt 
from numpy import sin, cos, tan
from matplotlib.widgets import Slider, Button
from scipy.special import lambertw


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


''' Lambert W function
x = np.linspace(-5,5,1000)
y = lambertw(x)

plt.figure(figsize=(5,4))
plt.plot(x,y,color='k')
plt.grid()
plt.tight_layout()
plt.savefig('image/lambert_W.pdf')
plt.close()
#'''


''' range_vs_angle

# parameters
gamma = 10 
theta_d = np.linspace(0,90,1000)
theta = theta_d*np.pi/180 

tmp = 1 + gamma*sin(theta)
x = cos(theta)/tmp*lambertw(-tmp*np.exp(-tmp)) + cos(theta)

plt.figure()
plt.axes([.12,.2,.83,.75])
l,=plt.plot(theta_d,x,color='k')
plt.grid()
plt.xlabel(r'$\theta\ (^\circ)$')
plt.ylabel(r'$\hat x$')
plt.savefig('image/range_vs_angle.pdf')
sld = Slider(ax=plt.axes([.1,.03,.8,.04]),label=r'$\Gamma$',valmin=0,
    valmax=gamma,valinit=1)

def update(val):

    # new solutions
    gamma = sld.val 
    tmp = 1 + gamma*sin(theta)
    x = cos(theta)/tmp*lambertw(-tmp*np.exp(-tmp)) + cos(theta)

    # update plots
    l.set_ydata(x)
    fig.canvas.draw_idle()

sld.on_changed(update)
plt.show()
#'''


''' best angle vs gamma

# parameters
gamma = np.logspace(-4,5,1000)
theta_d = np.linspace(0,90,1000)

theta = theta_d*np.pi/180
c = cos(theta) 
s = sin(theta)

theta_dm = np.full(len(gamma),np.nan)
for i in range(len(gamma)):
    tmp = 1 + gamma[i]*s
    x = c/tmp*lambertw(-tmp*np.exp(-tmp)) + c
    theta_dm[i] = theta_d[np.nanargmax(x)]

# theta_dm1 = 45-theta_dm[::-1]

plt.figure(figsize=(5,4))
plt.ylim(0,50)
plt.plot(gamma,theta_dm,color='k')
# plt.plot(gamma*2.27,theta_dm1,color='r')
plt.axhline(22.5,color='k',ls='--')
plt.scatter(0,45,color='k')
plt.grid()
plt.xlabel(r'$\Gamma$')
plt.ylabel(r'$\theta={\rm argmax}(\hat x)$')
plt.tight_layout()
plt.savefig('image/best_angle_vs_gamma.pdf')
plt.xscale('log')
plt.savefig('image/best_angle_vs_gamma_log.pdf')
plt.show()
#'''









