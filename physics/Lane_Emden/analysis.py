'''
To solve the Lane-Emden equation (see 
https://en.wikipedia.org/wiki/Lane%E2%80%93Emden_equation).
'''

import num2tex
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Ellipse, Rectangle, Arc, Polygon


# change default matplotlib fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['font.size'] = 14


def eq_LE(y,xi,n):
    ''' 
    The Lane-Emden equation. y[0]=theta, y[1]=d(theta)/d(xi). Returns 
    [dy0/dxi,dy1/dxi].
    '''

    y0, y1 = y
    dy0 = y1
    dy1 = -y0**n - 2/xi*y1

    return [dy0,dy1]



# global parameters 
Xi = np.logspace(-6,1,10000) # i.e. r/r_hat, Xi[0] is small (should be 0)  
ic = [1,0] # initial condition [theta,d(theta)/d(xi)]


# demo =========================================================================


#''' density profile vs n ------------------------------------------------------

# parameters
n = 2 # initial polytropic index


# solution for initial plot
Theta = odeint(eq_LE,ic,Xi,(n,))[:,0]
Eta = Theta**n # rho/rho0


# figure parameters
w_p, h_p = 6, 4 # [in], sizes of the plot axis 
lf_p = .8
rt_p = .3
up_p = .3
bt_p = .8
h_s = .2 # [in], height of the sliders
lf_s = 1. # [in], left/right margin sizes of the sliders
rt_s = .8

# derived figure parameters
fig_x = w_p+lf_p+rt_p 
fig_y = up_p+h_p+bt_p+2*h_s 
fx_p = lf_p/fig_x 
fy_p = (bt_p+2*h_s)/fig_y 
fw_p = w_p/fig_x 
fh_p = h_p/fig_y 
fx_s = lf_s/fig_x 
fy_s = h_s/fig_y
fw_s = (fig_x-lf_s-rt_s)/fig_x 
fh_s = h_s/fig_y 


fig = plt.figure(figsize=(fig_x,fig_y))

# the plot 
ax = plt.axes([fx_p,fy_p,fw_p,fh_p])
# plt.xscale('log')
# plt.yscale('log')
line,=ax.plot(Xi,Eta,color='k')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\eta_\rho$')
plt.grid()


# sliders
ax_n = plt.axes([fx_s,fy_s,fw_s,fh_s])
n_slider = Slider(ax=ax_n,label=r'$n$',valmin=-2,valmax=10,valinit=n)

# The function to be called anytime a slider's value changes
def update(val):

    # new slider values
    n = n_slider.val 

    # new solutions
    Theta = odeint(eq_LE,ic,Xi,(n,))[:,0]
    Eta = Theta**n # rho/rho0

    # upload plots
    line.set_ydata(Eta)
    fig.canvas.draw_idle()

# register the update function with each slider
n_slider.on_changed(update)
plt.show()
#'''

