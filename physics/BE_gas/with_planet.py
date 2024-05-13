'''
To derive the density profile of an isothermal, self-gravitating gas on a planet 
in the hydrostatic equilibrium. See knowledge/gas_with_planet.md for details.
'''

import num2tex
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.special import expi
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Ellipse, Rectangle, Arc, Polygon
from matplotlib.colors import LogNorm


# change default matplotlib fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['font.size'] = 14


def eq_den_prof(y,t,xi_g):
    '''
    The differential equation of the density profile. y[0] = eta_rho, 
    y[1] = xi^2/eta_rho*d(eta_rho)/d(xi), t=xi. See 
    knowledge/isothermal_atmosphere for the definitions of these variables. 
    Returns [dy0dt,dy1dt].
    '''

    y0, y1 = y
    dy0dt = y0*y1/t**2 
    dy1dt = -(t*xi_g)**2*y0

    return [dy0dt,dy1dt]


def integral(x):
    # Returns int u^-4*e^u*du.
    return (expi(x)-(x**2+x+2)*np.exp(x)/x**3)/6


def eq_Mc_eq_Mg(xi,xi_g,xi_H):
    '''
    The equation M_gas_no_self_G(xi) = M_c. See knowledge/isothermal_atmosphere 
    for details. 
    '''

    I = integral(xi_H)-integral(xi_H/xi)
    return xi_H**2*np.exp(-xi_H)*I - 1/xi_g**2


# demo =========================================================================

#''' density profile -----------------------------------------------------------

# parameters
xi_g = 10
xi_H = .01
Xi = np.logspace(0,5,10000) # i.e. r/r_c
Xi_BE0 = np.logspace(-4,5,1000) # for calculating the BE sphere

Ind_BE = Xi_BE0>=Xi.min()
Xi_BE = Xi_BE0[Ind_BE] # Xi for demonstrating the BE solution
R_hat0 = Xi_BE0*xi_g # r/r_BE


# solution for initial plot
Eta_rho = odeint(eq_den_prof,[1,-xi_H],Xi,(xi_g,))[:,0] # rho/rho0

# solution without gas self-G
Eta_rho_ng = np.exp(xi_H*(1/Xi-1))

# solution without planet gravity
Eta_rho_np = odeint(eq_den_prof,[1,0],Xi,(xi_g,))[:,0]

# BE sphere
Eta_rho_BE0 = odeint(eq_den_prof,[1,0],Xi_BE0,(xi_g,))[:,0]
Eta_rho_BE = Eta_rho_BE0[Ind_BE]

# approximated BE sphere
Eta_rho_BE_appr0 = 1/(1+(xi_g*Xi_BE0/2)**2)
Eta_rho_BE_appr = 1/(1+(xi_g*Xi/2)**2)

# find xi where M_gas_noGasGravity = M_c
xi_cr_g = fsolve(eq_Mc_eq_Mg,1.5,args=(xi_g,xi_H))[0]


# density profile of gas+planet ................................................

# figure parameters
w_p, h_p = 6, 4 # [in], sizes of the plot axis 
lf_p = .8
rt_p = .3
up_p = .3
bt_p = .8
h_s = .2 # [in], height of the sliders
lf_s = 1. # [in], left/right margin sizes of the sliders
rt_s = .8
show_all = True  
show_np = True 
show_ng = True 
show_BE = True 
show_BE_appr = False 
show_xi_cr_g = True
show_xi_g_slider = True

# derived figure parameters
fig_x = w_p+lf_p+rt_p 
fig_y = up_p+h_p+bt_p+4*h_s 

fx_p = lf_p/fig_x 
fy_p = (bt_p+4*h_s)/fig_y 
fw_p = w_p/fig_x 
fh_p = h_p/fig_y 

fx_s = lf_s/fig_x 
fy_s1 = 3*h_s/fig_y # upper slider
fy_s2 = h_s/fig_y # lower slider
fw_s = (fig_x-lf_s-rt_s)/fig_x 
fh_s = h_s/fig_y 


fig = plt.figure(figsize=(fig_x,fig_y))

# the plot 
ax = plt.axes([fx_p,fy_p,fw_p,fh_p])
plt.xscale('log')
plt.yscale('log')
ax.set_ylim(Eta_rho.min(),Eta_rho.max()) 

if show_BE:
    line_BE,=ax.plot(Xi_BE,Eta_rho_BE,color='y',alpha=.3,linewidth=10,
        label='BE sphere')

if show_BE_appr:
    line_BE_appr,=ax.plot(Xi,Eta_rho_BE_appr,color='brown',alpha=.3,
        linewidth=10,label='Approx. BE sphere')

if show_ng:
    line_ng,=ax.plot(Xi,Eta_rho_ng,color='r',alpha=.5,linewidth=4,
        label='No gas self-G')

if show_np:
    line_np,=ax.plot(Xi,Eta_rho_np,color='green',alpha=.5,linewidth=4,
        label='No planet gravity') 

if show_all:
    line,=ax.plot(Xi,Eta_rho,color='k',label='All considered')

if show_xi_cr_g:
    vline_xi_cr_g=plt.axvline(xi_cr_g,color='r',linestyle='--') 

    # text of xi_cr_g
    xmin, xmax = plt.xlim()
    x_text = np.log10(xi_cr_g/xmin)/np.log10(xmax/xmin)+.01
    text=plt.text(x_text,.05,r'$\xi_{\rm cr,g}$',color='r',fontsize=16,
        transform=ax.transAxes)

# demo of the planet
ax.set_xlim(ax.get_xlim())
plt.axvspan(.1,1,color='k',alpha=.2)

plt.legend()
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\eta_\rho$')
plt.grid()

# sliders
ax_xi_g = plt.axes([fx_s,fy_s1,fw_s,fh_s])
xi_g_slider = Slider(ax=ax_xi_g,label=r'${\rm log_{10}}\alpha_{\rm s}$',
    valmin=-2,valmax=2,valinit=np.log10(xi_g))
ax_xi_g.set_visible(show_xi_cr_g)

ax_xi_H = plt.axes([fx_s,fy_s2,fw_s,fh_s])
xi_H_slider = Slider(ax=ax_xi_H,label=r'${\rm log_{10}}\alpha_{\rm g}$',
    valmin=-2,valmax=2,valinit=np.log10(xi_H))

# The function to be called anytime a slider's value changes
def update(val):

    # new slider values
    xi_g = 10**xi_g_slider.val 
    xi_H = 10**xi_H_slider.val

    # new solutions
    Eta_rho = odeint(eq_den_prof,[1,-xi_H],Xi,(xi_g,))[:,0]
    Eta_rho_ng = np.exp(xi_H*(1/Xi-1))
    Eta_rho_np = odeint(eq_den_prof,[1,0],Xi,(xi_g,))[:,0]
    Eta_rho_BE = odeint(eq_den_prof,[1,0],Xi_BE0,(xi_g,))[:,0][Ind_BE]
    Eta_rho_BE_appr = 1/(1+(xi_g*Xi/2)**2)
    xi_cr_g = fsolve(eq_Mc_eq_Mg,1.5,args=(xi_g,xi_H))


    # upload plots
    ax.set_ylim(Eta_rho.min(),Eta_rho.max())
    if show_all:
        line.set_ydata(Eta_rho)
    if show_ng:
        line_ng.set_ydata(Eta_rho_ng)
    if show_np:
        line_np.set_ydata(Eta_rho_np)
    if show_BE:
        line_BE.set_ydata(Eta_rho_BE)
    if show_BE_appr:
        line_BE_appr.set_ydata(Eta_rho_BE_appr)
    if show_xi_cr_g:
        vline_xi_cr_g.set_xdata(xi_cr_g)
        x_text = np.log10(xi_cr_g/xmin)/np.log10(xmax/xmin)+.01
        text.set_position((x_text,.05))

    fig.canvas.draw_idle()

# register the update function with each slider
xi_g_slider.on_changed(update)
xi_H_slider.on_changed(update)
plt.show()
#'''


''' demo of Eq M_gas_no_self_G(xi) = M_c (update needed) -----------------------

# parameters
xi_g = 1 
xi_H = 1
Xi = np.logspace(0,5,10000) # i.e. r/r_c

Y = eq_Mc_eq_Mg(Xi,xi_g,xi_H)


# figure parameters
w_p, h_p = 6, 4 # [in], sizes of the plot axis 
lf_p = .8
rt_p = .3
up_p = .3
bt_p = .8
h_s = .2 # [in], height of the sliders
lf_s = 1.3 # [in], left/right margin sizes of the sliders
rt_s = .8

# derived figure parameters
fig_x = w_p+lf_p+rt_p 
fig_y = up_p+h_p+bt_p+4*h_s 

fx_p = lf_p/fig_x 
fy_p = (bt_p+4*h_s)/fig_y 
fw_p = w_p/fig_x 
fh_p = h_p/fig_y 

fx_s = lf_s/fig_x 
fy_s1 = 3*h_s/fig_y # upper slider
fy_s2 = h_s/fig_y # lower slider
fw_s = (fig_x-lf_s-rt_s)/fig_x 
fh_s = h_s/fig_y 


# demo
fig = plt.figure(figsize=(fig_x,fig_y))
ax = plt.axes([fx_p,fy_p,fw_p,fh_p])
plt.ylim(-1,1)
plt.xscale('log')
line, = ax.plot(Xi,Y,color='k')
plt.xlabel(r'$r/r_{\rm c}$')
plt.ylabel(r'$Y$'+' (eq Y=0)')
plt.grid()

# sliders
ax_xi_g = plt.axes([fx_s,fy_s1,fw_s,fh_s])
xi_g_slider = Slider(ax=ax_xi_g,label=r'${\rm log_{10}}(r_{\rm g}/r_{\rm c})$',
    valmin=-2,valmax=2,valinit=np.log10(xi_g))
ax_xi_H = plt.axes([fx_s,fy_s2,fw_s,fh_s])
xi_H_slider = Slider(ax=ax_xi_H,label=r'${\rm log_{10}}(r_{\rm H}/r_{\rm c})$',
    valmin=-2,valmax=2,valinit=np.log10(xi_H))

# The function to be called anytime a slider's value changes
def update(val):

    # new slider values
    xi_g = 10**xi_g_slider.val 
    xi_H = 10**xi_H_slider.val

    # new solutions
    Y = eq_Mc_eq_Mg(Xi,xi_g,xi_H)

    line.set_ydata(Y)
    fig.canvas.draw_idle()

# register the update function with each slider
xi_g_slider.on_changed(update)
xi_H_slider.on_changed(update)
plt.show()
#'''


''' map: xi_cr_g vs xi_g vs xi_H -----------------------------------------------

# parameters
xi_g = np.logspace(-2,2,200) # x axis
xi_H = np.logspace(-2,2,200) # y axis
xi_guess = 1.5 # initial guess


nx = len(xi_g)
ny = len(xi_H)
lg_xi_g = np.log10(xi_g)
lg_xi_H = np.log10(xi_H)

Xi_Mc_eq_Mg = np.full((ny,nx),np.nan)
for i in range(ny):
    for j in range(nx):
        Xi_Mc_eq_Mg[i,j]=fsolve(eq_Mc_eq_Mg,xi_guess,args=(xi_g[j],xi_H[i]))[0]


plt.figure(figsize=(7,6))
plt.imshow(Xi_Mc_eq_Mg,origin='lower',norm=LogNorm(),
    extent=[lg_xi_g.min(),lg_xi_g.max(),lg_xi_H.min(),lg_xi_H.max()])
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'$r/r_{\rm c}$'+' for '+r'$M_{\rm ngg}=M_{\rm c}$')
plt.xlabel(r'${\rm log_{10}}(r_{\rm g}/r_{\rm c})$')
plt.ylabel(r'${\rm log_{10}}(r_{\rm H}/r_{\rm c})$')
plt.tight_layout()
plt.savefig('image/xi_cr_g.pdf')
plt.show()
#'''


#''' xi_cr_g vs xi_H -----------------------------------------------------------

# parameters
xi_g = np.logspace(-2,1,8) # x axis
xi_H = np.logspace(0,1.3,200) # y axis
xi_guess = 1.5 # initial guess
Color = ['r','orange','y','green','c','blue','purple','k']


nx = len(xi_g)
ny = len(xi_H)
lg_xi_g = np.log10(xi_g)
lg_xi_H = np.log10(xi_H)

Xi_Mc_eq_Mg = np.full((ny,nx),np.nan)
for i in range(ny):
    for j in range(nx):
        Xi_Mc_eq_Mg[i,j]=fsolve(eq_Mc_eq_Mg,xi_guess,args=(xi_g[j],xi_H[i]))[0]


plt.figure(figsize=(7,5))
plt.xscale('log')
plt.yscale('log')
for i in range(nx):
    plt.plot(xi_H,Xi_Mc_eq_Mg[:,i],color=Color[i],
        label=r'$\alpha_{\rm s}=%s$'%num2tex.num2tex('%.2f'%xi_g[i]))
plt.grid()
plt.legend()
plt.xlabel(r'$\alpha_{\rm g}$')
plt.ylabel(r'$\xi_{\rm cr,g}$')
plt.tight_layout()
plt.savefig('image/with_planet/xi_cr_g_vs_xi_H.pdf')
plt.close()
#'''
