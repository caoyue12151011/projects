'''

'''

import num2tex
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider, Button


# change default matplotlib fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['font.size'] = 14


def eq(y,xi,eta0,gamma):
    '''

    '''

    y0, y1 = y
    # dy0 = y0/xi**2*(y0-eta0)**(1-1/gamma)*y1 
    # dy1 = -xi**2*(1-eta0)**(1/gamma)*gamma*y0

    dy0 = y1/(xi**2/y0*(y0-eta0)**(1/gamma-1))
    dy1 = -xi**2*y0

    return [dy0,dy1]


def find_xi_p(xi,eta):
    ind = np.nan 
    xi_p = np.nan 

    if np.any(np.isnan(eta)):
        ind = np.where(np.isnan(eta))[0][0]
        xi_p = xi[ind]

    return ind, xi_p




# global parameters 
ic = [1.,0.] # initial condition [eta,d(eta)/d(xi)] 


# demo =========================================================================


''' density profile 

# parameters 
eta0 = .9
gamma = 1
Xi = np.logspace(-6,6,10000) # i.e. r/r_hat, Xi[0] is small (should be 0)  


# density profile
Eta = odeint(eq,ic,Xi,args=(eta0,gamma))[:,0] 


# figure parameters
w_p, h_p = 5, 3 # [in], sizes of the plot axis 
lf_p = .9 # margins of the plot axis 
rt_p = .4
tp_p = .4
bt_p = .8
h_s = .2 # height of the sliders
lf_s = .9 
rt_s = .9 


# derived figure parameters
fig_x = lf_p+w_p+rt_p
fig_y = tp_p+h_p+bt_p+4*h_s 
fx_p = lf_p/fig_x 
fy_p = (bt_p+4*h_s)/fig_y 
fw_p = w_p/fig_x 
fh_p = h_p/fig_y 
fx_s = lf_s/fig_x
fy_s1 = 3*h_s/fig_y 
fy_s2 = h_s/fig_y 
fw_s = 1-(lf_s+rt_s)/fig_x 
fh_s = h_s/fig_y 


# demo
fig = plt.figure(figsize=(fig_x,fig_y))
ax = plt.axes([fx_p,fy_p,fw_p,fh_p])
plt.xlim(Xi.min(),Xi.max())
plt.ylim(-.3,1.1)
plt.xscale('log')

l, = ax.plot(Xi,Eta,color='k',lw=1)
plt.grid()
plt.xlabel('$\\xi$')
plt.ylabel(r'$\rho/\rho_{\rm c}$')


# eta0 slider
ax_s1 = plt.axes([fx_s,fy_s1,fw_s,fh_s])
sld1 = Slider(ax=ax_s1,label=r'$\eta_0$',
    valmin=0,valmax=1,valinit=eta0,orientation='horizontal')

# gamma slider
ax_s2 = plt.axes([fx_s,fy_s2,fw_s,fh_s])
sld2 = Slider(ax=ax_s2,label=r'$\gamma$',
    valmin=0,valmax=5,valinit=gamma,orientation='horizontal')


# The function to be called anytime a slider's value changes
def update(val):

    # new slider values
    eta0 = sld1.val 
    gamma = sld2.val 

    # new solutions
    Eta = odeint(eq,ic,Xi,args=(eta0,gamma))[:,0] 

    # upload plots
    l.set_ydata(Eta)
    fig.canvas.draw_idle()

# register the update function with each slider
sld1.on_changed(update)
sld2.on_changed(update)

plt.savefig('image/density_profile.pdf')
plt.show()
#'''


''' M-R relation  

# parameters 
gamma = 2
p = 6
Xi = np.logspace(-6,6,1000) # i.e. r/r_hat, Xi[0] is small (should be 0)  
x = np.linspace(-20,10,200)

Eta0 = p**x/2*(x<0) + (1-p**-x/2)*(x>=0)


# find xi_p 
Xi_p = []
ETA = []
for i in range(len(Eta0)):
    Eta = odeint(eq,ic,Xi,args=(Eta0[i],gamma))[:,0]
    ind, xi_p = find_xi_p(Xi,Eta)
    if not np.isnan(ind):
        Eta[ind:] = 0. 

    ETA.append(Eta)
    Xi_p.append(xi_p)
ETA = np.array(ETA)
Xi_p = np.array(Xi_p)


# planet size & mass
M = gamma**-1.5 * Eta0**(2-1.5/gamma) *np.trapz(Xi**2*ETA,Xi)
Ri = (3*M)**(1/3) # incompressible planet 
R = gamma**-.5 * Eta0**(1-.5/gamma)*Xi_p

# singular solutions
Rs = np.nan 
Ms = np.nan 
if gamma>.75:
    Rs = (2*(4*gamma-3))**.5/(2*gamma-1)
    Ms = (8*(4*gamma-3))**.5/(2*gamma-1)**2


# xi_p vs eta0
# Xi_p_s = (2*gamma*(4*gamma-3))**.5/(2*gamma-1)*Eta0**(.5/gamma-1)

# plt.figure()
# plt.xscale('log')
# plt.yscale('log')
# plt.scatter(Eta0,Xi_p,s=2)
# plt.plot(Eta0,Xi_p_s)
# plt.xlabel(r'$\eta_0$')
# plt.ylabel(r'$\xi_p$')
# plt.show()



# figure parameters
wp1 = 4 # [in], sizes of the plot axis 
wp2 = 3
hp = 3
lp = .8 # margins of the plot axis 
rp = .4
gp = 1.1
tp = .4
bp = .8
hs = .2 # height of the sliders
ls = .9 
rs = .9 
bs = .4
fct_lim = 1.2


# derived figure parameters
fig_x = lp+wp1+gp+wp2+rp 
fig_y = tp+hp+bp+hs+bs
fxp1 = lp/fig_x 
fxp2 = (lp+wp1+gp)/fig_x 
fyp = (bp+hs+bs)/fig_y 
fwp1 = wp1/fig_x 
fwp2 = wp2/fig_x 
fhp = hp/fig_y 
fxs = ls/fig_x
fys = bs/fig_y 
fws = 1-(ls+rs)/fig_x 
fhs = hs/fig_y 


# demo
fig = plt.figure(figsize=(fig_x,fig_y))


# M-R
ax1 = plt.axes([fxp1,fyp,fwp1,fhp])
ax1.set_xlim(np.nanmin(M)/fct_lim,fct_lim*np.nanmax(M))
ax1.set_ylim(0,fct_lim*np.nanmax(R))
ax1.set_xscale('log')
l_MR, = ax1.plot(M,R,color='k',lw=1)
l_MRi, = ax1.plot(M,Ri,color='b',lw=1,label='Incompressible')
dot = ax1.scatter(Ms,Rs,fc='r',ec='none',s=18,label='Singular')
ax1.grid()
ax1.legend()
ax1.set_xlabel(r'$\tilde M$')
ax1.set_ylabel(r'$\tilde R$')


# R-eta0
ax2 = plt.axes([fxp2,fyp,fwp2,fhp/2])
ax2.set_xscale('log')
ax2.set_ylim(0,fct_lim*np.nanmax(R))
l_Re, = ax2.plot(Eta0,R,color='k',lw=1)
l_Rs = ax2.axhline(Rs,color='r',lw=1,ls='--',label='Singular')
ax2.grid()
ax2.legend()
ax2.set_xlabel(r'$\eta_{\rho0}$')
ax2.set_ylabel(r'$\tilde R$')


# M-eta0 
ax3 = plt.axes([fxp2,fyp+fhp/2,fwp2,fhp/2])
ax3.set_xscale('log')
ax3.set_ylim(np.nanmin(M)/fct_lim,fct_lim*np.nanmax(M))
l_Me, = ax3.plot(Eta0,M,color='k',lw=1)
l_Ms = ax3.axhline(Ms,color='r',lw=1,ls='--')
ax3.grid()
ax3.set_xticklabels([])
ax3.set_ylabel(r'$\tilde M$')


# gamma slider
ax_s = plt.axes([fxs,fys,fws,fhs])
sld = Slider(ax=ax_s,label=r'$\gamma$',
    valmin=0,valmax=3,valinit=gamma,orientation='horizontal')
sld.vline._linewidth = 0

# critical gamma
tick_s = [.5,.75]
ticklabel_s = [str(i) for i in tick_s]
sld.ax.xaxis.set_ticks(tick_s)
sld.ax.xaxis.set_ticklabels(ticklabel_s)


# The function to be called anytime a slider's value changes
def update(val):

    # new slider values
    gamma = sld.val 


    # find xi_p 
    Xi_p = []
    ETA = []
    for i in range(len(Eta0)):
        Eta = odeint(eq,ic,Xi,args=(Eta0[i],gamma))[:,0]
        ind, xi_p = find_xi_p(Xi,Eta)
        if not np.isnan(ind):
            Eta[ind:] = 0. 

        ETA.append(Eta)
        Xi_p.append(xi_p)
    ETA = np.array(ETA)
    Xi_p = np.array(Xi_p)


    # planet size & mass
    M = gamma**-1.5 * Eta0**(2-1.5/gamma) *np.trapz(Xi**2*ETA,Xi)
    Ri = (3*M)**(1/3) # incompressible planet 
    R = gamma**-.5 * Eta0**(1-.5/gamma)*Xi_p

    # singular solutions
    Rs = np.nan 
    Ms = np.nan 
    if gamma>.75:
        Rs = (2*(4*gamma-3))**.5/(2*gamma-1)
        Ms = (8*(4*gamma-3))**.5/(2*gamma-1)**2


    # upload plots
    ax1.set_xlim(np.nanmin(M)/fct_lim,fct_lim*np.nanmax(M))
    ax1.set_ylim(0,fct_lim*np.nanmax(R))
    l_MR.set_xdata(M)
    l_MR.set_ydata(R)
    l_MRi.set_xdata(M)
    l_MRi.set_ydata(Ri)
    dot.set_offsets([Ms,Rs])

    ax2.set_ylim(0,fct_lim*np.nanmax(R))
    l_Re.set_ydata(R)
    l_Rs.set_ydata(Rs)

    ax3.set_ylim(np.nanmin(M)/fct_lim,fct_lim*np.nanmax(M))
    l_Me.set_ydata(M)
    l_Ms.set_ydata(Ms)

    fig.canvas.draw_idle()

# register the update function with each slider
sld.on_changed(update)

plt.savefig('image/M_R.pdf')
plt.show()
#'''












