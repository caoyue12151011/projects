'''
Solutions of the Bondi-Hoyle accretion.
'''

import sys 
import num2tex
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from matplotlib.widgets import Slider, Button

sys.path.append('../module')
import math_tk
reload(math_tk)


# change default matplotlib fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['font.size'] = 14


def eq_eta(eta,xi,lamda,n):
    ''' 
    The equation f(eta)=0. See BH_accretion for the definitions of the 
    variables. Returns f(eta).
    '''

    y = None
    if n==1:
        y = lamda**2/(2*xi**4*eta**2) + np.log(eta) - 1/xi 
    else:
        y = lamda**2/(2*xi**4*eta**2) + n/(n-1)*(eta**(n-1)-1) - 1/xi 

    return y 



def find_interval(f,x0,q,method,*args):
    '''
    To find the initial interval for the dichotomy search of f(x)=0. 
    
    Inputs
    ------
    f: the function defined on positive numbers.
    x0: initial value to determine the interval
    q: >1, folding factor in each search iteration
    method: can be 'valley', 'slope'

        'valley': returns [x1,x2] s.t. f(x1)*f(x0)<=0 and f(x2)*f(x0)<=0. 
            This is often used when f(x0) is low(high) while f(x1) and f(x2) 
            are high(low).

        'slope': returns [x1,x2] s.t. f(x1)*f(x2)<=0. This is often used when
            f(x1)>=f(x0)>=f(x2) or f(x1)<=f(x0)<=f(x2) 

    Returns
    -------
    x1, x2: the interval (x1<=x2)

    '''

    x1 = x0/q
    x2 = x0*q

    if method=='valley':
        while f(x1,*args)*f(x0,*args)>0:
            x1 /= q 
        while f(x2,*args)*f(x0,*args)>0:
            x2 *= q 

    elif method=='slope':
        while f(x1,*args)*f(x2,*args)>0:
            x1 /= q 
            x2 *= q 
    else:
        x1 = None
        x2 = None

    return x1, x2



def calc_lamda_c(n):
    # To calculate the max accretion rate. n is the polytropic index. Returns 
    # nan if not defined.

    lamda_c = np.nan 
    if n==1:
        lamda_c = np.exp(1.5)/4  
    elif n==5/3:
        lamda_c = .03*15**.5 
    elif 0<n and n<5/3:
        lamda_c = (2/(5-3*n))**((5-3*n)/(2*n-2))/(4*n**1.5) 

    return lamda_c



def BH_accretion(xi,lamda,n):
    '''
    To find the solution of Bondi-Hoyle accretion. 

    Inputs 
    ------
    xi: dimensionless distance to the center, >=0
    lamda: dimensionless accretion rate, >=0
    n: polytropic index, can be any real number  

    Returns
    -------
    beta: dimensionless velocity, 
        = scalar if no solutions (nan) or only one solution 
        = 2-element array if two solutions
    eta: density ratio, same as beta
    '''

    beta = None
    eta = None

    # hydrostatic solutions
    if lamda==0:

        if n<0 or n>1:
            beta = 0 
            eta = ((n-1)/(n*xi)+1)**(1/(n-1))

        elif n==1:
            beta = 0 
            eta = np.exp(1/xi)

        else:
            beta = np.nan 
            eta = np.nan

    # accretion solutions
    else:

        if n>5/3:
            beta = np.nan 
            eta = np.nan

        elif n==0:
            eta = lamda/(2*xi**3)**.5
            beta = lamda/xi**2/eta 

        elif n<0: # one solution
            eta0 = 1 # initial value of find_interval
            eta01, eta02 = find_interval(eq_eta,eta0,10,'slope',*(xi,lamda,n))
            eta = math_tk.dichotomy(eq_eta,eta01,eta02,1e-8,True,*(xi,lamda,n))
            beta = lamda/xi**2/eta 


        else: # 0<n<=5/3 
            lamda_c = calc_lamda_c(n) 

            # solve eta, beta
            if lamda>lamda_c:
                beta = np.nan 
                eta = np.nan

            else:
                if n<1 and xi<=1/n-1: # one solution
                    eta0 = 1 # initial value of find_interval
                    eta01, eta02 = find_interval(eq_eta,eta0,10,'slope',
                        *(xi,lamda,n))
                    eta = math_tk.dichotomy(eq_eta,eta01,eta02,1e-8,True,
                        *(xi,lamda,n))
                    beta = lamda/xi**2/eta 

                else: # two solutions
                    eta0 = (lamda**2/n/xi**4)**(1/(n+1))
                    eta01, eta02 = find_interval(eq_eta,eta0,10,'valley',
                        *(xi,lamda,n))
                    eta1 = math_tk.dichotomy(eq_eta,eta01,eta0,1e-8,True,
                        *(xi,lamda,n))
                    eta2 = math_tk.dichotomy(eq_eta,eta0,eta02,1e-8,True,
                        *(xi,lamda,n))
                    eta = np.array([eta1,eta2])
                    beta = lamda/xi**2/eta 

    return beta, eta


# demo =========================================================================

''' lamda_c vs n --------------------------------------------------------------
# lamda_c is defined on 0<n<=5/3

# parameters
n_c = 5/3
n = np.logspace(-3,np.log10(5/3),1000)

lamda_c = np.array([calc_lamda_c(i) for i in n])
lamda_c_c = calc_lamda_c(n_c)

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(n,lamda_c,color='k')
plt.scatter(n_c,lamda_c_c,fc='w',ec='k',s=16,zorder=45)
plt.text(.4,lamda_c_c,r'$\left(\frac{5}{3},\frac{3\sqrt{15}}{100}\right)$',
    fontsize=16)
plt.text(.01,3000,r'$\rm Slope=-1.5$',fontsize=16)
plt.grid()
plt.xlabel('Polytropic index '+r'$n$')
plt.ylabel(r'$\lambda_{\rm c}$')
plt.tight_layout()
plt.savefig('image/lamda_c_vs_n.pdf')
plt.close()
#'''


''' f(eta) --------------------------------------------------------------------

# parameters
lamda = 1
n = .5
xi = 1 
eta = np.logspace(-4,4,1000)

# solution for initial plot
f_eta = eq_eta(eta,xi,lamda,n)

# demo .........................................................................

# figure parameters
w_p, h_p = 4, 3 # [in], sizes of the plot axis 
lf_p = .9 # margins of the plot axis 
up_p = .4
bt_p = .8
w_s = .3 # width of the sliders
mag_lim = 1.5


# derived figure parameters
fig_x = lf_p+w_p+7*w_s 
fig_y = bt_p+h_p+up_p

fx_p = lf_p/fig_x 
fy_p = bt_p/fig_y 
fw_p = w_p/fig_x 
fh_p = h_p/fig_y 

fx_s1 = (lf_p+w_p+w_s)/fig_x 
fx_s2 = (lf_p+w_p+3*w_s)/fig_x 
fx_s3 = (lf_p+w_p+5*w_s)/fig_x 
fy_s = bt_p/fig_y 
fw_s = w_s/fig_x 
fh_s = h_p/fig_y 


fig = plt.figure(figsize=(fig_x,fig_y))

# f_eta vs eta
ax = plt.axes([fx_p,fy_p,fw_p,fh_p])
plt.xscale('log')
line, = ax.plot(eta,f_eta,color='k')
plt.axhline(0,color='r')
ax.set_ylim(np.nanmin(f_eta)/mag_lim,np.nanmax(f_eta)*mag_lim) 
plt.grid()
plt.xlabel(r'$\eta=\frac{\rho}{\rho_\infty}$')
plt.ylabel('Eq'+r'$(\eta)$')

# log xi slider
ax_s1 = plt.axes([fx_s1,fy_s,fw_s,fh_s])
xi_slider = Slider(ax=ax_s1,label=r'$\rm log$'+'$\\xi$',
    valmin=-4,valmax=4,valinit=np.log10(xi),orientation='vertical')

# lamda slider
ax_s2 = plt.axes([fx_s2,fy_s,fw_s,fh_s])
lamda_slider = Slider(ax=ax_s2,label=r'$\lambda$',
    valmin=0,valmax=5,valinit=lamda,orientation='vertical')

# n slider
ax_s3 = plt.axes([fx_s3,fy_s,fw_s,fh_s])
n_slider = Slider(ax=ax_s3,label=r'$n$',valmin=-2,valmax=2,valinit=n,
    orientation='vertical')


# The function to be called anytime a slider's value changes
def update(val):

    # new slider values
    xi = 10**xi_slider.val 
    lamda = lamda_slider.val 
    n = n_slider.val

    # new solutions
    f_eta = eq_eta(eta,xi,lamda,n)

    # upload plots
    line.set_ydata(f_eta)
    ax.set_ylim(np.nanmin(f_eta)/mag_lim,np.nanmax(f_eta)*mag_lim) 
    fig.canvas.draw_idle()

# register the update function with each slider
xi_slider.on_changed(update)
lamda_slider.on_changed(update)
n_slider.on_changed(update)

plt.savefig('image/Eq_eta.pdf')
plt.show()
#'''


#''' density/velocity profile --------------------------------------------------

# parameters
lamda = .1
lamda_min = 0 
lamda_max0 = 1.5 
    # if lamda_c exists, lamda_max=lamda_max0*lamda_c, else = lamda_max0
n = 1.1
n_min = -3 
n_max = 5
Xi = np.logspace(-3,3,500) 

# solution for initial plot ....................................................

# max accretion rate
lamda_c = calc_lamda_c(n) 

# lamda_max
lamda_max = lamda_max0
if not np.isnan(lamda_c):
    lamda_max *= lamda_c


# velocity/density profiles
Beta1, Beta2, Eta1, Eta2, Xi2 = [],[],[],[],[]
for xi in Xi:
    beta, eta = BH_accretion(xi,lamda,n)
    if isinstance(beta,np.ndarray):
        Beta1.append(beta[0])
        Beta2.append(beta[1])
        Eta1.append(eta[0])
        Eta2.append(eta[1])
        Xi2.append(xi)
    else:
        Beta1.append(beta)
        Eta1.append(eta)

Beta1 = np.array(Beta1)
Beta2 = np.array(Beta2)
Eta1 = np.array(Eta1)
Eta2 = np.array(Eta2)
Xi2 = np.array(Xi2)

Beta = np.concatenate((Beta1,Beta2))
Eta = np.concatenate((Eta1,Eta2))

# demo .........................................................................

# figure parameters
w_p, h_p = 4, 2.5 # [in], sizes of the plot axis 
lf_p = 1 # margins of the plot axis 
rt_p = .4
up_p = .4
bt_p = .8
g_p = .1 # gap between the two plots
w_s = .3 # width of the sliders
g_s = .5 # gap between the two sliders
rt_s = .3 # right margin of the sliders
mag_lim = 1.5 # for setting figure limits


# derived figure parameters
fig_x = lf_p+w_p+rt_p+2*w_s+g_s+rt_s
fig_y = bt_p+2*h_p+g_p+up_p

fx_p = lf_p/fig_x 
fy_p1 = bt_p/fig_y 
fy_p2 = (bt_p+h_p+g_p)/fig_y 
fw_p = w_p/fig_x 
fh_p = h_p/fig_y 

fx_s1 = (lf_p+w_p+rt_p)/fig_x 
fx_s2 = (lf_p+w_p+rt_p+w_s+g_s)/fig_x 
fy_s = bt_p/fig_y 
fw_s = w_s/fig_x 
fh_s = (2*h_p+g_p)/fig_y 


fig = plt.figure(figsize=(fig_x,fig_y))

# beta vs xi  
ax1 = plt.axes([fx_p,fy_p1,fw_p,fh_p])
plt.xscale('log')
plt.yscale('log')
line_b1, = ax1.plot(Xi,Beta1,color='C1')
line_b2, = ax1.plot(Xi2,Beta2,color='C0')
plt.grid()
plt.xlabel('$\\xi$'+r'$=r\frac{p_\infty}{GM\rho_\infty}$')
plt.ylabel(r'$\beta=v\sqrt{\frac{\rho_\infty}{p_\infty}}$')

# eta vs xi  
ax2 = plt.axes([fx_p,fy_p2,fw_p,fh_p])
plt.xscale('log')
plt.yscale('log')
line_e2, = ax2.plot(Xi2,Eta2,color='C0',label='Solution 1')
line_e1, = ax2.plot(Xi,Eta1,color='C1',label='Solution 2')
ax2.get_xaxis().set_ticklabels([])
plt.legend()
plt.grid()
plt.ylabel(r'$\eta=\frac{\rho}{\rho_\infty}$')


# lamda slider
ax_s1 = plt.axes([fx_s1,fy_s,fw_s,fh_s])
lamda_slider = Slider(ax=ax_s1,label=r'$\lambda$',
    valmin=lamda_min,valmax=lamda_max,valinit=lamda,orientation='vertical')
lamda_slider.hline.remove() # remove the initial hline

# lamda slider ticks
ticks_l = [0,lamda_c]
ticklabels_l = ['0',r'$\lambda_{\rm c}$']
ax_s1.set_yticks(ticks_l)
ax_s1.set_yticklabels(ticklabels_l)

# critical values
ax_s1.axhline(0,color='k')
line_lc = ax_s1.axhline(lamda_c,color='k')


# n slider
ax_s2 = plt.axes([fx_s2,fy_s,fw_s,fh_s])
n_slider = Slider(ax=ax_s2,label=r'$n$',valmin=n_min,valmax=n_max,valinit=n,
    orientation='vertical')
n_slider.hline.remove() # remove the initial hline

# n slider ticks
ticks_n = list(np.arange(int(np.ceil(n_min)),int(np.ceil(n_max))))
ticklabels_n = [str(i) for i in ticks_n]
ticks_n.append(5/3)
ticklabels_n.append('5/3')
ax_s2.set_yticks(ticks_n)
ax_s2.set_yticklabels(ticklabels_n)

# critical values
ax_s2.axhline(0,color='k')
ax_s2.axhline(1,color='k')
ax_s2.axhline(5/3,color='k')


# The function to be called anytime a slider's value changes
def update(val):

    # update lamda slider ......................................................

    n = n_slider.val

    lamda_c = calc_lamda_c(n) # max accretion rate

    # lamda_max
    lamda_max = lamda_max0
    if not np.isnan(lamda_c):
        lamda_max *= lamda_c

    ax_s1.set_ylim(lamda_min,lamda_max)
    lamda_slider.valmin = lamda_min
    lamda_slider.valmax = lamda_max
    line_lc.set_ydata(lamda_c)

    # lamda slider ticks
    ticks_l = [0,lamda_c]
    ax_s1.set_yticks(ticks_l)
    ax_s1.set_yticklabels(ticklabels_l)

    # update the plots .........................................................

    lamda = lamda_slider.val 

    Beta1, Beta2, Eta1, Eta2, Xi2 = [],[],[],[],[]
    for xi in Xi:
        beta, eta = BH_accretion(xi,lamda,n)
        if isinstance(beta,np.ndarray):
            Beta1.append(beta[0])
            Beta2.append(beta[1])
            Eta1.append(eta[0])
            Eta2.append(eta[1])
            Xi2.append(xi)
        else:
            Beta1.append(beta)
            Eta1.append(eta)

    Beta1 = np.array(Beta1)
    Beta2 = np.array(Beta2)
    Eta1 = np.array(Eta1)
    Eta2 = np.array(Eta2)
    Xi2 = np.array(Xi2)

    Beta = np.concatenate((Beta1,Beta2))
    Eta = np.concatenate((Eta1,Eta2))


    # upload plots
    line_b1.set_ydata(Beta1)
    line_b2.set_xdata(Xi2)
    line_b2.set_ydata(Beta2)
    line_e1.set_ydata(Eta1)
    line_e2.set_xdata(Xi2)
    line_e2.set_ydata(Eta2)

    if np.nanmax(Beta)>np.nanmin(Beta):
        ax1.set_ylim(np.nanmin(Beta)/mag_lim,np.nanmax(Beta)*mag_lim) 
    if np.nanmax(Eta)>np.nanmin(Eta):
        ax2.set_ylim(np.nanmin(Eta)/mag_lim,np.nanmax(Eta)*mag_lim) 

    fig.canvas.draw_idle()

# register the update function with each slider
n_slider.on_changed(update)
lamda_slider.on_changed(update)

plt.savefig('image/solution.pdf')
plt.show()
#'''


