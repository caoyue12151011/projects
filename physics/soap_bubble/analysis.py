import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize 
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Arc, Polygon


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def eq_double_bubble(x,r20):
    t1, r2 = x
    r1 = 10**t1+1

    # derived parameters
    theta1 = np.arctan(3**.5*r2/(2*r1-r2))
    thetam = np.pi/3-theta1
    theta2 = 2*np.pi/3-theta1

    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    s2 = np.sin(theta2)
    c2 = np.cos(theta2)
    sm = np.sin(thetam)
    cm = np.cos(thetam)

    # volume eq
    Eq1 = .5*(1+c1)*r1**3 + .25*s1**2*c1*r1**3 - 1
    Eq2 = .5*(1+c2)*r2**3 + .25*s2**2*c2*r2**3 - r20**3

    if isinstance(Eq1,np.ndarray):
        msk = ~(r1==r2)
        rm = 1/(1/r2-1/r1) # radius of the middle soap film
        Eq1[msk] += (-.5*(1-cm)*rm**3 + .25*sm**2*cm*rm**3)[msk]
        Eq2[msk] += (.5*(1-cm)*rm**3 - .25*sm**2*cm*rm**3)[msk]
    else:
        if not r1==r2:
            rm = 1/(1/r2-1/r1) # radius of the middle soap film
            Eq1 += -.5*(1-cm)*rm**3 + .25*sm**2*cm*rm**3
            Eq2 += .5*(1-cm)*rm**3 - .25*sm**2*cm*rm**3

    Eq = (Eq1**2+Eq2**2)**.5

    return Eq


# global constants 
eta_equal = 2/3*4**(1/3) # r/r0 when r10=r20 
eta2_small = .4*50**(1/3) # r2/r20 when r2->0


# demo =========================================================================

''' 
# parameters
r20 = np.arange(.1,1.1,.1) # (0,1], it's actually r20/r10 since r10=1 
r1 = np.linspace(0,2,300) # assume that r1>=r2
r2 = np.linspace(0,2,300)[...,np.newaxis] # (r2,r1)

plt.figure()
for rr20 in r20:
    Eq = eq_double_bubble(rr20,r1,r2)
    ind2, ind1 = divmod(np.nanargmin(Eq),Eq.shape[0])
    plt.scatter(r1[ind1],r2[ind2])
plt.grid()
plt.xlabel('r1')
plt.ylabel('r2')
plt.show()
#'''


'''
# parameters
r20 = .6 # (0,1], it's actually r20/r10 since r10=1 
t1 = np.linspace(-5,-1.1,300) # = lg(r1-1)
r2 = np.linspace(1e-3,1.08,300)[...,np.newaxis] # (r2,r1)

T1, R2 = np.meshgrid(t1,r2)
Eq = eq_double_bubble([t1,r2],r20)

plt.figure()
plt.imshow(Eq,origin='lower',extent=[t1.min(),t1.max(),r2.min(),r2.max()],
    cmap='coolwarm',aspect='auto')
plt.contour(T1,R2,Eq,50,colors='w',linewidths=.5)
plt.grid()
plt.xlabel('lg(r1-1)')
plt.ylabel('r2')
plt.show()
#'''


''' relations between r20,r10 and r1,r2 

# parameters
r20 = np.logspace(-4,0,100) # (0,1], it's actually r20/r10 since r10=1 
x0 = [-1.7,.67] # t1,r2

# find r1, r2
t1 = np.full(r20.shape,np.nan)
r2 = t1.copy()
for i in range(len(r20)):
    res = minimize(eq_double_bubble,x0,args=r20[i],method='Nelder-Mead')
    if res.success:
        t1[i] = res.x[0]
        r2[i] = res.x[1]
    else:
        print('Warning: %s'%res.message)
r1 = 10**t1+1

# r1/2_rel_vs_r20
plt.figure(figsize=(5,4))
plt.plot(r20,r1,label='Large bubble')
plt.plot(r20,r2/r20,label='Small bubble')

plt.scatter(1,eta_equal,color='k',s=5,zorder=40)
plt.text(1,eta_equal+.02,r'$\frac{2}{3}\sqrt[3]{4}$',fontsize=14)

plt.scatter(0,eta2_small,color='C1',s=5,zorder=40)
plt.text(0,eta2_small+.02,r'$\frac{2}{5}\sqrt[3]{50}$',fontsize=14)

plt.legend()
plt.grid()
plt.xlabel(r'$r_{20}/r_{10}$')
plt.ylabel(r'$r/r_0$')
plt.tight_layout()
plt.savefig('image/r_rel_vs_r20.pdf')
plt.close()
#'''


''' double bubble shapes

# parameters 
r20 = .8
x0 = [-1.7,.67] # t1,r2

# solve the problem
res = minimize(eq_double_bubble,x0,args=r20,method='Nelder-Mead')
t1, r2 = res.x

r1 = 10**t1+1
rm = 1/(1/r2-1/r1)
theta1 = np.arctan(3**.5*r2/(2*r1-r2))
thetam = np.pi/3-theta1
theta2 = 2*np.pi/3-theta1

theta1d = theta1*180/np.pi
theta2d = theta2*180/np.pi
thetamd = thetam*180/np.pi

s1m = rm/np.sin(theta1)*np.sin(2*np.pi/3) # 1-m distance 
s2m = np.sin(np.pi/3)*r2/np.sin(thetam)
s12 = s1m-s2m


# the plot 
fig = plt.figure(figsize=(8,5))
ax = plt.axes([.05,.05,.8,.9])
ax.axis('equal')
ax.axis('off')
ax.plot([-1,2.2,2.2,-1],[-1,-1,1,1],lw=0)

cir1 = Arc((0,0),2*r1,2*r1,theta1=theta1d,theta2=360-theta1d,
    fc='none',ec='k',lw=1) # large bubble 
cir2 = Arc((s12,0),2*r2,2*r2,theta1=theta2d-180,theta2=180-theta2d,
    fc='none',ec='k',lw=1) # small bubble 
cirm = Arc((s1m,0),2*rm,2*rm,theta1=180-thetamd,theta2=180+thetamd,
    fc='none',ec='k',lw=1) # middle bubble 

cir10 = Arc((0,0),2,2,theta1=theta1d,theta2=360-theta1d,
    fc='none',ec='k',lw=1,alpha=.5)
cir20 = Arc((s12,0),2*r20,2*r20,theta1=theta2d-180,theta2=180-theta2d,
    fc='none',ec='k',lw=1,alpha=.5)

ax.add_artist(cir1)
ax.add_artist(cir2)
ax.add_artist(cir10)
ax.add_artist(cir20)
ax.add_artist(cirm)

# sliders
ax_s = plt.axes([.85,.1,.05,.8])
slider = Slider(ax=ax_s,label=r'$r_{20}/r_{10}$',valmin=1e-3,valmax=1,
    valinit=r20,orientation='vertical')
slider.label.set_size(14)

# The function to be called anytime a slider's value changes
def update(val):

    # new slider values
    r20 = slider.val 


    # new solutions
    res = minimize(eq_double_bubble,x0,args=r20,method='Nelder-Mead')
    t1, r2 = res.x

    r1 = 10**t1+1
    rm = 1/(1/r2-1/r1)
    theta1 = np.arctan(3**.5*r2/(2*r1-r2))
    thetam = np.pi/3-theta1
    theta2 = 2*np.pi/3-theta1

    theta1d = theta1*180/np.pi
    theta2d = theta2*180/np.pi
    thetamd = thetam*180/np.pi

    s1m = rm/np.sin(theta1)*np.sin(2*np.pi/3) # 1-m distance 
    s2m = np.sin(np.pi/3)*r2/np.sin(thetam)
    s12 = s1m-s2m


    # upload plots
    ax.clear()
    ax.axis('equal')
    ax.axis('off')
    ax.plot([-1,2.2,2.2,-1],[-1,-1,1,1],lw=0)

    cir1 = Arc((0,0),2*r1,2*r1,theta1=theta1d,theta2=360-theta1d,
        fc='none',ec='k',lw=1) # large bubble 
    cir2 = Arc((s12,0),2*r2,2*r2,theta1=theta2d-180,theta2=180-theta2d,
        fc='none',ec='k',lw=1) # small bubble 
    cirm = Arc((s1m,0),2*rm,2*rm,theta1=180-thetamd,theta2=180+thetamd,
        fc='none',ec='k',lw=1) # middle bubble 
    cir10 = Arc((0,0),2,2,theta1=theta1d,theta2=360-theta1d,
        fc='none',ec='k',lw=1,alpha=.5)
    cir20 = Arc((s12,0),2*r20,2*r20,theta1=theta2d-180,theta2=180-theta2d,
        fc='none',ec='k',lw=1,alpha=.5)

    ax.add_artist(cir1)
    ax.add_artist(cir2)
    ax.add_artist(cir10)
    ax.add_artist(cir20)
    ax.add_artist(cirm)

    fig.canvas.draw_idle()

# register the update function with each slider
slider.on_changed(update)
plt.savefig('image/shape.pdf')
plt.show()
#'''











