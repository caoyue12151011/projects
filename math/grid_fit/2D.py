import matplotlib
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.special import erf
from scipy.optimize import curve_fit


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


np.random.seed(0)


def gaussian(xy,A,mu_x,mu_y,fwhm):
    # x, y, z are 1D arrays. z is organized as (y,x)

    x, y = xy
    y = y[:,np.newaxis]
    z = A*np.exp(-4*np.log(2)*( ((x-mu_x)**2+(y-mu_y)**2)/fwhm**2) )

    return z.ravel()


def gridding(delta,xi_x,xi_y,lim):
    '''
    To grid function gaussian(x,y,1,0,0,1). Returns xy_g and G. xy_g=(xg,yg)
    xg,yg,G are (y,x) 1D arrays.
    '''
    
    # i, j ranges
    i1 = int(np.floor((-lim-xi_x)/delta)) 
    i2 = int(np.ceil((lim-xi_x)/delta))
    j1 = int(np.floor((-lim-xi_y)/delta))
    j2 = int(np.ceil((lim-xi_y)/delta))

    # grid points 
    xg = xi_x + delta*np.arange(i1,i2+1)
    yg = xi_y + delta*np.arange(j1,j2+1)
    yg1 = yg[:,np.newaxis]

    # gridded function
    a = (np.log(2))**.5
    G = np.pi/(4*a*delta)**2 * (erf(a*(2*xg+delta)) - erf(a*(2*xg-delta)))
    G = G * (erf(a*(2*yg1+delta)) - erf(a*(2*yg1-delta)))

    return (xg,yg), G.ravel()


# gridding & fitting -----------------------------------------------------------

# parameters
lim = 5 # range of x [-lim,lim] for the gridding&fitting
Delta = np.linspace(.05,2,50) # channel width
N = 60 # number of MC simulations

Phi_x = np.random.uniform(-.5,.5,N)
Phi_y = np.random.uniform(-.5,.5,N)

Xi_x = Phi_x[:,np.newaxis]*Delta # shape (N, delta)
Xi_y = Phi_y[:,np.newaxis]*Delta # shape (N, delta)

AA = np.full(Xi_x.shape,np.nan)
Mu_x = AA.copy()
Mu_y = AA.copy()
Fwhm = AA.copy()

for i in range(N):
#     if not i==0:
#         continue
    for j in range(len(Delta)):
#         if not j==0:
#             continue 
        xy_g, G = gridding(Delta[j],Xi_x[i,j],Xi_y[i,j],lim)
            
        try:
            res = curve_fit(gaussian,xy_g,G,[1,0,0,1])
            AA[i,j],Mu_x[i,j],Mu_y[i,j],Fwhm[i,j] = res[0]
        except:
            None

# remove bad fittings
msk = Fwhm<.9
AA[msk] = np.nan 
Mu_x[msk] = np.nan 
Mu_y[msk] = np.nan 
Fwhm[msk] = np.nan 

Vol = AA*Fwhm**2
Dist = (Mu_x**2+Mu_y**2)**.5


# demo -------------------------------------------------------------------------

# the data
Para = {
    'A': {
        'data': AA,
        'ylabel': r'$A/A_0$',
    },
    'mu_x': {
        'data': Mu_x,
        'ylabel': r'$\mu_x/FWHM_0$',
    },
    'mu_y': {
        'data': Mu_y,
        'ylabel': r'$\mu_y/FWHM_0$',
    },
    'dist': {
        'data': Dist,
        'ylabel': r'$\sqrt{\mu_x^2+\mu_y^2}/FWHM_0$',
    },    
    'fwhm': {
        'data': Fwhm,
        'ylabel': r'$FWHM/FWHM_0$',
    },
    'volume': {
        'data': Vol,
        'ylabel': r'$A\cdot FWHM^2/(A_0\cdot FWHM_0^2)$',
    },
}



#''' diagram for demo 

# parameters 
delta = 1 # [FWHM0]
phi_x = -.4
phi_y = .4
lim = 1

xi_x = phi_x*delta
xi_y = phi_y*delta

# i, j ranges
i1 = int(np.floor((-lim-xi_x)/delta)) 
i2 = int(np.ceil((lim-xi_x)/delta))
j1 = int(np.floor((-lim-xi_y)/delta))
j2 = int(np.ceil((lim-xi_y)/delta))
I = np.arange(i1,i2+1)
J = np.arange(j1,j2+1)

# grid points
(xg,yg), G = gridding(delta,xi_x,xi_y,lim)
Xg, Yg = np.meshgrid(xg,yg)

# Gaussian contours
lvl = np.arange(.1,1,.3)
r = (-np.log(lvl)/(4*np.log(2)))**.5 # contour circle radii


# demo
plt.figure(figsize=(5.5,5))
ax = plt.gca()

# grid points
plt.scatter(Xg,Yg,color='r',s=10)
plt.scatter(0,0,color='k',marker='x',s=10)
for i in range(len(xg)):
    for j in range(len(yg)):
        plt.text(xg[i]+.02,yg[j]+.02,r'$G_{%d,%d}$'%(I[i],J[j]),fontsize=14,
            color='r')

# grid lines
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
for x in xg:
    plt.axvline(x+.5*delta,color='r',lw=1,ls=':')
for y in yg:
    plt.axhline(y+.5*delta,color='r',lw=1,ls=':')

# Gaussian contours
for rr in r:
    ax.add_artist(plt.Circle((0,0),rr,fc='none',ec='k',lw=.5))

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
plt.savefig('image/2D/demo.pdf')
plt.close()
#'''


#''' success rate vs parameters

# vs delta
rate = 1-np.sum(np.isnan(AA),axis=0)/N # rate of success

plt.figure(figsize=(6,4))
plt.ylim(-.05,1.05)
plt.plot(Delta,rate,lw=1.5,color='k')
plt.grid()
plt.xlabel(r'$\Delta/FWHM_0$')
plt.ylabel('Success rate')
plt.tight_layout()
plt.savefig('image/2D/success_vs_delta.pdf')
plt.close()

# vs phi
rate = 1-np.sum(np.isnan(AA),axis=1)/len(Phi_x) # rate of success

plt.figure(figsize=(6,4))
plt.ylim(-.05,1.05)
plt.scatter(Phi_x,rate,lw=1.5,color='k')
plt.grid()
plt.xlabel(r'$\phi_x$')
plt.ylabel('Success rate')
plt.tight_layout()
plt.savefig('image/2D/success_vs_phi_x.pdf')
plt.close()

plt.figure(figsize=(6,4))
plt.ylim(-.05,1.05)
plt.scatter(Phi_y,rate,lw=1.5,color='k')
plt.grid()
plt.xlabel(r'$\phi_y$')
plt.ylabel('Success rate')
plt.tight_layout()
plt.savefig('image/2D/success_vs_phi_y.pdf')
plt.close()
#'''


#''' fitting parameters vs delta

# parameters
s = 2
coef = .5 # coefficient for adjusting A&FWHM

DELTA = np.array([Delta]*N)

for prop in Para:
    # if not prop=='volume':
    #     continue

    data = Para[prop]['data']
    ylabel = Para[prop]['ylabel']

    # xxx vs delta
    plt.figure(figsize=(5,4))
    plt.scatter(DELTA,data,s=s,color='k')

    if prop=='A':
        xx = np.linspace(Delta[0],Delta[-1],100)
        yy = 1/(1+coef*xx**2)
        plt.plot(xx,yy,color='r',ls='--',
            label=r'$\frac{A}{A_0}=\frac{FWHM_0^2}{FWHM_0^2+%s\Delta^2}$'%coef)
        plt.legend(fontsize=18)

    if prop=='fwhm':
        xx = np.linspace(Delta[0],Delta[-1],100)
        yy = (coef*xx**2 + 1**2)**.5
        plt.plot(xx,yy,color='r',ls='--',
            label=r'$FWHM^2=FWHM_0^2+%s\Delta^2$'%coef)
        plt.legend()

    plt.grid()
    plt.xlabel(r'$\Delta/FWHM_0$')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig('image/2D/%s.pdf'%prop)
    plt.close()
#'''


    
