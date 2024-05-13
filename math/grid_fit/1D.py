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


def gaussian(x,A,mu,fwhm):
    return A*np.exp(-4*np.log(2)*((x-mu)/fwhm)**2)


def gridding(dx,xi,x_lim):
    # to grid function gaussian(x,1,0,1)
    
    # i range 
    i_min = int(np.floor((-x_lim-xi)/dx))
    i_max = int(np.ceil((x_lim-xi)/dx))
    print(i_min,i_max)
    
    # grid points 
    xg = xi + dx*np.arange(i_min,i_max+1)
    
    # gridded function
    a = (np.log(2))**.5
    G = np.pi**.5/(4*a*dx) * (erf(a*(2*xg+dx)) - erf(a*(2*xg-dx)))

    return xg, G


# gridding & fitting -----------------------------------------------------------

# parameters
x_lim = 5 # range of x [-x_lim,x_lim] for the gridding&fitting
Dx = np.linspace(.05,2,50) # channel width
Phi = np.linspace(-.5,.5,60,endpoint=False)

Xi = Phi[:,np.newaxis]*Dx # shape (xi, dx)

AA = np.full(Xi.shape,np.nan)
Mu = AA.copy()
Fwhm = AA.copy()

for i in range(len(Xi)):
#     if not i==0:
#         continue
    for j in range(len(Dx)):
#         if not j==0:
#             continue 
        xg, G = gridding(Dx[j],Xi[i,j],x_lim)
            
        try:
            AA[i,j],Mu[i,j],Fwhm[i,j] = curve_fit(gaussian,xg,G,[1,0,1])[0]
        except:
            None

# remove bad fittings
msk = Fwhm<.8 
AA[msk] = np.nan 
Mu[msk] = np.nan 
Fwhm[msk] = np.nan 

Area = AA*Fwhm 


# demo -------------------------------------------------------------------------

# mappables for drawing colormaps
cmap_ph = cm.ScalarMappable(cmap='viridis',norm=colors.Normalize(-.5,.5))
cmap_dx = cm.ScalarMappable(cmap='coolwarm',norm=colors.Normalize(np.nanmin(Dx),
    np.nanmax(Dx)))

# color arrays
C_ph = matplotlib.cm.get_cmap('viridis')(Phi+.5) 
C_dx = matplotlib.cm.get_cmap('coolwarm')((Dx-np.nanmin(Dx))/
    (np.nanmax(Dx)-np.nanmin(Dx))) 

# the data
Para = {
    'A': {
        'data': AA,
        'ylabel': r'$A/A_0$',
    },
    'mu': {
        'data': Mu,
        'ylabel': r'$\mu/FWHM_0$',
    },
    'fwhm': {
        'data': Fwhm,
        'ylabel': r'$FWHM/FWHM_0$',
    },
    'area': {
        'data': Area,
        'ylabel': r'$A\cdot FWHM/(A_0\cdot FWHM_0)$',
    },
}



''' diagram for demo 

# parameters 
dx = .8 # [FWHM0]
phi = -.4
x_lim = 2.5

xi = phi*dx
x = np.linspace(-x_lim,x_lim,200)
S0 = gaussian(x,1,0,1)
xg, G = gridding(dx,xi,x_lim)

plt.figure(figsize=(8,4))
plt.plot(x,S0,color='k',lw=2)
plt.step(xg,G,where='mid',color='r')
plt.scatter(xg,G,marker='x',fc='r',ec='r',s=30)
plt.axvline(0,color='k',ls='--')
plt.tight_layout()
plt.savefig('image/1D/demo.pdf')
plt.close()
#'''


#''' successful fittings 
plt.figure(figsize=(6,4))
plt.imshow(~np.isnan(AA),extent=[np.nanmin(Dx),np.nanmax(Dx),np.nanmin(Phi),
    np.nanmax(Phi)],aspect='auto')

plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot(0,-1,lw=7,color='yellow',label='Successful fittings')
plt.plot(0,-1,lw=7,color='purple',label='Failed fittings')
plt.legend(loc='upper left')

plt.grid()
plt.xlabel(r'$\Delta/FWHM_0$')
plt.ylabel(r'$\xi/\Delta$')
plt.tight_layout()
plt.savefig('image/1D/success_fit.pdf')
plt.close()
#'''


#''' fitting parameters vs dx/phi

# parameters
s = 3
coef = .5

for prop in Para:
    # if not prop=='A':
    #     continue

    data = Para[prop]['data']
    ylabel = Para[prop]['ylabel']

    # xxx vs dx
    plt.figure(figsize=(10,4))
    plt.subplot(121)
        
    for j in range(len(Dx)):
        plt.scatter([Dx[j]]*len(Phi),data[:,j],s=s,color=C_ph)

    if prop=='A':
        xx = np.linspace(Dx[0],Dx[-1],100)
        yy = 1/(1+coef*xx**2)**.5
        plt.plot(xx,yy,color='k',ls='--',
            label=r'$\frac{A}{A_0}=\frac{FWHM_0}{\sqrt{FWHM_0^2+%s\Delta^2}}$'
                %coef)
        plt.legend(fontsize=18)

    if prop=='fwhm':
        xx = np.linspace(Dx[0],Dx[-1],100)
        yy = (coef*xx**2 + 1**2)**.5
        plt.plot(xx,yy,color='k',ls='--',
            label=r'$FWHM^2=FWHM_0^2+%s\Delta^2$'%coef)
        plt.legend()

    plt.colorbar(cmap_ph,label=r'$\xi/\Delta$')
    plt.grid()
    plt.xlabel(r'$\Delta/FWHM_0$')
    plt.ylabel(ylabel)
    
    # xxx vs Phi
    plt.subplot(122)
    for i in range(len(Phi)):
        plt.scatter([Phi[i]]*len(Dx),data[i],s=s,color=C_dx)
    plt.colorbar(cmap_dx,label=r'$\Delta/FWHM_0$')
    plt.grid()
    plt.xlabel(r'$\xi/\Delta$')
    plt.tight_layout()
    plt.savefig('image/1D/%s.pdf'%prop)
    plt.close()
#'''


    
