# To calculate the self-similar collapse solution of Shu (1977) 
# (https://ui.adsabs.harvard.edu/abs/1977ApJ...214..488S%2F/abstract).

import matplotlib
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def eq(y,x):
    # Eq. 11&12 in Shu (1977).

    v, a = y 
    dvdx = (a*(x-v)-2/x)*(x-v) / ((x-v)**2-1)
    dadx = (a-2/x*(x-v))*(x-v) / ((x-v)**2-1) * a

    return [dvdx,dadx]



# parameters
A = np.array([2.01,2.02,2.1,2.2,2.4,3,5,10,20])
x = np.linspace(.1,10,300)[::-1]

c_scale = np.log(A-2)
color = matplotlib.cm.get_cmap('coolwarm')((c_scale-c_scale[0])/
    (c_scale[-1]-c_scale[0]))


v, a = [], []
for i in range(len(A)):

    v0 = -(A[i]-2)/x[0]-(1-A[i]/6)*(A[i]-2)/x[0]**3 # Eq 19 in Shu (1977)
    a0 = A[i]/x[0]**2-A[i]*(A[i]-2)/(2*x[0]**4)

    res = odeint(eq,[v0,a0],x)
    v.append(res[:,0])
    a.append(res[:,1])



plt.figure(figsize=(8,4))

plt.subplot(121)
plt.xscale('log')
plt.yscale('log')
for i in range(len(A)):
    plt.plot(x,abs(v[i]),color=color[i],lw=1)
plt.grid()
plt.xlabel(r'$x=\frac{r}{at}$')
plt.ylabel(r'$|v|=|\frac{u}{a}|$')

plt.subplot(122)
plt.xscale('log')
plt.yscale('log')
for i in range(len(A)):
    plt.plot(x,a[i],color=color[i],lw=1)
    plt.text(1.05,i/len(A),r'$A=%s$'%A[i],color=color[i],fontsize=12,
        transform=plt.gca().transAxes,weight='bold')
plt.grid()
plt.xlabel(r'$x=\frac{r}{at}$')
plt.ylabel(r'$a=4\pi Gt^2\rho$')

plt.tight_layout()
plt.savefig('Shu_1977_solutions.pdf')
plt.close()



