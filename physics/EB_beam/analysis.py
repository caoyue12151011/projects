'''
To illustrate the vibration of a cantilevered Euler-Bernoulli beam. See 
knowledge/EB_beam.md for more information.
'''

import os
import shutil
import num2tex
import imageio
import numpy as np
from numpy import sin, cos, sinh, cosh
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# change default matplotlib fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['font.size'] = 14


def mode_const(n):
    # n is the No. of the mode. Returns the constants in the mode solution.

    beta = None # normed spatial frequency
    if n==1:
        beta = .596864*np.pi 
    elif n==2:
        beta = 1.49418*np.pi 
    elif n==3:
        beta = 2.50025*np.pi 
    elif n==4:
        beta = 3.49999*np.pi 
    else:
        beta = (n-.5)*np.pi 

    c = cos(beta)
    s=  sin(beta)
    ch = cosh(beta)
    sh = sinh(beta)

    phi = -(ch+c)/(sh+s)
    C = -2*(ch*s-sh*c)/beta**4/(sh+s)

    return [beta, phi, C]



def mode(xi,beta,phi,C,normed=False):
    y = cosh(beta*xi)-cos(beta*xi)+phi*(sinh(beta*xi)-sin(beta*xi))
    if not normed:
        y *= C

    return y



def solver(n_max,np_xi,np_t,fp_t):
    '''
    To solve the beam oscillation problem. 

    Inputs
    ------
    n_max: highest number of the modes 
    np_xi: sampling number of the shortest spatial period 
    np_t: sampling number of the shortest temporal period 
    fp_t: temporal sampling range as a fraction of the longest period

    Returns
    -------
    Xi: 1D array 
    T: 1D array 
    Y0: len(Xi) array, the exact initial shape 
    Y_n: n*len(Xi) array, normed modes
    Y: n*len(Xi) array, modes of the beam 
    Y_c: n*len(Xi) array, cumulative modes
    YT_n: n*len(T)*len(Xi), oscillating normed modes
    YT: n*len(T)*len(Xi), oscillating modes
    YT_c: n*len(T)*len(Xi), cumulative oscillating modes
    '''

    N = np.arange(1,n_max+1)

    # constants     
    beta, phi, C = np.transpose([mode_const(n) for n in N])

    # determine the shapes of Xi, T 
    n_xi = int(beta[-1]*np_xi/(2*np.pi))
    n_t = int(beta[-1]**2*np_t/(2*np.pi))
    Xi = np.linspace(0,1,n_xi)
    T = np.linspace(0,fp_t*2*np.pi/beta[0]**2,n_t)
    print('len(Xi)=%d, len(T)=%d.'%(n_xi,n_t))

    # the initial shape
    Y0 = (Xi**3-3*Xi**2)/6

    # the modes
    beta1 = beta[...,np.newaxis]
    phi1 = phi[...,np.newaxis]
    C1 = C[...,np.newaxis]
    Y_n = mode(Xi,beta1,phi1,C1,normed=True)
    Y = mode(Xi,beta1,phi1,C1,normed=False)
    Y_c = np.cumsum(Y,axis=0)

    # oscillating modes
    F = cos(beta1**2*T)
    YT_n = np.expand_dims(Y_n,1)*np.expand_dims(F,-1)
    YT = np.expand_dims(Y,1)*np.expand_dims(F,-1)
    YT_c = np.cumsum(YT,axis=0)

    return Xi, T, Y0, Y_n, Y, Y_c, YT_n, YT, YT_c


# demo =========================================================================

# global parameters
Color = ['r','orangered','orange','y','green','c','blue','purple','k']


''' Constants vs n

# parameters 
n_max = 9

N = np.arange(1,n_max+1)
beta, phi, C = np.transpose([mode_const(n) for n in N])

# beta vs n
plt.figure()
plt.scatter(N,beta,color='k',s=15)
plt.plot(N,(N-.5)*np.pi,color='k',linewidth=1,
    label=r'$\beta=\pi\left(n-\frac{1}{2}\right)$')
plt.plot(1,0)
plt.legend()
plt.grid()
plt.xlabel(r'$n$')
plt.ylabel(r'$\beta_n$')
plt.tight_layout()
plt.savefig('image/beta_vs_n.png')
plt.close()


# C vs n
plt.figure()
plt.yscale('log')
for n, c in zip(N,C):
    cl = 'r'
    if c<0:
        cl = 'b'
    plt.scatter(n,np.abs(c),color=cl,s=15)
plt.scatter(1,0,s=15,c='r',label='Positive')
plt.scatter(1,0,s=15,c='b',label='Negative')

Nx = np.linspace(N.min(),N.max(),100)
plt.plot(Nx,2/(np.pi*(Nx-.5))**4,color='k',linewidth=1,
    label=r'$|C|=\frac{2}{\pi^4}\left(n-\frac{1}{2}\right)^{-4}$')
plt.legend()
plt.grid()
plt.xlabel(r'$n$')
plt.ylabel(r'$|C_n|$')
plt.tight_layout()
plt.savefig('image/C_vs_n.png')
plt.close()
#'''


''' the modes 

# parameters
n_max = 5 
np_xi = 100
np_t = 50 
fp_t = .5

# solve
Xi,T,Y0,Y_n,Y,Y_c,YT_n,YT,YT_c = solver(n_max,np_xi,np_t,fp_t)

# static normed modes
plt.figure(figsize=(12,5))
for i in range(n_max):
    plt.plot(Xi,Y_n[i],c=Color[i],label=r'$n=%d$'%(i+1))
plt.legend()
plt.grid()
plt.xlabel('$\\xi=x/l$')
plt.ylabel('Normed '+r'$w$')
plt.tight_layout()
plt.savefig('image/modes.png')
plt.close()

# oscillating normed modes .....................................................

# recreate directory 
if os.path.isdir('image/tmp/'):
    shutil.rmtree('image/tmp/') 
os.mkdir('image/tmp/')


Range = range(0,300,2)

for i in Range:
    # if not i==0:
    #     continue

    plt.figure(figsize=(10,4.5))
    plt.xlim(-.05,1.05)
    plt.ylim(-2.2,2.2)
    for j in range(n_max):
        plt.plot(Xi,YT_n[j,i],c=Color[j],label=r'$n=%d$'%(j+1))
    plt.text(.14,1.5,r'$\hat{t}=t\sqrt{\frac{EI}{\mu}}l^{-2}=%.3f$'%T[i],
        fontsize=18)
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('$\\xi=x/l$')
    plt.ylabel('Normed '+r'$Y$')
    plt.tight_layout()
    plt.savefig('image/tmp/%d.png'%i)
    plt.close()

# make gif
Name_f = ['image/tmp/%s.png'%i for i in Range]
with imageio.get_writer('image/oscil_modes.gif',mode='I',fps=30) as writer:
    for filename in Name_f:
        image = imageio.imread(filename)
        writer.append_data(image)
shutil.rmtree('image/tmp/') 
#'''


''' oscillating beam

# parameters
n_max = 4
np_xi = 100
np_t = 50 
fp_t = 4

# solve
Xi,T,Y0,Y_n,Y,Y_c,YT_n,YT,YT_c = solver(n_max,np_xi,np_t,fp_t)


# recreate directory 
if os.path.isdir('image/tmp/'):
    shutil.rmtree('image/tmp/') 
os.mkdir('image/tmp/')


Range = range(0,int(len(T)/4),1)
ymin = -1.05*YT_c.max()
ymax = 1.05*YT_c.max()

for i in Range:
    # if not i==0:
    #     continue

    plt.figure(figsize=(10,4.5))
    plt.xlim(-.05,1.05)
    plt.ylim(ymin,ymax)
    plt.plot(Xi,YT_c[0,i],c=Color[0],label=r'$n=1$')
    plt.plot(Xi,YT_c[-1,i],c=Color[-1],label=r'$n\leq%d$'%n_max)
    plt.text(.2,.2,r'$\hat{t}=t\sqrt{\frac{EI}{\mu}}l^{-2}=%.3f$'%T[i],
        fontsize=18)
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel('$\\xi=x/l$')
    plt.ylabel(r'$\hat{y}=y\frac{EI}{Fl^3}$')
    plt.tight_layout()
    plt.savefig('image/tmp/%d.png'%i)
    plt.close()

# make gif
Name_f = ['image/tmp/%s.png'%i for i in Range]
with imageio.get_writer('image/oscil_beam.gif',mode='I',fps=30) as writer:
    for filename in Name_f:
        image = imageio.imread(filename)
        writer.append_data(image)
shutil.rmtree('image/tmp/') 
#'''



