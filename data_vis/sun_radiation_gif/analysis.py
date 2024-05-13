# To animate photons emitting from a star in real time.

import os
import num2tex
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13



# parameters
r = 696340 # [km], radius of the star, 696340 for the Sun
c = 2.99792e5 # [km/s] speed of light
n = 5 # number of photons emitted per second
t = 10 # [s], duration of the animation 
fps = 30

dt = 1/fps
l_tail = .1*r # tail length of the photon
l = 4*r # size of the canvas 
n_im = int(t*fps) # number of images
n_im_per_photon = int(round(fps/n))


Xp, Yp, Theta = np.array([]), np.array([]), np.array([]) # photon positions 
Xl, Yl = np.array([]), np.array([]) # positions of the tail ends

for i in range(n_im):
    # if not i==0:
    #     continue

    plt.figure(figsize=(5,5))
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    ax = plt.gca()
    ax.set_facecolor('#1e045b')
    plt.xlim(-l/2,l/2)
    plt.ylim(-l/2,l/2)

    # the Sun
    ax.add_artist(Circle((0,0),r,fc='#f96300',ec='none',zorder=10)) 
    ax.add_artist(Circle((0,0),.97*r,fc='#faa53b',ec='none',zorder=11)) 
    ax.add_artist(Circle((0,0),.9*r,fc='#fddeb9',ec='none',zorder=12)) 

    # draw photons 
    plt.scatter(Xp,Yp,s=6,fc='yellow',ec='none',zorder=5)
    plt.plot([Xl,Xp],[Yl,Yp],color='w',lw=.5,zorder=5)

    # update photon positions
    Xp += c*dt*np.cos(Theta)
    Yp += c*dt*np.sin(Theta)
    Xl += c*dt*np.cos(Theta)
    Yl += c*dt*np.sin(Theta)

    # generate new photons
    if i%n_im_per_photon==0:
        theta = np.random.rand()*2*np.pi 
        xp = r*np.cos(theta)
        yp = r*np.sin(theta)
        xl = (r-l_tail)*np.cos(theta)
        yl = (r-l_tail)*np.sin(theta)

        Theta = np.concatenate((Theta,np.array([theta])))
        Xp = np.concatenate((Xp,np.array([xp])))
        Yp = np.concatenate((Yp,np.array([yp])))
        Xl = np.concatenate((Xl,np.array([xl])))
        Yl = np.concatenate((Yl,np.array([yl])))

    plt.savefig('image/%d.png'%i)
    plt.close()


# make gif
Name_f = ['image/%s.png'%i for i in range(n_im)]
with imageio.get_writer('sun.gif',mode='I',fps=30) as writer:
    for filename in Name_f:
        image = imageio.imread(filename)
        writer.append_data(image)














