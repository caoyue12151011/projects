'''
The problem: in what phase does Venus reach its brightest observed from the 
Earth? Assuming that all the planetary orbits are circles and that the distance
between the two planets is much larger than the planet sizes at all times. 
Consider only the intrinsic brightness of the planet but ignore the influence 
of the daytime sky and the Sun.
'''

import sys 
import numpy as np 
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Ellipse, Rectangle, Arc, Polygon
from collections import OrderedDict
from importlib import reload

sys.path.append('../module')
import drawings
reload(drawings)


# change default matplotlib fonts
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"



def derivation(a,a0,theta):
    '''
    To derive the physical quantities given a,a0,theta_d.

    Inputs
    ------
    a: orbital semi-major of the target planet (TP)
    a0: orbital semi-major of the observer's planet (OP)
    theta: [rad], 0<=theta<2pi, the OP-Sun-TP angle 

    Outputs
    -------
    d: the OP-TP distance
    alpha: [rad], -pi<=alpha<=pi, the OP-TP-Sun angle, the "phase" angle,   
        0=full, pi/2=east half (bright), pi=crescent, -pi/2=west half
    phi:[rad], -pi<=phi<=pi, the TP-OP-Sun angle, or the TP-Sun angular distance 
    br: relative brightness, normalized such that br(a,a0,theta_br)=1
    theta_br: [rad], the theta on which the TP reaches its brightest. 
    d_br: distance when theta=theta_br
        There are two theta's (theta and 2pi-theta) that meet the condition, 
        but only the theta in [0,pi] is returned
    alpha_br: alpha when theta=theta_br
    phi_br: phi when theta=theta_br
    '''

    r_a = a/a0
    d = (a**2+a0**2-2*a*a0*np.cos(theta))**.5 
    alpha = np.arccos((a**2+d**2-a0**2)/(2*a*d))*np.sign(np.pi-theta)
    phi = np.pi-alpha-theta
    br0 = ((a+d)**2-a0**2)/(4*a*d**3) 

    theta_br = np.nansum([(r_a<=.25)*np.pi,(1<=r_a)*0.,
        (.25<r_a)*(r_a<1)*np.arccos(2*(r_a**2+3)**.5-2*r_a-1/r_a)],axis=0)
    d_br = (a**2+a0**2-2*a*a0*np.cos(theta_br))**.5 
    br_br = ((a+d_br)**2-a0**2)/(4*a*d_br**3) 
    br = br0/br_br # normalization
    alpha_br = (np.arccos((a**2+d_br**2-a0**2)/(2*a*d_br))*
        np.sign(np.pi-theta_br))
    phi_br = np.pi-alpha_br-theta_br

    return d,alpha,phi,br,theta_br,d_br,alpha_br,phi_br


# Demonstration ================================================================

# global parameters 
Planets = OrderedDict({
    'Mercury': {
        'a': 0.387,
        'color': 'gray',
    },

    'Venus': {
        'a': 0.723,
        'color': 'gold',
    },

    'Mars': {
        'a': 1.52,
        'color': 'orangered',
    },

    'Jupiter': {
        'a': 5.20,
        'color': 'brown',
    },

    'Saturn': {
        'a': 9.54,
        'color': 'y',
    },

    'Uranus': {
        'a': 19.2,
        'color': 'skyblue',
    },

    'Neptune': {
        'a': 30.1,
        'color': 'blue',
    },
})


''' theta vs br/alpha/phi -----------------------------------------------------

# parameters
a0 = 1 
theta = np.linspace(0,2*np.pi,500)

theta_d = theta*180/np.pi


# derivation of the physical quantities of the planets
for name in Planets:
    a = Planets[name]['a']

    d,alpha,phi,br,theta_br,d_br,alpha_br,phi_br = derivation(a,a0,theta)
    alpha_d = alpha*180/np.pi
    phi_d = phi*180/np.pi
    theta_br_d = theta_br*180/np.pi
    alpha_br_d = alpha_br*180/np.pi
    phi_br_d = phi_br*180/np.pi

    Planets[name]['parameters'] = (d,alpha,phi,br,theta_br,d_br,alpha_br,phi_br,
        alpha_d,phi_d,theta_br_d,alpha_br_d,phi_br_d)


plt.figure(figsize=(7,9))

# theta vs Rel. brightness .....................................................

plt.subplot(313)
for name in Planets:
    tmp = Planets[name]
    a = tmp['a']
    color = tmp['color']
    (d,alpha,phi,br,theta_br,d_br,alpha_br,phi_br,alpha_d,phi_d,theta_br_d,
        alpha_br_d,phi_br_d) = tmp['parameters']

    plt.plot(theta_d,br,color=color)
ax = plt.gca()
ax.set_xticks(np.arange(0,361,45))
ax.set_xticklabels(np.arange(0,361,45).astype(str))
plt.grid()
plt.xlabel('Earth-Sun-planet angle '+r'$\theta\ (^\circ)$')
plt.ylabel('Rel. brightness')

# theta vs phage angle .........................................................

plt.subplot(312)
for name in Planets:
    tmp = Planets[name]
    a = tmp['a']
    color = tmp['color']
    (d,alpha,phi,br,theta_br,d_br,alpha_br,phi_br,alpha_d,phi_d,theta_br_d,
        alpha_br_d,phi_br_d) = tmp['parameters']

    plt.plot(theta_d,alpha_d,color=color)
plt.grid()
plt.ylabel('Phase angle '+r'$\alpha\ (^\circ)$')
ax = plt.gca()
ax.set_xticks(np.arange(0,361,45))
ax.set_xticklabels(np.arange(0,361,45).astype(str))
ax.set_yticks(np.arange(-180,181,45))
ax.set_yticklabels(np.arange(-180,181,45).astype(str))

# y_to_x scale conversion factor for further use 
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
bbox_w = ax.bbox.width 
bbox_h = ax.bbox.height
ratio = (ymax-ymin)/bbox_h*bbox_w/(xmax-xmin)

# phage angle indicators .......................................................

for alpha_d in np.arange(-180,181,45):
    drawings.moon_phase(x=180,y=alpha_d,alpha_d=alpha_d,dx=10,c_br='w',c_dk='k',
        lw=.2,zorder=40,ax=ax)
plt.text(180-12,180-5,'E',fontsize=8)
plt.text(180+8,180-5,'W',fontsize=8)

# theta vs phi .................................................................

plt.subplot(311)
for name in Planets:
    tmp = Planets[name]
    a = tmp['a']
    color = tmp['color']
    (d,alpha,phi,br,theta_br,d_br,alpha_br,phi_br,alpha_d,phi_d,theta_br_d,
        alpha_br_d,phi_br_d) = tmp['parameters']

    plt.plot(theta_d,phi_d,color=color,label='%s (%s AU)'%(name,a))
    plt.scatter(theta_br_d,phi_br_d,marker='x',color=color)
    plt.scatter(360-theta_br_d,-phi_br_d,marker='x',color=color)

plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.scatter(1e3,1e3,marker='x',color='k',label='Brightest')

plt.legend(ncol=2,fontsize=8)
plt.grid()
ax = plt.gca()
ax.set_xticks(np.arange(0,361,45))
ax.set_xticklabels(np.arange(0,361,45).astype(str))
ax.set_yticks(np.arange(-180,181,45))
ax.set_yticklabels(np.arange(-180,181,45).astype(str))
plt.ylabel('Planet-Sun angular distance '+r'$\phi\ (^\circ)$')

# y_to_x scale conversion factor for further use 
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
bbox_w = ax.bbox.width 
bbox_h = ax.bbox.height
ratio = (ymax-ymin)/bbox_h*bbox_w/(xmax-xmin)

# Mercury & Venus at their brightest ...........................................

dx = 45
for name in ['Mercury','Venus']:
    tmp = Planets[name]
    a = tmp['a']
    color = tmp['color']
    (d,alpha,phi,br,theta_br,d_br,alpha_br,phi_br,alpha_d,phi_d,theta_br_d,
        alpha_br_d,phi_br_d) = tmp['parameters']

    x, y = None, None 
    if name=='Mercury':
        x, y = 30, -110
    elif name=='Venus':
        x, y = 100, -110

    plt.text(x-.5*dx,y+1.5*dx,'Brightest '+name,fontsize=8,color='k',
        weight='bold')
    plt.text(x,y,(r'$\theta=%.1f^\circ$'%theta_br_d+'\n'+
        r'$\alpha=%.1f^\circ$'%alpha_br_d+'\n'+
        r'$\phi=%.1f^\circ$'%phi_br_d),fontsize=8,zorder=20)
    drawings.moon_phase(x=x,y=y,alpha_d=alpha_br_d,dx=dx,c_br='w',c_dk=color,
        lw=.5,zorder=5,ax=ax)

# solar system demo ............................................................

# parameters 
x_sun, y_sun = 175, 25
theta_demo = np.pi/6 
r_e = 50 # orbital radius of the earth in x-axis units 
r_p = 35 # orbital radius of the planet in x-axis units 
lw = .5 # linewidth

# derived parameters 
theta_demo_d = theta_demo*180/np.pi
x_p = x_sun - r_p*np.sin(theta_demo) # position of the planet 
y_p = y_sun + ratio*r_p*np.cos(theta_demo)
x_e = x_sun 
y_e = y_sun + ratio*r_e 

# The OP-TP-Sun triangle
plt.plot([x_sun,x_e,x_p,x_sun],[y_sun,y_e,y_p,y_sun],c='k',linewidth=lw)
plt.scatter([x_sun,x_e,x_p],[y_sun,y_e,y_p],color='k',s=4)
plt.text(x_sun+3,y_sun,'Sun',fontsize=8)
plt.text(x_e+3,y_e,'Earth',fontsize=8)
plt.text(x_p-20,y_p,'Planet',fontsize=8)

# orbital arc (lots of hard coding)
ax.add_artist(Arc((x_sun,y_sun),2*r_e,ratio*2*r_e,0,80,100,edgecolor='k',
    linewidth=lw))
ax.add_artist(Arc((x_sun,y_sun),2*r_p,ratio*2*r_p,0,theta_demo_d+62,
    theta_demo_d+90,edgecolor='k',linewidth=lw))

# angle marker 
plt.text(x_sun-4,y_sun+20,r'$\theta$',fontsize=8)
plt.text(x_e-5,y_e-25,r'$\phi$',fontsize=8)
plt.text(x_p+3,y_p-5,r'$\alpha$',fontsize=8)

plt.tight_layout()
plt.savefig('image/theta_vs_quantities.pdf')
plt.close()
#'''


#''' the demo plot -------------------------------------------------------------

# parameters. theta and r_a are the only two variables that control the whole 
# plot (while a0 is regarded as a constant). The data sets that need to be 
# plotted are Data(theta,r_a), Data(theta,r_a_array), and Data(theta_array,r_a)
a0 = 1
theta_d = 0 # theta_d value for initial plot  
r_a = .7 # a/a0 value for initial plot  
R_a = np.linspace(.05,2,400) # array of r_a
Theta = np.linspace(0,2*np.pi,500) # array of theta


# derived parameters
a = a0*r_a
theta = theta_d*np.pi/180
A = a0*R_a
Theta_d = Theta*180/np.pi
r_a_min = .1 # slider limits 
r_a_max = 1.5


# derived physical quantities for initial plot
d,alpha,phi,br,theta_br,d_br,alpha_br,phi_br = derivation(a,a0,theta)
D1,Alpha1,Phi1,Br1,Theta_br1,D_br1,Alpha_br1,Phi_br1 = derivation(A,a0,theta)
D2,Alpha2,Phi2,Br2,Theta_br2,D_br2,Alpha_br2,Phi_br2 = derivation(a,a0,Theta)

alpha_d = alpha*180/np.pi
theta_br_d = theta_br*180/np.pi
alpha_br_d = alpha_br*180/np.pi
Theta_br_d1 = Theta_br1*180/np.pi


# figure parameters
fig_x, fig_y = 8, 6 # [in], figure size
h = .15 # proportion of the space occupied by sliders
lf_s, rt_s = .7, .7 # [in], margin sizes of the slider axes
lf13, rt13, up13, bt13 = .7, 0, .2, .4 # [in], margin sizes of axes 1 & 3
mg24 = .05 # [in], margin sizes of axes 2 & 4


# derived figure parameters
fx_s = lf_s/fig_x # left margin of the slider in [0,1]
fy_theta = h/5
fy_a = h*3/5
fw_s = 1-(lf_s+rt_s)/fig_x  # width of the slider in [0,1]
fh_s = h/5 # height of the slider in [0,1] 

fx13 = lf13/fig_x # left margin of axes 1&3 in [0,1]
fy1 = (1+h)/2 + bt13/fig_y
fy3 = h + bt13/fig_y
fw13 = .5-(lf13+rt13)/fig_x 
fh13 = (1-h)/2 - (up13+bt13)/fig_y

fx24 = .5+mg24/fig_x 
fy2 = (1+h)/2 + mg24/fig_y
fy4 = h+mg24/fig_y 
fw24 = .5-2*mg24/fig_x 
fh24 = (1-h)/2 - 2*mg24/fig_y 


# create canvas
fig = plt.figure(figsize=(fig_x,fig_y))

# R_a vs Theta_br (initial plot) ...............................................

ax1 = plt.axes([fx13,fy1,fw13,fh13])
plt.plot(R_a,Theta_br_d1,color='k',linewidth=1)
dot1 = plt.scatter(r_a,theta_br_d,s=5,facecolor='r',edgecolor='none')
Dot1 = plt.scatter(r_a,theta_br_d,s=50,facecolor='r',edgecolor='none',alpha=.3)
plt.grid()
ax1.set_yticks(np.arange(0,181,45))
ax1.set_yticklabels(np.arange(0,181,45).astype(str))
plt.xlabel(r'$a/a_0$')
plt.ylabel(r'$\theta_{\rm brightest}\ (^\circ)$')

# theta vs br (initial plot) ...................................................

ax3 = plt.axes([fx13,fy3,fw13,fh13])
line, = plt.plot(Theta_d,Br2,color='k',linewidth=1)
dot3 = plt.scatter(theta_d,br,s=5,facecolor='r',edgecolor='none')
Dot3 = plt.scatter(theta_d,br,s=50,facecolor='r',edgecolor='none',alpha=.3)
plt.grid()
ax3.set_xticks(np.arange(0,361,45))
ax3.set_xticklabels(np.arange(0,361,45).astype(str))
plt.xlabel('Earth-Sun-planet angle '+r'$\theta\ (^\circ)$')
plt.ylabel('Rel. brightness')

# solar system diagram .........................................................

ax2 = plt.axes([fx24,fy2,fw24,fh24])
plt.axis('equal')
ax2.axis('off')

xmax = r_a_max*a0

plt.plot([-xmax,xmax,xmax,-xmax],[-xmax,-xmax,xmax,xmax],color='none')
cir_e = Circle((0,0),a0,fc='none',ec='green',lw=1) # Earth's orbital circle 
cir_p = Circle((0,0),a,fc='none',ec='r',lw=1) # planet's orbital circle 
ax2.add_artist(cir_e)
ax2.add_artist(cir_p)
plt.scatter(0,0,s=60,fc='k',ec='none') # Sun
plt.scatter(0,a0,s=30,fc='green',ec='none') # Earth 
dot_p = plt.scatter(-a*np.sin(theta),a*np.cos(theta),s=30,fc='r',ec='none') # TP
tri,=plt.plot([0,0,-a*np.sin(theta),0],[0,a0,a*np.cos(theta),0],color='gray',
    lw=1)
plt.text(.1,0,'Sun',fontsize=10)
plt.text(.1,a0,'Earth',fontsize=10,color='green')
text = plt.text(-a*np.sin(theta)+.1,a*np.cos(theta),'Planet',fontsize=10,c='r')
dot2 = plt.scatter([-a*np.sin(theta_br)+.1,a*np.sin(theta_br)],
    [a*np.cos(theta_br)]*2,marker='x',color='r',s=30)

# phase demo ...................................................................

ax4 = plt.axes([fx24,fy4,fw24,fh24])
plt.xlim(-1,1)
plt.ylim(-1,1)
ax4.axis('off')
drawings.moon_phase(0,0,alpha_d,abs(a-a0)/d,'w','gray',lw=1,zorder=1,ax=ax4)

# sliders ......................................................................

# theta slider 
ax_theta = plt.axes([fx_s,fy_theta,fw_s,fh_s])
theta_slider = Slider(ax=ax_theta,label=r'$\theta\ (^\circ)$',valmin=0,
    valmax=360,valinit=theta_d)


# r_a slider 
ax_r_a = plt.axes([fx_s,fy_a,fw_s,fh_s])
r_a_slider = Slider(ax=ax_r_a,label=r'$a/a_0$',valmin=r_a_min,valmax=r_a_max,
    valinit=r_a)


# The function to be called anytime a slider's value changes
def update(val,a0=a0,Theta=Theta_d,ax=ax4):
    Theta = Theta_d*np.pi/180 

    # new slider values
    r_a_new = r_a_slider.val 
    theta_d_new = theta_slider.val

    a_new = a0*r_a_new
    theta_new = theta_d_new*np.pi/180  
    xmax_new = max(a_new,a0)

    # d,alpha,phi,br,theta_br,d_br,alpha_br,phi_br
    # 0,1,    2,  3, 4,       5,   6,       7
    para = derivation(a_new,a0,theta_new) 
    d_new = para[0]
    alpha_new = para[1]
    br_new = para[3]
    theta_br_new = para[4] 

    alpha_d_new = alpha_new*180/np.pi 
    theta_br_d_new = theta_br_new*180/np.pi 

    Para = derivation(a_new,a0,Theta) 
    Br2_new = Para[3]

    # update plots in ax1
    dot1.set_offsets([r_a_new,theta_br_d_new])
    Dot1.set_offsets([r_a_new,theta_br_d_new])

    # update plots in ax2
    cir_p.set_radius(a_new)
    dot_p.set_offsets([-a_new*np.sin(theta_new),a_new*np.cos(theta_new)])
    tri.set_xdata([0,0,-a_new*np.sin(theta_new),0])
    tri.set_ydata([0,a0,a_new*np.cos(theta_new),0])
    text.set_position([-a_new*np.sin(theta_new)+.1,a_new*np.cos(theta_new)])
    dot2.set_offsets([[-a_new*np.sin(theta_br_new),a_new*np.cos(theta_br_new)],
        [a_new*np.sin(theta_br_new),a_new*np.cos(theta_br_new)]])

    # update plots in ax3
    dot3.set_offsets([theta_d_new,br_new])
    Dot3.set_offsets([theta_d_new,br_new])
    line.set_ydata(Br2_new)

    # update plots in ax4
    ax4.clear()
    ax4.set_xlim(-1,1)
    ax4.set_ylim(-1,1)
    ax4.axis('off')
    drawings.moon_phase(0,0,alpha_d_new,abs(a_new-a0)/d_new,'w','gray',lw=1,
        zorder=1,ax=ax4)

    fig.canvas.draw_idle()


# register the update function with each slider
theta_slider.on_changed(update)
r_a_slider.on_changed(update)

plt.savefig('image/demo_plot.pdf')
plt.show()
#'''



















