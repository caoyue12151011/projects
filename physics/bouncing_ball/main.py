# TODO: animation, sliding ball, billiard problem
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import jit
from importlib import reload
from matplotlib.patches import Circle, Arc
from matplotlib.widgets import Slider
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

sys.path.append('/Users/yuecao/Documents/coding/module')
import math_tk
reload(math_tk)

# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13

def iterator(xi, yi, vxi, vyi, tol):
    """
    To iterate the motion of next collision given the initial condition.

    Inputs
    ------
    xi, yi, vxi, vyi: scalar or np.ndarray of same shape
    tol: tolerance of imagenary part to treat the roots as reals

    Returns
    -------
    xi, yi, vxi, vyi, ti: parameters of the next collision
    """
    # coefficients of the equation
    a = .25 if not hasattr(xi,'__len__') else np.full(xi.shape, .25)
    b = -vyi
    c = vxi**2 + vyi**2 - yi
    d = 2*(xi*vxi + yi*vyi)

    # solve equation, sol = 1d or (3, shape(a)) array
    sol = np.array(math_tk.cubic_solver(a, b, c, d))

    # find the physical solution, i.e. the smallest non-negative root
    sol[np.abs(sol.imag) > tol] = np.nan 
    sol = np.sign(sol.real) * np.abs(sol)
    sol[sol < 0] = np.nan 
    ti = np.sort(sol, axis=0)[0]

    # parameters right before the collision
    xipp = xi + vxi*ti  # x_(i+1)
    yipp = yi + vyi*ti - .5*ti**2
    vyip = vyi - ti  # vyi_prime; vxip = vxi

    # fix the precision error: make collision point on the circle
    r2 = xipp**2 + yipp**2 
    xipp *= r2**-.5
    yipp *= r2**-.5

    # velocity after the collision
    c1 = 2*xipp**2 - 1
    c2 = 2*xipp*yipp
    vxipp = -c1*vxi - c2*vyip
    vyipp = -c2*vxi + c1*vyip

    # fix the precision error: make energy conserved 
    v2 = vxipp**2 + vyipp**2
    v2_assumed = vxi**2 + vyi**2 + 2*(yi-yipp)
    fct = (v2_assumed / v2)**.5
    vxipp *= fct
    vyipp *= fct

    return xipp, yipp, vxipp, vyipp, ti


def solver(theta0, theta_v0, v0, N_bounce, tol):
    """
    To calculate the motion of a bouncing ball in a circle.
    theta0, theta_v0, v0 can be scalar or np.ndarray of the same shape
    theta0: [-pi, pi]
    theta_v0: (-pi/2, pi/2)
    v0: [0, inf)
    """
    # derived initial parameters
    x0 = np.sin(theta0)
    y0 = -np.cos(theta0)
    vx0 = -v0 * np.sin(theta0 - theta_v0)
    vy0 = v0 * np.cos(theta0 - theta_v0)

    # array parameters
    if not hasattr(theta0, '__len__'):
        Xi = np.full(N_bounce+1, np.nan)
        Ti = np.full(N_bounce, np.nan)
    else:
        Xi = np.full([N_bounce+1]+list(theta0.shape), np.nan)
        Ti = np.full([N_bounce]+list(theta0.shape), np.nan)
    Yi = Xi.copy()
    Vxi = Xi.copy()
    Vyi = Xi.copy()
    
    # initial setup
    Xi[0] = x0
    Yi[0] = y0
    Vxi[0] = vx0
    Vyi[0] = vy0
    for i in range(N_bounce):
        # For scalars, time of iteration last axis (Xi[...,i]) is 13 µs,
        # while for fist axis it's 7 µs. For arrays no difference. 
        Xi[i+1],Yi[i+1],Vxi[i+1],Vyi[i+1],Ti[i]=iterator(
            Xi[i], Yi[i], Vxi[i], Vyi[i], tol)

    # derived array parameters
    Theta = np.arctan2(Xi, -Yi)
    Alpha = np.arctan2(Vyi, Vxi)  # angle between v and x-axis
    Theta_v = np.pi/2 + Theta - Alpha
    Theta_v = (Theta_v+np.pi/2)%np.pi - np.pi/2  # make in [-pi/2, pi/2]
    Vi = (Vxi**2 + Vyi**2)**.5

    return Theta, Theta_v, Vi, Xi, Yi, Vxi, Vyi, Ti


def calculator(t_abs, Xi, Yi, Vxi, Vyi, Ti):
    """
    Calculate parameters at absolute time t_abs. 

    Inputs
    ------
    t_abs: scalar or 1d array of shape (t_abs)
    Xi, Yi, Vxi, Vyi, Ti: 1d (t) or (t, shape) np.ndarray of the same shape

    Returns
    -------
    x, y, vx, vy: np.ndarray of shape (shape) or (t_abs, shape)
    """
    is_scalar = not hasattr(t_abs, '__len__')
    if is_scalar:
        t_abs = np.array([t_abs])  # scalar -> 1d array (t_abs)
    
    # (t_abs) -> (t_abs, t, shape)
    for i in range(Xi.ndim):
        t_abs = t_abs[..., None]

    Ti = np.concatenate(([np.zeros_like(Ti[0])], Ti))  # (t, shape)
    cumsum_ti = np.cumsum(Ti, axis=0)  # (t, shape)

    flag_large = t_abs >= cumsum_ti  # (t_abs, t, shape)
    ind = np.sum(flag_large, axis=1) - 1  # (t_abs, shape)
    sum_ti = np.take_along_axis(cumsum_ti[None], ind[:,None], axis=1)[:,0]  
        # (t_abs, shape)
    t = t_abs[:,0] - sum_ti  # (t_abs, shape)

    xi = np.take_along_axis(Xi[None], ind[:,None], axis=1)[:,0]
    yi = np.take_along_axis(Yi[None], ind[:,None], axis=1)[:,0]
    vxi = np.take_along_axis(Vxi[None], ind[:,None], axis=1)[:,0]
    vyi = np.take_along_axis(Vyi[None], ind[:,None], axis=1)[:,0]

    x = xi + vxi*t
    y = yi + vyi*t - .5*t**2
    vx = vxi
    vy = vyi - t
    if is_scalar:
        x, y, vx, vy = x[0], y[0], vx[0], vy[0]

    return x, y, vx, vy

# main ========================================================================
"""
Note: 
Physical parameters are dimensionless. 
x_hat = x/R, t_hat = t*sqrt(g/R), v_hat = v/sqrt(gR). In this way R=g=1.
""" 
# global parameters 
tol = 1e-10  # tolerance of imaginary part to regard as real

#'''
# parameters 
N_bounce = 100000  # total number of simulated bounces
theta0_d = 4.5  # [deg] 
theta_v0_d = -3.1  # [deg] 
v0 = 1.6

# derived parameters 
theta0 = theta0_d * np.pi/180 
x0 = np.sin(theta0)
y0 = -np.cos(theta0) 
theta_v0 = theta_v0_d * np.pi/180  
vx0 = -v0 * np.sin(theta0-theta_v0)
vy0 = v0 * np.cos(theta0-theta_v0)
case = f'C{theta0_d:.2f}_{theta_v0_d:.2f}_{v0:.3f}'

# solve the problem
print(f'Case {case}.')
t = time.time()
Theta,Theta_v,Vi,Xi,Yi,Vxi,Vyi,Ti = solver(theta0,theta_v0,v0,N_bounce,tol)
t = time.time() - t

# number of successful iterations 
print(f'Time: {1e6*t/N_bounce:.2f}µs * {N_bounce} = {t:.1f}s.')

# derived array parameters 
Theta_d = Theta * 180/np.pi
Theta_v_d = Theta_v * 180/np.pi

# make image folder
os.makedirs(f'image/{case}', exist_ok=True)
#'''
# demo ========================================================================
"""
Contents (Figures used the above simulation data are marked with *)
--------
model demo 
*animation
block animation
*trajectory viewer
*probability map
*phase diagram
*chaos check
chaos map
*histograms
interactive plot
"""

''' model demo 

# parameters 
theta0_d = -70  # deg
vx0 = .8
vy0 = 1
N_bounce = 3  # total number of simulated bounces
tol = 1e-10  # tolerance of imaginary part
M = 300  # number of sampling points of parabola
v_fct = .4  # factor of velocity for drawing

# derived parameters 
theta0 = theta0_d * np.pi/180  
x0 = np.sin(theta0)
y0 = -np.cos(theta0)
theta_v0 = np.arctan2(vx0, vy0)
theta_v0_d = theta_v0 * 180/np.pi
theta_v0_rel = theta0 + theta_v0
theta_v0_rel_d = theta_v0_rel * 180/np.pi

# solve the problem
Xi, Yi, Vxi, Vyi, Ti = solver(theta0, vx0, vy0, N_bounce, tol)

# trajectories
x = []
y = []
for i in range(N_bounce):
    xi, yi = Xi[i], Yi[i]
    vxi, vyi = Vxi[i], Vyi[i]
    ti = Ti[i]
    t = np.linspace(0, ti, M)
    xx = xi + vxi*t
    yy = yi + vyi*t - .5*t**2
    x.extend(list(xx))
    y.extend(list(yy))

# demo 
plt.figure(figsize=(6,6))
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
ax = plt.gca()
ax.add_patch(Circle((0,0), 1, ec='none', fc='whitesmoke', lw=1))
# initial position
plt.scatter(x0, y0, ec='k', fc='w', s=30, zorder=48)  
# position angle 
plt.plot([0,x0], [0,y0], color='k', ls='--', lw=1)
plt.text(-.1, -.1, r'$\theta_0$', fontsize=16)
ax.add_patch(Arc((0,0), .1, .1, theta1=270+theta0_d, theta2=270, color='k'))
# initial velocity
plt.arrow(x0, y0, v_fct*vx0, v_fct*vy0, color='r', lw=1, head_width=.03)
plt.text(x0+v_fct*vx0, y0+v_fct*vy0+.1, r'$(v_{x0},v_{y0})$', fontsize=16, 
         color='r')
# velocity angle 
plt.plot([x0,x0], [y0,y0+.5], color='k', ls='--', lw=1)
plt.text(x0, y0+.15, r'$\theta_{v0}$', fontsize=16)
ax.add_patch(Arc((x0,y0), .1, .1, theta1=90+theta0_d, theta2=90, color='k'))
# rel velocity angle 
plt.text(x0+.15, y0+.1, r'$\theta_{v0,\rm rel}$', fontsize=16)
# trajectories
plt.plot(x, y, color='k', lw=1)
# collision points
for i in range( N_bounce+1):
    text = rf'$A_{i}$' if i>0 else r'$A_0\ (x_0,y_0)$'
    plt.text(Xi[i], Yi[i]-.1, text, fontsize=16)
plt.grid()
plt.xlabel(r'$\hat{x}$')
plt.ylabel(r'$\hat{y}$')
plt.tight_layout()
plt.savefig(f'image/model_demo.pdf')
plt.close()
#'''

''' animation 
# parameters 
t_f = sum(Ti[:3])
fps = 30
duration = 5  # [sec]
len_v_demo = .35

t_video = np.linspace(0, duration, fps*duration)
t_simu = t_video/duration*t_f

x, y, vx, vy = calculator(t_simu, Xi, Yi, Vxi, Vyi, Ti)

fig = plt.figure(figsize=(8,8))
plt.axis('equal')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
ax = plt.gca()
ax.add_patch(Circle((0,0), 1, ec='grey', fc='none', lw=1))
plt.scatter(x0, y0, ec='gray', fc='w', s=30, zorder=48)  # initial position
dot = plt.scatter(x0, y0, ec='k', fc='w', s=30, zorder=49)
plt.arrow(x0, y0, vx0/v0*len_v_demo, vy0/v0*len_v_demo, color='r', lw=.8, 
          head_width=.02, zorder=47)  # initial velocity
plt.grid()
plt.xlabel(r'$\hat{x}$')
plt.ylabel(r'$\hat{y}$')
plt.tight_layout()

def make_frame(t):
    xx = np.interp(t, t_video, x)
    yy = np.interp(t, t_video, y)
    dot.set_offsets([xx, yy])
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration=duration)
animation.write_videofile(f'image/{case}/animation.mov', fps=fps, 
                          codec='mpeg4')
#'''

''' block animation 
# simulation parameters 
N_bounce = 500
theta0 = np.linspace(-43, -45, 50) *np.pi/180
theta_v0 = 10 *np.pi/180
v0 = np.linspace(1, 1.05, 50)

M = len(theta0) * len(v0)
s = 5e3/M

# simulation
theta0, v0 = np.meshgrid(theta0, v0)  # (v0, theta0)
theta_v0 = theta_v0 * np.ones_like(v0)
Theta,Theta_v,Vi,Xi,Yi,Vxi,Vyi,Ti = solver(theta0,theta_v0,v0,N_bounce,tol)

# time check 
t_f_max = np.sum(Ti, axis=0).min()
print(f'Simulation time of the video should be < {t_f_max:.1f} s.')

# demo parameters 
t_i = 0  # simulation starting time
t_f = 20  # simulation ending time
play_speed = 1
fps = 30

duration = (t_f-t_i)/play_speed  # [sec]

# demo
fig = plt.figure(figsize=(8,8))
plt.axis('equal')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
ax = plt.gca()
ax.add_patch(Circle((0,0), 1, ec='grey', fc='none', lw=1))
dot = plt.scatter(Xi[0], Yi[0], ec='none', fc='k', s=s, zorder=49)
plt.grid()
plt.xlabel(r'$\hat{x}$')
plt.ylabel(r'$\hat{y}$')
plt.tight_layout()

def make_frame(t_vid):
    t = t_vid*play_speed + t_i
    x, y, vx, vy = calculator(t, Xi, Yi, Vxi, Vyi, Ti)
    dot.set_offsets(np.transpose([x.flatten(), y.flatten()]))
    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration=duration)
animation.write_videofile(f'image/animation.mov', fps=fps, codec='mpeg4')
#'''

''' trajectory viewer
# parameters 
M = 100  # number of sampling for each trajectory
N_demo = 1000  # number of trajectories to be shown
len_v_demo = .35

# derived parameters
ind_demo = np.random.choice(N_bounce, size=N_demo, replace=False)
lw = 50/N_demo

# trajectories
xx, yy = [], []
for i in ind_demo:
    xi, yi = Xi[i], Yi[i]
    vxi, vyi = Vxi[i], Vyi[i]
    ti = Ti[i]
    t = np.linspace(0, ti, M)
    x = xi + vxi*t
    y = yi + vyi*t - .5*t**2
    xx.extend(list(x) + [np.nan])
    yy.extend(list(y) + [np.nan])

plt.figure(figsize=(8,8))
plt.axis('equal')
ax = plt.gca()
ax.add_patch(Circle((0,0), 1, ec='grey', fc='none', lw=1))
plt.scatter(x0, y0, ec='k', fc='w', s=30, zorder=48)  # initial position
plt.plot(xx, yy, color='k', lw=lw)
plt.arrow(x0, y0, vx0/v0*len_v_demo, vy0/v0*len_v_demo, color='r', lw=.8, 
          head_width=.02, zorder=47)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid()
plt.xlabel(r'$\hat{x}$')
plt.ylabel(r'$\hat{y}$')
plt.tight_layout()
plt.savefig(f'image/{case}/trajectory.pdf')
plt.close()
#'''

''' probability map
# parameters 
t_f = sum(Ti[:2])
M = 100  # number of sampling 

t_abs = np.random.uniform(0, t_f, M)
s = 5e2/M

x, y, vx, vy = np.full(M,np.nan), np.full(M,np.nan), np.full(M,np.nan), np.full(M,np.nan)
for i in range(M):
    x[i], y[i], vx[i], vy[i] = calculator(t_abs[i], Xi, Yi, Vxi, Vyi, Ti)

plt.figure(figsize=(8,8))
plt.axis('equal')
ax = plt.gca()
ax.add_patch(Circle((0,0), 1, ec='grey', fc='none', lw=1))
plt.scatter(x0, y0, ec='k', fc='w', s=30, zorder=48)  # initial position
plt.scatter(x, y, s=s, fc='k', ec='none')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid()
plt.xlabel(r'$\hat{x}$')
plt.ylabel(r'$\hat{y}$')
plt.tight_layout()
plt.savefig(f'image/{case}/probability.pdf')
plt.show()
#'''

''' phase diagram
plt.figure()
plt.scatter(Theta_d, Theta_v_d, s=3e4/N_bounce, fc='k', ec='none')
for i in range(10):
    plt.scatter(Theta_d[i], Theta_v_d[i], s=30, fc='w', ec='k')
    plt.text(Theta_d[i]+1, Theta_v_d[i], f'{i}', fontsize=14)
plt.grid()
plt.xlabel(r'$\theta\ (^\circ)$')
plt.ylabel(r'$\theta_v\ (^\circ)$')
plt.tight_layout()
plt.savefig(f'image/{case}/phase.pdf')
plt.close()
#'''

''' chaos check
# parameters 
dtheta0 = 0
dtheta_v0 = 1e-6

text = (r'$\Delta\theta_0=$' + f'{dtheta0:.2e}\n' + r'$\Delta\theta_{v0}=$' + 
        f'{dtheta_v0:.2e}')

res = solver(theta0+dtheta0, theta_v0+dtheta_v0, v0, N_bounce, tol)
Theta1, Theta_v1 = res[0], res[1]
Dist = ((Theta-Theta1)**2 + (Theta_v-Theta_v1)**2)**.5

plt.figure()
plt.plot(np.arange(N_bounce+1), Dist, color='k', lw=.2)
plt.text(.05, .85, text, fontsize=14, transform=plt.gca().transAxes)
plt.grid()
plt.xlabel('Number of iteration')
plt.ylabel('Distance in phase space')
plt.tight_layout()
plt.savefig(f'image/{case}/chaos_check.pdf')
plt.show()
#'''

''' chaos map
# parameters 
dtheta0 = 0
dtheta_v0 = 1e-6
M_theta = 200  # number of sampling
M_theta_v = 200  # M_theta*M_theta_v do not exceed ~4e6 due to memory limit
N_bounce = 30
v0 = 1

# derived parameters 
N_iter = 2 * M_theta * M_theta_v * N_bounce
text = (r'$\Delta\theta_0=$' + f'{dtheta0:.2e} ' + 
        r'$\Delta\theta_{v0}=$' + f'{dtheta_v0:.2e} ' + 
        r'$\hat v_0=$' + f'{v0}')

# meshgrid
theta0_d = np.linspace(-180, 180, M_theta)
theta_v0_d = np.linspace(-89, 89, M_theta_v)
theta0 = theta0_d * np.pi/180
theta_v0 = theta_v0_d * np.pi/180
theta0, theta_v0 = np.meshgrid(theta0, theta_v0)  # (theta_v, theta)
theta0 = np.array([theta0, theta0+dtheta0])  # (diff, theta_v, theta)
theta_v0 = np.array([theta_v0, theta_v0+dtheta_v0])

# solve the problem
t = time.time()
Theta, Theta_v, Vi, Xi, Yi, Vxi, Vyi, Ti = solver(
                                        theta0, theta_v0, v0, N_bounce, tol)
t = time.time() - t  
print(f'Time: {1e6*t/N_iter:.2f}µs * {N_iter} {Ti.shape} = {t:.1f}s.')

# distance in phase space
Dist = ((Theta[:,0]-Theta[:,1])**2 + (Theta_v[:,0]-Theta_v[:,1])**2)**.5
Dist = np.mean(np.log(Dist), axis=0)

plt.figure(figsize=(7,5.5))
plt.title(text)
plt.imshow(Dist, 'inferno', origin='lower', aspect='auto', vmin=-13, vmax=-2,
           extent=[theta0_d.min(),theta0_d.max(),
                   theta_v0_d.min(),theta_v0_d.max()])
plt.colorbar()
plt.xlabel(r'$\theta\ (^\circ)$')
plt.ylabel(r'$\theta_v\ (^\circ)$')
plt.tight_layout()
plt.savefig('image/chaos_map.pdf')
plt.close()
#'''

''' histograms
# hist of Ti ..................................................................
plt.figure()
plt.hist(Ti, int(.2*N_bounce**.5), histtype='stepfilled', ec='k', 
         fc='whitesmoke', lw=1)
plt.grid()
plt.xlabel('Time between collisions '+r'$\hat t$')
plt.ylabel('Numbers')
plt.tight_layout()
plt.savefig(f'image/{case}/hist_Ti.pdf')
plt.close()

# hist of Theta ...............................................................
# parameters
bottom = 1
max_height = .4
angle = np.linspace(0, 2*np.pi, 500)

nums, bins = np.histogram(Theta-np.pi/2, int(.2*N_bounce**.5))
nums = nums * max_height/nums.max()
cen = .5 * (bins[1:] + bins[:-1])
width = cen[1] - cen[0]

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
plt.plot(angle, [1]*len(angle), color='k', lw=1)
bars = plt.bar(cen, nums, width=width, bottom=bottom, fc='whitesmoke', ec='k', 
               lw=.5)
plt.tight_layout()
plt.savefig(f'image/{case}/hist_theta.pdf')
plt.close()

# hist of Theta_v .............................................................
# parameters
bottom = 0
max_height = 1

nums, bins = np.histogram(Theta_v+np.pi/2, int(.2*N_bounce**.5))
nums = nums * max_height/nums.max()
cen = .5 * (bins[1:] + bins[:-1])
width = cen[1] - cen[0]

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
bars = plt.bar(cen, nums, width=width, bottom=bottom, fc='whitesmoke', ec='k', 
               lw=.5)
plt.tight_layout()
plt.savefig(f'image/{case}/hist_theta_v.pdf')
plt.close()
#'''

''' interactive plot ---------------------------------------------------------

# parameters 
theta0_d = 25.2
theta_v0_d = 0
v0 = .63
N_bounce = 10000  # total number of simulated bounces
N_demo = 1000  # number of traces for demo
M = 200  # number of sampling points of parabola
tol = 1e-10  # tolerance of imaginary part

# derived parameters
theta0 = theta0_d * np.pi/180 
x0 = np.sin(theta0)
y0 = -np.cos(theta0) 
theta_v0 = theta_v0_d * np.pi/180  
vx0 = -v0 * np.sin(theta0-theta_v0)
vy0 = v0 * np.cos(theta0-theta_v0)

# solve the problem 
Theta,Theta_v,Vi,Xi,Yi,Vxi,Vyi,Ti = solver(theta0,theta_v0,v0,N_bounce,tol)

# derived array parameters 
Theta_d = Theta * 180/np.pi
Theta_v_d = Theta_v * 180/np.pi

# theta boundary by energy
cos_theta = np.cos(theta0) - .5*v0**2
cos_theta = cos_theta*(cos_theta>=-1) - (cos_theta<-1)
theta_max = np.arccos(cos_theta)
theta_max_d = theta_max * 180/np.pi

# figure parameters 
panel_x = 4  # [in]
panel_y = 4
left = .9
right = .3
bottom = .7
top = .3
width_sld = .3
sep_sld = .2

# derived figure parameters 
fig_x = 3*(width_sld+sep_sld) + 2*(panel_x+left) + right
fig_y = bottom + panel_y + top
lf1 = (3*(width_sld+sep_sld) + left) / fig_x  # 1st panel
lf2 = (3*(width_sld+sep_sld) + 2*left + panel_x) / fig_x
lf_sld1 = sep_sld / fig_x
lf_sld2 = (2*sep_sld + width_sld) / fig_x
lf_sld3 = (3*sep_sld + 2*width_sld) / fig_x
bt = bottom / fig_y
w = panel_x / fig_x  # for panels
w_sld = width_sld / fig_x
h = panel_y / fig_y
s = 1e4 / N_bounce  # marker size

fig = plt.figure(figsize=(fig_x, fig_y))

# trace plot ..................................................................
Ind = np.random.choice(N_bounce, size=N_demo, replace=False)
x = np.full(N_demo*M + N_demo-1, np.nan)
y = x.copy()
for i in range(N_demo):
    ind = Ind[i]
    xi, yi = Xi[ind], Yi[ind]
    vxi, vyi = Vxi[ind], Vyi[ind]
    ti = Ti[ind]
    t = np.linspace(0, ti, M)
    xx = xi + vxi*t
    yy = yi + vyi*t - .5*t**2
    x[i*(M+1) : (i+1)*M+i] = xx
    y[i*(M+1) : (i+1)*M+i] = yy

plt.axes([lf1, bt, w, h])
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
ax = plt.gca()
ax.add_patch(Circle((0,0), 1, ec='grey', fc='none', lw=1))
# initial position
p0 = plt.scatter(x0, y0, ec='k', fc='w', s=30, zorder=48)  
# traces
l, = plt.plot(x, y, color='k', lw=50/N_demo)
# text
text = (r'$N_{\rm bounce}=$' + str(N_bounce) + '\n' + 
        r'$N_{\rm trace\ demo}=$' + str(N_demo))
plt.text(.05, .85, text, fontsize=16, transform=ax.transAxes)
# velocity vector
lv, = plt.plot([x0,x0+vx0], [y0,y0+vy0], color='r', lw=1)
plt.grid()
plt.xlabel(r'$\hat x$')
plt.ylabel(r'$\hat y$')

# scatter plot ................................................................
plt.axes([lf2, bt, w, h])
plt.xlim(-190, 190)
plt.ylim(-100, 100)
# dots
dots = plt.scatter(Theta_d, Theta_v_d, s=s, fc='k', ec='none')
# theta boundaries
x_bound = [-theta_max_d, theta_max_d, theta_max_d, -theta_max_d, -theta_max_d]
y_bound = [-90, -90, 90, 90, -90]
l_bound, = plt.plot(x_bound, y_bound, color='gray', lw=1, ls='--')
plt.grid()
plt.xlabel(r'$\theta\ (^\circ)$')
plt.ylabel(r'$\theta_v\ (^\circ)$')

# sliders
sld1 = Slider(ax=plt.axes([lf_sld1,bt,w_sld,h]), label=r'$\theta_0$', 
        valmin=-180, valmax=180, valinit=theta0_d, orientation='vertical')
sld2 = Slider(ax=plt.axes([lf_sld2,bt,w_sld,h]), label=r'$\theta_{v0}$',
              valmin=-90, valmax=90, valinit=theta_v0_d, 
              orientation='vertical')
sld3 = Slider(ax=plt.axes([lf_sld3,bt,w_sld,h]), label=r'$\hat v_0$',
              valmin=0, valmax=6, valinit=v0, orientation='vertical')

# The function to be called anytime a slider's value changes
def update(val):
    tt0 = time.time()
    # new slider values
    theta0_d = sld1.val 
    theta_v0_d = sld2.val 
    v0 = sld3.val 

    # derived parameters
    theta0 = theta0_d * np.pi/180 
    x0 = np.sin(theta0)
    y0 = -np.cos(theta0) 
    theta_v0 = theta_v0_d * np.pi/180  
    vx0 = -v0 * np.sin(theta0-theta_v0)
    vy0 = v0 * np.cos(theta0-theta_v0)

    # solve the problem 
    Theta,Theta_v,Vi,Xi,Yi,Vxi,Vyi,Ti = solver(theta0,theta_v0,v0,N_bounce,tol)
        
    # derived array parameters 
    Theta_d = Theta * 180/np.pi
    Theta_v_d = Theta_v * 180/np.pi

    # theta boundary by energy
    cos_theta = np.cos(theta0) - .5*v0**2
    cos_theta = cos_theta*(cos_theta>=-1) - (cos_theta<-1)
    theta_max = np.arccos(cos_theta)
    theta_max_d = theta_max * 180/np.pi
    x_bound = [-theta_max_d, theta_max_d, theta_max_d, -theta_max_d, 
               -theta_max_d]

    # traces
    x = np.full(N_demo*M + N_demo-1, np.nan)
    y = x.copy()
    for i in range(N_demo):
        ind = Ind[i]
        xi, yi = Xi[ind], Yi[ind]
        vxi, vyi = Vxi[ind], Vyi[ind]
        ti = Ti[ind]
        t = np.linspace(0, ti, M)
        xx = xi + vxi*t
        yy = yi + vyi*t - .5*t**2
        x[i*(M+1) : (i+1)*M+i] = xx
        y[i*(M+1) : (i+1)*M+i] = yy

    # update plots
    tt1 = time.time()
    p0.set_offsets([x0, y0])
    l.set_data([x, y])
    lv.set_data([[x0,x0+vx0], [y0,y0+vy0]])
    l_bound.set_data([x_bound, y_bound])
    dots.set_offsets(np.column_stack((Theta_d, Theta_v_d)))

    tt2 = time.time()
    tt_all = tt2 - tt0 
    pct_demo = (tt2 - tt1) / tt_all * 100
    print(f'Time: {1e6*tt_all/N_bounce:.2f}µs * {N_bounce} = '
          f'{1e3*tt_all:.1f}ms (demo fraction={pct_demo:.1f}%).')

# register the update function with each slider
sld1.on_changed(update)
sld2.on_changed(update)
sld3.on_changed(update)
plt.show()
#'''



