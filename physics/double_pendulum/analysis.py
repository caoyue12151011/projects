'''
The double-pendulum problem. See knowledge/double_pendulum.md for more 
information.
'''

import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tan 
from scipy.integrate import odeint
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider, Button


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def f(y,t):
    '''
    The differential equations of the double-pendulum. 

    Inputs
    ------
    y: [y1,y2,y3,y4]=[theta1,theta2,dtheta1,dtheta2]
    t: dimensionless time, =t*(g/l)^.5

    Outputs
    -------
    [dy1,dy2,dy3,dy4]: time derivation of y
    '''

    y1, y2, y3, y4 = y 

    y12 = y1-y2
    den = 2-cos(y12)**2 
    num1 = sin(y12)*y4**2 + .5*sin(2*y12)*y3**2 + 4*sin(y1) - 2*cos(y12)*sin(y2) 
    num2=.5*sin(2*y12)*y4**2 + 2*sin(y12)*y3**2 + 4*cos(y12)*sin(y1) - 4*sin(y2)

    dy1 = y3 
    dy2 = y4 
    dy3 = -num1/den 
    dy4 = num2/den

    return [dy1,dy2,dy3,dy4]



def calc_energy(p1,p2,dp1,dp2):
    T = .125*(dp1**2 + .5*dp2**2 + cos(p1-p2)*dp1*dp2)
    V = -.25*(2*cos(p1)+cos(p2))
    E = T+V

    return T, V, E



def solver(y0,t):
    '''
    To solve the double-pendulum problem given i.c. y0 and the t array. Returns 
    the angles, angular velocities.
    '''

    p1, p2, dp1, dp2 = np.transpose(odeint(f,y0,t))

    # check whether energy is conserved
    # E = calc_energy(p1,p2,dp1,dp2)[2]
    # dE_r_max = np.abs(E/np.mean(E)-1).max()
    # print('max |(E-E_av)/E_av|=%f'%dE_r_max)

    return p1, p2, dp1, dp2


# demo =========================================================================

# demo of a case ---------------------------------------------------------------

#''' the case 

# parameters
y0 = [0,0,1,0]
dt = .01
t = np.arange(0,100000+dt,dt)

N = len(t)
p10,p20,dp10,dp20 = y0 
fname = 'case_%s_%s_%s_%s'%(p10,p20,dp10,dp20)
lw = .15*100000/N

# the solution
p1, p2, dp1, dp2 = solver(y0,t) 

x10 = .5*sin(p10)
y10 = -.5*cos(p10)
x20 = x10 + .5*sin(p20)
y20 = y10 - .5*cos(p20)

x1 = .5*sin(p1)
y1 = -.5*cos(p1)
x2 = x1 + .5*sin(p2)
y2 = y1 - .5*cos(p2)

# energies
T0,V0,E0 = calc_energy(p10,p20,dp10,dp20)
T,V,E = calc_energy(p1,p2,dp1,dp2)

# distance to i.c. in phase space
dist_ic = ((p1-p10)**2 + (p2-p20)**2 + (dp1-dp10)**2 + (dp2-dp20)**2)**.5
dist_s_ic = ((p1-p10)**2 + (p2-p20)**2)**.5 # in space
ds = ((x1-x10)**2 + (y1-y10)**2 + (x2-x20)**2 + (y2-y20)**2)**.5
ds1 = ((x1-x10)**2 + (y1-y10)**2)**.5
ds2 = ((x2-x20)**2 + (y2-y20)**2)**.5

# ic text
text_ic = (r'$\theta_{10}=%s$'%p10+'\n'+r'$\theta_{20}=%s$'%p20+'\n'+
    r'$\dot\theta_{10}=%s$'%dp10+'\n'+r'$\dot\theta_{20}=%s$'%dp20+'\n'+
    r'$\tau=0$-$%s$'%t.max())

# create directory 
if not os.path.isdir('image/'+fname):
    os.mkdir('image/'+fname)
#'''


''' influence of time resolution on results ...................................

# parameters
fct = 2 # integer, factor of sparse sampling of time

ts = t[::fct] # sparser time
p1s, p2s, dp1s, dp2s = solver(y0,ts)
p1s0 = p1[::fct]
p2s0 = p2[::fct]
dp1s0 = dp1[::fct]
dp2s0 = dp2[::fct]

# errors
e_p1 = p1s-p1s0
e_p2 = p2s-p2s0
e_dp1 = dp1s-dp1s0
e_dp2 = dp2s-dp2s0

plt.figure(figsize=(16,8))

plt.subplot(212)
plt.xlim(0,ts.max())
plt.plot(ts,e_p1,color='C0',linewidth=1,label='Joint')
plt.plot(ts,e_p2,color='C1',linewidth=1,label='End')
plt.text(.5,.9,'Factor of sparse sampling: %d'%fct,
    transform=plt.gca().transAxes,fontsize=20)
plt.grid()
plt.legend()
plt.xlabel(r'$\tau=t\sqrt{g/l}$')
plt.ylabel('Error of '+r'$\theta$')

plt.subplot(211)
plt.xlim(0,ts.max())
plt.plot(ts,e_dp1,color='C0',linewidth=1)
plt.plot(ts,e_dp2,color='C1',linewidth=1)
plt.grid()
plt.ylabel('Error of '+r'$\dot\theta$')
plt.tight_layout()
plt.savefig('image/%s/sparser_time.pdf'%fname)
plt.close()
#'''


''' phase diagram .............................................................
plt.figure(figsize=(15,5))

# p1 vs p2
plt.subplot(131)
plt.plot(p1,p2,color='k',lw=lw)
plt.scatter(p10,p20,fc='w',ec='k',s=20,zorder=45)
plt.grid()
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')


# p vs dp
plt.subplot(132)
plt.plot(p1,dp1,color='C0',linewidth=lw,zorder=20)
plt.plot(p2,dp2,color='C1',linewidth=lw,zorder=15)
plt.scatter(p10,dp10,fc='C0',ec='k',s=20,zorder=45)
plt.scatter(p20,dp20,fc='C1',ec='k',s=20,zorder=45)

# labels
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot(1000,1000,color='C0',label='Joint')
plt.plot(1000,1000,color='C1',label='End')

# i.c.
text = (r'$\theta_{10}=%s$'%p10+'\n'+r'$\theta_{20}=%s$'%p20+'\n'+
    r'$\dot\theta_{10}=%s$'%dp10+'\n'+r'$\dot\theta_{20}=%s$'%dp20+'\n'+
    r'$\tau=0$-$%.2f$'%t.max())
plt.text(.05,.65,text,transform=plt.gca().transAxes,fontsize=14,zorder=45)

plt.legend()
plt.grid()
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\dot\theta$')


# dp1 vs dp2
plt.subplot(133)
plt.plot(dp1,dp2,color='k',lw=lw)
plt.scatter(dp10,dp20,fc='w',ec='k',s=20,zorder=45)

# ellipse of max T
width = 2*(8*(3+5**.5)*(3/4+E0))**.5
height =2*(8*(3-5**.5)*(3/4+E0))**.5
angle = 180/np.pi*(np.pi/2+np.arctan(2)/2)
ell = Ellipse((0,0),width,height,angle,fc='none',ec='r',linestyle='--',
    label=r'$\hat{T}_{\rm max}$')
plt.gca().add_patch(ell)

plt.legend()
plt.grid()
plt.xlabel(r'$\dot\theta_1$')
plt.ylabel(r'$\dot\theta_2$')

plt.tight_layout()
plt.savefig('image/%s/phase.pdf'%fname)
plt.close()
#'''


''' T-shirt ...................................................................

alpha = 25*np.pi/180 

c = cos(alpha)
s = sin(alpha)

dp1_r = c*dp1+s*dp2
dp2_r =-s*dp1+c*dp2

plt.figure(figsize=(5.8,8))
plt.plot(dp1_r,dp2_r,color='k',lw=lw)
plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.tight_layout()
plt.savefig('image/T-shirt.pdf')
plt.show()
#'''


''' properties vs tau ..........................................................
plt.figure(figsize=(16,10))

# theta vs tau
plt.subplot(414)
plt.xlim(t.min(),t.max())
plt.plot(t,p1,color='C0',lw=lw,zorder=20,label='Joint')
plt.plot(t,p2,color='C1',lw=lw,zorder=15,label='End')

# i.c.
text = (r'$\theta_{10}=%s$'%p10+'\n'+r'$\theta_{20}=%s$'%p20+'\n'+
    r'$\dot\theta_{10}=%s$'%dp10+'\n'+r'$\dot\theta_{20}=%s$'%dp20+'\n'+
    r'$\tau=0$-$%.2f$'%t.max())
plt.text(.9,.3,text,transform=plt.gca().transAxes,fontsize=14,zorder=45)

plt.legend()
plt.grid()
plt.xlabel(r'$\tau=t\sqrt{g/l}$')
plt.ylabel(r'$\theta$')


# dtheta vs tau
plt.subplot(413)
plt.xlim(t.min(),t.max())
plt.plot(t,dp1,color='C0',lw=lw,zorder=20)
plt.plot(t,dp2,color='C1',lw=lw,zorder=15)
plt.grid()
plt.gca().set_xticklabels([])
plt.ylabel(r'$\dot\theta$')

# T vs tau
plt.subplot(412)
plt.xlim(t.min(),t.max())
plt.plot(t,T,color='k',lw=lw)
plt.grid()
plt.gca().set_xticklabels([])
plt.ylabel(r'$\hat{T}=T/(mgl)$')

# dist_ic vs tau
plt.subplot(411)
plt.xlim(t.min(),t.max())
plt.plot(t,dist_ic,color='k',lw=lw)
plt.grid()
plt.gca().set_xticklabels([])
plt.ylabel('Dist. to I.C.')

plt.tight_layout()
plt.savefig('image/%s/time_evol.pdf'%fname)
plt.close()
#'''


''' hist of properties .........................................................
plt.figure(figsize=(15,5))

# hist of theta
bins1 = np.linspace(p1.min(),p1.max(),int(.5*N**.5))
bins2 = np.linspace(p2.min(),p2.max(),int(.5*N**.5))

plt.subplot(131)
plt.hist(p1,bins1,color='C0',histtype='step',density=True,label='Joint')
plt.hist(p2,bins2,color='C1',histtype='step',density=True,label='End')

# i.c.
text = (r'$\theta_{10}=%s$'%p10+'\n'+r'$\theta_{20}=%s$'%p20+'\n'+
    r'$\dot\theta_{10}=%s$'%dp10+'\n'+r'$\dot\theta_{20}=%s$'%dp20+'\n'+
    r'$\tau=0$-$%.2f$'%t.max())
plt.text(.05,.4,text,transform=plt.gca().transAxes,fontsize=14,zorder=45)

plt.grid()
plt.legend()
plt.xlabel(r'$\theta$')
plt.ylabel('Probability density')


# hist of dtheta
bins1 = np.linspace(dp1.min(),dp1.max(),int(.5*N**.5))
bins2 = np.linspace(dp2.min(),dp2.max(),int(.5*N**.5))

plt.subplot(132)
plt.hist(dp1,bins1,color='C0',histtype='step',density=True)
plt.hist(dp2,bins2,color='C1',histtype='step',density=True)
plt.grid()
plt.xlabel(r'$\dot\theta$')


# hist of T
bins = np.linspace(T.min(),T.max(),int(.5*N**.5))

plt.subplot(133)
plt.hist(T,bins,color='k',histtype='step',density=True)
plt.grid()
plt.xlabel(r'$\hat{T}=T/(mgl)$')

plt.tight_layout()
plt.savefig('image/%s/hist.pdf'%fname)
plt.close()
#'''


''' min(dist_ic) vs tau .......................................................

# remove the first dynamic timescale
Ind = t>1
t1 = t[Ind]

min_dist_ic = np.minimum.accumulate(dist_ic[Ind])
min_dist_s_ic = np.minimum.accumulate(dist_s_ic[Ind])
min_ds = np.minimum.accumulate(ds[Ind])
min_ds1 = np.minimum.accumulate(ds1[Ind])
min_ds2 = np.minimum.accumulate(ds2[Ind])

# demo
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(t1,min_dist_ic,label='Phase space')
plt.plot(t1,min_dist_s_ic,label='Space')
plt.text(.05,.5,text_ic,transform=plt.gca().transAxes,fontsize=14,zorder=45)
plt.legend()
plt.grid()
plt.xlabel(r'$\tau$')
plt.ylabel('min '+r'$D_{\rm IC}(\leq\tau)$')
plt.tight_layout()
plt.savefig('image/%s/dist_ic.pdf'%fname)
plt.close()

# a practical case .............................................................

# parameters
l = 1 # [m]
g = 9.8 # [m/s^2]

ds_m = min_ds*l # [m]
t1_s = t1*(l/g)**.5 # [s]
text = text_ic + '\n'+r'$l=%s{\rm\ m}$'%l + '\n'+r'$g=%s{\rm\ m\ s^{-2}}$'%g

# linear fitting 
x = np.log10(t1_s)
y = np.log10(ds_m)
x_av = np.mean(x)
y_av = np.mean(y)
k = np.sum((x-x_av)*(y-y_av))/np.sum((x-x_av)**2)
b = y_av-k*x_av 

xx = np.array([-.3,13])
yy = k*xx+b


# demo
plt.figure(figsize=(6,4))
plt.xscale('log')
plt.yscale('log')
plt.plot(t1_s,min_ds,color='k')
plt.plot(10**xx,10**yy,color='r',ls='--')
plt.text(.7,.45,text,transform=plt.gca().transAxes,zorder=45)

ax = plt.gca()
ax.xaxis.set_ticks([1,60,3600,86400,86400*7,86400*365,
    86400*36500,86400*365e3,86400*365e6])
ax.xaxis.set_ticklabels(['1s','1m','1h','1d','1wk','1yr',
    '1c','1kyr','1Myr'])
ax.yaxis.set_ticks([.1,.01,.001,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10])
ax.yaxis.set_ticklabels(['1dm','1cm','1mm','0.1mm',r'$10\rm\mu m$',
    r'$1\rm\mu m$',r'$0.1\rm\mu m$','10nm','1nm',r'$1\AA$'])

plt.grid()
plt.xlabel('Time')
plt.ylabel('Min displacement rel. to i.c.')
plt.tight_layout()
plt.savefig('image/%s/ds_t.pdf'%fname)
plt.close()


# point 1/2
plt.figure(figsize=(6,4))
plt.xscale('log')
plt.yscale('log')
plt.plot(t1_s,min_ds1,label='Point 1')
plt.plot(t1_s,min_ds2,label='Point 2')
plt.text(.7,.45,text,transform=plt.gca().transAxes,zorder=45)
plt.legend()
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Min displacement rel. to i.c. (m)')
plt.tight_layout()
plt.savefig('image/%s/ds1_ds2_t.pdf'%fname)
plt.close()
#'''


# interactive plots ------------------------------------------------------------

''' phase diagram

# parameters
y0 = [0,0,1.395,0]
t = np.arange(0,300,.01)
lw = .2 # linewidth


# solutions
N = len(t)
p10,p20,dp10,dp20 = y0 
p1, p2, dp1, dp2 = solver(y0,t) 


# figure parameters
wp = 4 # [in]
hp = 4 
lp = 1.2
rp = .2 
bp = .8
tp = .2 
gp = 1.2
ls = 1
rs = 1 
gs = 1.5
hs = .2 
bs = .2 

# derived figure parameters 
fig_x = lp+3*wp+2*gp+rp 
fig_y = tp+hp+bp+3*hs+bs 
fx_p1 = lp/fig_x 
fx_p2 = (lp+wp+gp)/fig_x 
fx_p3 = 1-(wp+rp)/fig_x 
fy_p = 1-(tp+hp)/fig_y 
fw_p = wp/fig_x 
fh_p = hp/fig_y 

ws = (fig_x-ls-gs-rs)/2
fx_s13 = ls/fig_x 
fx_s24 = (ls+ws+gs)/fig_x 
fy_s12 = (2*hs+bs)/fig_y
fy_s34 = bs/fig_y
fw_s = ws/fig_x
fh_s = hs/fig_y


fig = plt.figure(figsize=(fig_x,fig_y))

# p1 vs p2 .....................................................................

ax1 = plt.axes([fx_p1,fy_p,fw_p,fh_p])
curve1,=plt.plot(p1,p2,color='k',lw=lw)
ic1=plt.scatter(p10,p20,fc='w',ec='k',s=20,zorder=45)

# i.c.
text = r'$\tau=0$-$%.2f$'%t.max()
plt.text(.05,.9,text,transform=ax1.transAxes,fontsize=14,zorder=45)

plt.grid()
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')

# p vs dp ......................................................................

ax2 = plt.axes([fx_p2,fy_p,fw_p,fh_p])
curve2_p1,=plt.plot(p1,dp1,color='C0',linewidth=lw,zorder=20)
curve2_p2,=plt.plot(p2,dp2,color='C1',linewidth=lw,zorder=15)
ic2_p1=plt.scatter(p10,dp10,fc='C0',ec='k',s=20,zorder=45)
ic2_p2=plt.scatter(p20,dp20,fc='C1',ec='k',s=20,zorder=45)

# labels
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot(1000,1000,color='C0',label='Joint')
plt.plot(1000,1000,color='C1',label='End')
plt.legend()
plt.grid()
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\dot\theta$')

# dp1 vs dp2 ...................................................................

ax3 = plt.axes([fx_p3,fy_p,fw_p,fh_p])
curve3,=plt.plot(dp1,dp2,color='k',lw=lw)
ic3=plt.scatter(dp10,dp20,fc='w',ec='k',s=20,zorder=45)
plt.grid()
plt.xlabel(r'$\dot\theta_1$')
plt.ylabel(r'$\dot\theta_2$')

# sliders ......................................................................

# theta10 slider
sld1 = Slider(ax=plt.axes([fx_s13,fy_s12,fw_s,fh_s]),label=r'$\theta_{10}$',
    valmin=-3,valmax=3,valinit=p10)

# theta20 slider
sld2 = Slider(ax=plt.axes([fx_s24,fy_s12,fw_s,fh_s]),label=r'$\theta_{20}$',
    valmin=-3,valmax=3,valinit=p20)

# dtheta10 slider
sld3 = Slider(ax=plt.axes([fx_s13,fy_s34,fw_s,fh_s]),label=r'$\dot\theta_{10}$',
    valmin=-3,valmax=3,valinit=dp10)

# dtheta20 slider
sld4 = Slider(ax=plt.axes([fx_s24,fy_s34,fw_s,fh_s]),label=r'$\dot\theta_{20}$',
    valmin=-3,valmax=3,valinit=dp20)


# The function to be called anytime a slider's value changes
def update(val):

    # new slider values
    p10 = sld1.val 
    p20 = sld2.val 
    dp10 = sld3.val 
    dp20 = sld4.val 

    # new solutions
    p1, p2, dp1, dp2 = solver([p10,p20,dp10,dp20],t) 

    # update plots
    curve1.set_xdata(p1)
    curve1.set_ydata(p2)
    ic1.set_offsets([p10,p20])

    curve2_p1.set_xdata(p1)
    curve2_p1.set_ydata(dp1)
    curve2_p2.set_xdata(p2)
    curve2_p2.set_ydata(dp2)
    ic2_p1.set_offsets([p10,dp10])
    ic2_p2.set_offsets([p20,dp20])

    curve3.set_xdata(dp1)
    curve3.set_ydata(dp2)
    ic3.set_offsets([dp10,dp20])

    ax1.set_xlim(p1.min(),p1.max())
    ax1.set_ylim(p2.min(),p2.max())
    ax2.set_xlim(min(p1.min(),p2.min()),max(p1.max(),p2.max()))
    ax2.set_ylim(min(dp1.min(),dp2.min()),max(dp1.max(),dp2.max()))
    ax3.set_xlim(dp1.min(),dp1.max())
    ax3.set_ylim(dp2.min(),dp2.max())

    fig.canvas.draw_idle()

# register the update function with each slider
sld1.on_changed(update)
sld2.on_changed(update)
sld3.on_changed(update)
sld4.on_changed(update)
plt.savefig('image/interactive_phase.pdf')
plt.show()
#'''


#''' hist of properties

# parameters
y0 = [0,0,1,0]
t = np.arange(0,300,.01)

# solutions
N = len(t)
p10,p20,dp10,dp20 = y0 
p1, p2, dp1, dp2 = solver(y0,t) 
T = dp1**2 + .5*dp2**2 + cos(p1-p2)*dp1*dp2


# figure parameters
wp = 4 # [in]
hp = 4 
lp = 1.2
rp = .2 
bp = .8
tp = .2 
gp = 1.2
ls = 1
rs = 1 
gs = 1.5
hs = .2 
bs = .2 

# derived figure parameters 
fig_x = lp+3*wp+2*gp+rp 
fig_y = tp+hp+bp+3*hs+bs 
fx_p1 = lp/fig_x 
fx_p2 = (lp+wp+gp)/fig_x 
fx_p3 = 1-(wp+rp)/fig_x 
fy_p = 1-(tp+hp)/fig_y 
fw_p = wp/fig_x 
fh_p = hp/fig_y 

ws = (fig_x-ls-gs-rs)/2
fx_s13 = ls/fig_x 
fx_s24 = (ls+ws+gs)/fig_x 
fy_s12 = (2*hs+bs)/fig_y
fy_s34 = bs/fig_y
fw_s = ws/fig_x
fh_s = hs/fig_y


fig = plt.figure(figsize=(fig_x,fig_y))

# hist of theta ................................................................

bins1 = np.linspace(p1.min(),p1.max(),int(.5*N**.5))
bins2 = np.linspace(p2.min(),p2.max(),int(.5*N**.5))

ax1 = plt.axes([fx_p1,fy_p,fw_p,fh_p])
plt.hist(p1,bins1,color='C0',histtype='step',density=True,label='Joint')
plt.hist(p2,bins2,color='C1',histtype='step',density=True,label='End')

# i.c.
text = r'$\tau=0$-$%.2f$'%t.max()
plt.text(.05,.7,text,transform=ax1.transAxes,fontsize=14,zorder=45)

plt.grid()
plt.legend()
plt.xlabel(r'$\theta$')
plt.ylabel('Probability density')

# hist of dtheta ...............................................................

bins1 = np.linspace(dp1.min(),dp1.max(),int(.5*N**.5))
bins2 = np.linspace(dp2.min(),dp2.max(),int(.5*N**.5))

ax2 = plt.axes([fx_p2,fy_p,fw_p,fh_p])
plt.hist(dp1,bins1,color='C0',histtype='step',density=True)
plt.hist(dp2,bins2,color='C1',histtype='step',density=True)
plt.grid()
plt.xlabel(r'$\dot\theta$')

# hist of T ....................................................................

bins = np.linspace(T.min(),T.max(),int(.5*N**.5))

ax3 = plt.axes([fx_p3,fy_p,fw_p,fh_p])
plt.hist(T,bins,color='k',histtype='step',density=True)
plt.grid()
plt.xlabel(r'$\hat{T}=T/(mgl)$')

# sliders ......................................................................

# theta10 slider
sld1 = Slider(ax=plt.axes([fx_s13,fy_s12,fw_s,fh_s]),label=r'$\theta_{10}$',
    valmin=-3,valmax=3,valinit=p10)

# theta20 slider
sld2 = Slider(ax=plt.axes([fx_s24,fy_s12,fw_s,fh_s]),label=r'$\theta_{20}$',
    valmin=-3,valmax=3,valinit=p20)

# dtheta10 slider
sld3 = Slider(ax=plt.axes([fx_s13,fy_s34,fw_s,fh_s]),label=r'$\dot\theta_{10}$',
    valmin=-3,valmax=3,valinit=dp10)

# dtheta20 slider
sld4 = Slider(ax=plt.axes([fx_s24,fy_s34,fw_s,fh_s]),label=r'$\dot\theta_{20}$',
    valmin=-3,valmax=3,valinit=dp20)


# The function to be called anytime a slider's value changes
def update(val):

    # new slider values
    p10 = sld1.val 
    p20 = sld2.val 
    dp10 = sld3.val 
    dp20 = sld4.val 

    # new solutions
    p1, p2, dp1, dp2 = solver([p10,p20,dp10,dp20],t) 
    T = dp1**2 + .5*dp2**2 + cos(p1-p2)*dp1*dp2

    # hist of theta ............................................................

    bins1 = np.linspace(p1.min(),p1.max(),int(.5*N**.5))
    bins2 = np.linspace(p2.min(),p2.max(),int(.5*N**.5))

    ax1.cla()
    ax1.hist(p1,bins1,color='C0',histtype='step',density=True,label='Joint')
    ax1.hist(p2,bins2,color='C1',histtype='step',density=True,label='End')

    # i.c.
    text = r'$\tau=0$-$%.2f$'%t.max()
    ax1.text(.05,.9,text,transform=ax1.transAxes,fontsize=14,zorder=45)

    ax1.legend()
    ax1.grid()
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel('Probability density')

    # hist of dtheta ...........................................................

    bins1 = np.linspace(dp1.min(),dp1.max(),int(.5*N**.5))
    bins2 = np.linspace(dp2.min(),dp2.max(),int(.5*N**.5))

    ax2.cla()
    ax2.hist(dp1,bins1,color='C0',histtype='step',density=True)
    ax2.hist(dp2,bins2,color='C1',histtype='step',density=True)
    ax2.grid()
    ax2.set_xlabel(r'$\dot\theta$')

    # hist of T ................................................................

    bins = np.linspace(T.min(),T.max(),int(.5*N**.5))

    ax3.cla()
    ax3.hist(T,bins,color='k',histtype='step',density=True)
    ax3.grid()
    ax3.set_xlabel(r'$\hat{T}=T/(mgl)$')

# register the update function with each slider
sld1.on_changed(update)
sld2.on_changed(update)
sld3.on_changed(update)
sld4.on_changed(update)
plt.savefig('image/interactive_hist.pdf')
plt.show()
#'''


''' trajectories 

# parameters
y0 = [0,0,0,0]
t = np.arange(0,300,.01)
lw = .1 # linewidth 
h_max = 1. # max spatial length

# solutions
p10,p20,dp10,dp20 = y0 
E = calc_energy(p10,p20,dp10,dp20)[2]

p1, p2, dp1, dp2 = solver(y0,t) 
x1 = .5*sin(p1)
y1 = -.5*cos(p1)
x2 = .5*(sin(p1)+sin(p2))
y2 = .5*(-cos(p1)-cos(p2))


# figure parameters
sp = 4 # [in]
lp = 1
rp = .4
bp = .6
tp = .4
ws = .2
rs = .4
gs = .4
bs = .5
ts = .5 

# derived figure parameters 
fig_x = lp+sp+rp+4*ws+3*gs+rs 
fig_y = tp+sp+bp 

fx_p = lp/fig_x 
fy_p = bp/fig_y 
fw_p = sp/fig_x 
fh_p = sp/fig_y 

fx_s1 = (lp+sp+rp)/fig_x 
fx_s2 = (lp+sp+rp+ws+gs)/fig_x 
fx_s3 = (lp+sp+rp+2*(ws+gs))/fig_x 
fx_s4 = (lp+sp+rp+3*(ws+gs))/fig_x 
fy_s = bs/fig_y
fw_s = ws/fig_x
fh_s = 1-(ts+bs)/fig_y


fig = plt.figure(figsize=(fig_x,fig_y))

# trajectories .................................................................

ax = plt.axes([fx_p,fy_p,fw_p,fh_p])
ax.set_xlim(-h_max,h_max)
ax.set_ylim(-h_max,h_max)
l,=ax.plot(x2,y2,color='k',lw=lw)

# max V curve 
Y0 = np.linspace(-1,1,5000)
A = .5*(1-4*(Y0-2*E)**2)**.5
B = .5*(1-16*(Y0-E)**2)**.5
X1 =  A+B
X2 = -A+B
X3 =  A-B
X4 = -A-B 
Y = np.concatenate((Y0,Y0,Y0,Y0))
X = np.concatenate((X1,X2,X3,X4))
l_maxV,=plt.plot(X,Y,color='r')

# i.c.
ax.scatter(0,0,fc='k',ec='k',s=20,zorder=40)
dot=ax.scatter([x1[0],x2[0]],[y1[0],y2[0]],fc='w',ec='k',s=20,zorder=40,
    label='Initial position')
l_rod,=ax.plot([0,x1[0],x2[0]],[0,y1[0],y2[0]],color='k')

# time range
text = r'$\tau=0$-$%.2f$'%t.max()+'\n'+r'$\hat E=%.2f$'%E
txt = ax.text(.05,.85,text,transform=ax.transAxes,fontsize=14,zorder=45)

plt.grid()
plt.legend()
plt.xlabel(r'$\hat{x}_2=x_2/l$')
plt.ylabel(r'$\hat{y}_2=y_2/l$')

# sliders ......................................................................

# theta10 slider
sld1 = Slider(ax=plt.axes([fx_s1,fy_s,fw_s,fh_s]),label=r'$\theta_{10}$',
    valmin=-np.pi,valmax=np.pi,valinit=p10,orientation='vertical')
sld1.hline._linewidth = 0 # remove the initial line

# theta20 slider
sld2 = Slider(ax=plt.axes([fx_s2,fy_s,fw_s,fh_s]),label=r'$\theta_{20}$',
    valmin=-np.pi,valmax=np.pi,valinit=p20,orientation='vertical')
sld2.hline._linewidth = 0 # remove the initial line

# dtheta10 slider
sld3 = Slider(ax=plt.axes([fx_s3,fy_s,fw_s,fh_s]),label=r'$\dot\theta_{10}$',
    valmin=-3,valmax=3,valinit=dp10,orientation='vertical')
sld3.hline._linewidth = 0 # remove the initial line

# dtheta20 slider
sld4 = Slider(ax=plt.axes([fx_s4,fy_s,fw_s,fh_s]),label=r'$\dot\theta_{20}$',
    valmin=-3,valmax=3,valinit=dp20,orientation='vertical')
sld4.hline._linewidth = 0 # remove the initial line


# The function to be called anytime a slider's value changes
def update(val):

    # new slider values
    p10 = sld1.val 
    p20 = sld2.val 
    dp10 = sld3.val 
    dp20 = sld4.val 

    # new solutions
    p1, p2, dp1, dp2 = solver([p10,p20,dp10,dp20],t) 
    E = calc_energy(p10,p20,dp10,dp20)[2]
    x1 = .5*sin(p1)
    y1 = -.5*cos(p1)
    x2 = .5*(sin(p1)+sin(p2))
    y2 = .5*(-cos(p1)-cos(p2))

    # max V curve 
    A = .5*(1-4*(Y0-2*E)**2)**.5
    B = .5*(1-16*(Y0-E)**2)**.5
    X1 =  A+B
    X2 = -A+B
    X3 =  A-B
    X4 = -A-B 
    X = np.concatenate((X1,X2,X3,X4))

    # update plots
    dot.set_offsets([[x1[0],y1[0]],[x2[0],y2[0]]])
    l_rod.set_xdata([0,x1[0],x2[0]])
    l_rod.set_ydata([0,y1[0],y2[0]])
    l_maxV.set_xdata(X)
    txt.set_text(r'$\tau=0$-$%.2f$'%t.max()+'\n'+r'$\hat E=%.2f$'%E)
    l.set_xdata(x2)
    l.set_ydata(y2)

    fig.canvas.draw_idle()

# register the update function with each slider
sld1.on_changed(update)
sld2.on_changed(update)
sld3.on_changed(update)
sld4.on_changed(update)
plt.savefig('image/interactive_trace.pdf')
plt.show()
#'''



