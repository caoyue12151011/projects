'''
This script simulates some physcial processes based on the models in model.py.

Some constants:
d [m]
    0.24: basketball

mu [kg/m/s]
    1.81e-5: air at 15 Cel and 1 atm
    1.14e-3: water at 15 Cel

rho [kg/m^3]
    1.225: air at 15 Cel and 1 atm

rho_s [kg/m^3]
    2000: PM2.5 (roughly)
'''

import os
import dill
import num2tex
import matplotlib
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patheffects as path_effects
from ambiance import Atmosphere as Atm 
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


# load variables
spec_func = dill.load(open('variable/spec_func.p','rb'))
Cd_vs_Re = dill.load(open('variable/Cd_vs_Re.p','rb'))
Cd_vs_ReMa = dill.load(open('variable/Cd_vs_ReMa.p','rb'))
drag = dill.load(open('variable/drag.p','rb'))
Atm_data = dill.load(open('variable/Atm_data.p','rb'))
calc_atm = dill.load(open('variable/calc_atm.p','rb'))

h_min = Atm_data['h'][0]
h_max = Atm_data['h'][-1]

# constants 
h_tp = 11019 # [m], tropopause
h_sp = 47350 # stratopause
h_mp = 86000 # mesopause
g = 9.8
sigma_st = 5.67e-8 # [W/m2/K4], ST constant


def integrator(x,y,vx,vy,para,dh, Atm_data):
    '''
    To integrate the diff eq of a falling ball by one step.

    Inputs
    ------
    x,y,vx,vy: coordinates & velocities of the previous step 
    para: [d,rho_s,g], relevant physical parameters
    dh: [m], vertical displacement in one step
    Atm_data: dict, the atmosphere model

    Returns
    -------
    dt,x1,y1,vx1,vy1: time interval, parameters of the next step 
    '''

    d,rho_s,g = para 
    m = np.pi/6*d**3*rho_s # mass

    # determine dt
    dt = abs(dh/vy) if (not vy==0.) else (2*dh/g)**.5 # in space

    # increments
    v = (vx**2+vy**2)**.5
    rho, mu, cs = calc_atm(y, ['rho','mu','cs'], Atm_data)
    Fd = drag(v,d,rho,mu,cs)
    dvxdt = 0 if np.isnan(rho) else -Fd*vx/(m*v)
    dvydt = -g if np.isnan(rho) else (rho/rho_s-1)*g - Fd*vy/(m*v)

    # integration
    vx1 = vx + dvxdt*dt
    vy1 = vy + dvydt*dt
    x1 = x + (vx+vx1)*dt/2
    y1 = y + (vy+vy1)*dt/2

    return dt,x1,y1,vx1,vy1


def solver(x0,y0,vx0,vy0,para,dh,term_cond='ground'):
    '''
    To solve the diff eq of a falling ball.

    Inputs
    ------
    x0,y0,vx0,vy0: i.c.
    para: [d,rho_s,g], relevant physical parameters
    dh: [m], vertical displacement in one step
    term_cond: condition for terminating the integration

    Returns
    -------
    T, X, Y, Vx, Vy: 1d arrays, solutions
    '''

    d,rho_s,g = para 

    # initial values
    t, x, y, vx, vy = 0,x0,y0,vx0,vy0
    T, X, Y, Vx, Vy = [t],[x],[y],[vx],[vy]

    # solve the eq
    if term_cond=='ground':
        while y>=0:
            dt,x,y,vx,vy = integrator(x,y,vx,vy,para,dh)
            t += dt 
            T.append(t) 
            X.append(x) 
            Y.append(y) 
            Vx.append(vx)
            Vy.append(vy)

    elif term_cond=='top':
        v_th = 2*(2*g*dh)**.5
        while y<=h_max and vy>v_th:
            dt,x,y,vx,vy = integrator(x,y,vx,vy,para,dh)
            t += dt 
            T.append(t) 
            X.append(x) 
            Y.append(y) 
            Vx.append(vx)
            Vy.append(vy)   

    T,X,Y,Vx,Vy = [np.array(i) for i in [T,X,Y,Vx,Vy]]

    return T, X, Y, Vx, Vy


# demo =========================================================================

''' vertical fall 

# parameters
x0 = 0 
vx0 = 0 
y0 = 1e5 # [m]
vy0 = 0

d = np.array([.01,.03,.1,.3,1,3,10]) # [m]
rho_s = 2600 # [kg/m^3]
dh = 30 # [m]

epsilon = .5 # coefficient of drag power converting to heat on the ball 
c = 750 # [J/kg/K], specific heat capacity
tt0 = 300 # [K], initial temperature


# solve the problem 
T,Y,Vy,Ld,Re,Ma,Cd,TT = [],[],[],[],[],[],[],[]
for i in range(len(d)):
    # if not i==0:
    #     continue

    para = [d[i],rho_s,g]
    t, x, y, vx, vy = solver(x0,y0,vx0,vy0,para,dh)
    print('d=%.1f. No. of points = %d.'%(d[i],len(t)))

    rho, mu, cs = calc_atm(y, ['rho','mu','cs'], Atm_data)
    v = (vx**2+vy**2)**.5 
    re = rho*v*d[i]/mu 
    ma = v/cs 
    cd = Cd_vs_ReMa(re,ma,debug=False,broadcast=False)

    T.append(t)
    Y.append(y)
    Vy.append(-vy)
    Ld.append(np.gradient(vy)/np.gradient(t)/g+1) # overload
    Re.append(re)
    Ma.append(ma)
    Cd.append(cd)

    tt = np.ones(len(t))
    dt = np.gradient(t)
    tmp = tt0
    for j in range(len(t)):
        tmp_pos = -epsilon*cd[j]*rho[j]*vy[j]**3/8
        if np.isnan(tmp_pos):
            tmp_pos = 0 
        tmp_neg = sigma_st*tmp**4

        tmp += 6/(c*rho_s*d[i])*(tmp_pos-tmp_neg)*dt[j]

        tt[j] = tmp
    TT.append(tt)

TT_c = np.array(TT)-273.15


# data for plotting
data = [Vy,Ld,Re,Ma,Cd,TT_c]
name = ['v','overload','Re','Ma','Cd','temperature']
ylabel = [r'$v_y\rm\ (m\ s^{-1})$','Overload (g)','Re','Ma',r'$C_{\rm d}$',
    'Temperature '+r'$\rm(^\circ C)$']
yscale = ['linear','linear','log','linear','linear','linear']
note1 = ('ICAO Std. Atm. (1993)\n'+
        r'$\rho_{\rm s}=%.1f\rm\ g\ cm^{-3}$'%(rho_s/1e3)+'\n'+
        r'$v_{y0}=%s\rm\ m\ s^{-1}$'%vy0+'\n'+
        r'$g=%.1f\rm\ m\ s^{-2}$'%g+'\n')
colors = matplotlib.cm.get_cmap('rainbow_r')(np.linspace(0,1,len(d)))

# properties vs height 
for i in range(len(data)):
    # if i==0:
    #     break

    plt.figure(figsize=(8,5))
    plt.xlim(1.01*y0/1e3,0)
    plt.yscale(yscale[i])

    # plot data
    for j in range(len(d)):
        minute, sec = divmod(T[j][-1],60)
        label = r'$d=%s\rmm\ (%dm%.0fs)$'%(d[j],minute,sec)
        plt.plot(Y[j]/1e3,data[i][j],color=colors[j],label=label,lw=2)
        # plt.scatter(Y[j]/1e3,data[i][j],fc=colors[j],ec='none',s=5)

    # text 
    plt.text(.05,.65,note1,fontsize=12,transform=plt.gca().transAxes)

    plt.legend(fontsize=12)
    plt.grid()
    plt.xlabel(r'$H$'+' (km)')
    plt.ylabel(ylabel[i])
    plt.tight_layout()
    plt.savefig('image/vertical_fall/%s_vs_height.pdf'%name[i])
    plt.show()
#'''


#''' vertical fall animation

# control panel \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# I.C.
x0 = 0 
vx0 = 0 
y0 = 1e5 # [m]
vy0 = 0
tt0 = 300 # [K], initial temperature

# properties
d = 1 # [m]
rho_s = 2600 # [kg/m^3]
c = 750 # [J/kg/K], specific heat capacity
epsilon = .5 # coefficient of drag power converting to heat on the ball 

# computational parameters
dh = 30 # [m]

# //////////////////////////////////////////////////////////////////////////////

# solve the problem 
para = [d,rho_s,g]
t, x, y, vx, vy = solver(x0,y0,vx0,vy0,para,dh)
print('No. of points: %d.'%len(t))

rho, mu, cs = calc_atm(y, ['rho','mu','cs'], Atm_data)
v = (vx**2+vy**2)**.5 
re = rho*v*d/mu 
ma = v/cs 
cd = Cd_vs_ReMa(re,ma,debug=False,broadcast=False)
ld = np.gradient(vy)/np.gradient(t)/g+1 # overload
ld[np.abs(ld)<1e-10] = 0

tt = np.ones(len(t)) # [K] temperature
dt = np.gradient(t)
tmp = tt0
for j in range(len(t)):
    tmp_pos = -epsilon*cd[j]*rho[j]*vy[j]**3/8
    if np.isnan(tmp_pos):
        tmp_pos = 0 
    tmp_neg = sigma_st*tmp**4
    tmp += 6/(c*rho_s*d)*(tmp_pos-tmp_neg)*dt[j]
    tt[j] = tmp
tt_c = np.array(tt)-273.15 # [C] temperature

# animations ...................................................................

# data for plotting
fps = 24
dpi = 100
t_factor = 20 # x times faster
duration = t[-1]/t_factor
note1 = ('ICAO Std. Atm. (1993)\n'+
        r'$g=%.1f\rm\ m\ s^{-2}$'%g+'\n'+
        r'$d=%s\rm\ m$'%d+'\n'+
        r'$\rho_{\rm s}=%.1f\rm\ g\ cm^{-3}$'%(rho_s/1e3)+'\n'+
        r'$c=%s\rm\ J\ kg^{-1}K^{-1}$'%c+'\n'+
        r'$\epsilon_{\rm heat}=%s$'%epsilon)
note2 = (r'$t=%.1f\rm\ s$'%t[0]+'\n'+
        r'$H=%.2f\rm\ km$'%(y0/1e3)+'\n'+
        r'$v=%.1f\rm m\ s^{-1}$'%vy0+'\n'+
        r'Ma=%.2f'%ma[0]+'\n'+
        r'g-force=%.2f'%ld[0]+'\n'+
        r'$T=%.1f\rm^\circ C$'%tt_c[0])
ticks = np.arange(0,y0/1e3+5,10) # [km]
labels = ['%.0f'%i for i in ticks]

x_min = 0
x_max = 1
y_min = -.05*y0/1e3
y_max = 1.05*y0/1e3

# background image
Img = calc_atm(np.linspace(0,100,400)*1e3, 'rho', Atm_data)
Img[np.isnan(Img)] = 0 
Img = np.array([Img**.3]).transpose()


fig = plt.figure(figsize=(4,6),dpi=dpi)
ax = plt.gca()
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)

dot = plt.scatter(.2,y0/1e3,fc='w',ec='k',s=60,zorder=49)
plt.imshow(Img,origin='lower',extent=[x_min,x_max,y_min,y_max],cmap='seismic',
    aspect='auto',vmax=1.7*Img.max())

# static text
plt.text(.4,.7,note1,fontsize=14,color='w',transform=ax.transAxes)
# dynamic text 
txt = plt.text(.4,.4,note2,fontsize=14,color='w',transform=ax.transAxes)

plt.axhspan(-.05*y0/1e3,0,fc='orange',ec='none')
plt.axhline(0,color='k',lw=2)
plt.grid()
plt.xticks([])
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
plt.ylabel('Height (km)')
plt.tight_layout()

def make_frame(t1):
    y1 = np.interp(t1*t_factor,t,y,np.nan,np.nan)
    vy1 = np.interp(t1*t_factor,t,vy,np.nan,np.nan)
    ma1 = np.interp(t1*t_factor,t,ma,np.nan,np.nan)
    ld1 = np.interp(t1*t_factor,t,ld,np.nan,np.nan)
    tt_c1 = np.interp(t1*t_factor,t,tt_c,np.nan,np.nan)
    note2 = (r'$t=%.1f\rm\ s$'%(t1*t_factor)+'\n'+
            r'$H=%.2f\rm\ km$'%(y1/1e3)+'\n'+
            r'$v=%.1f\rm m\ s^{-1}$'%(-vy1)+'\n'+
            r'Ma=%.2f'%ma1+'\n'+
            r'Overload=%.2f'%ld1+'\n'+
            r'$T=%.1f\rm^\circ C$'%tt_c1)

    # update
    dot.set_offsets([.2,y1/1e3])
    txt.set_text(note2)

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame,duration=duration)
animation.write_gif('image/vertical_fall/fall.gif',fps=fps)
#'''


''' vertical cannon 

# parameters
x0 = 0 
vx0 = 0 
y0 = 0 # [m]
vy0 = np.logspace(2,9,20) # [m/s]
d = .1 # [m]
rho_s = 7850 # [kg/m^3]
dh = h_max/2e4

# H_est = (.1*vy0**2*(vy0<=300) + 1e3*vy0**.25*(vy0>300)*(vy0<=1e7) + 
#     1*vy0**.6*(vy0>1e7))

# solve the problem 
T, Y, Vy = [], [], []
for i in range(len(vy0)):
    t, x, y, vx, vy = solver(x0,y0,vx0,vy0[i],[d,rho_s,g],dh,'top')
    print('vy0=%.1f. No. of points = %d.'%(vy0[i],len(t)))
    T.append(t)
    Y.append(y)
    Vy.append(vy)

vy_f = np.array([vy[-1] for vy in Vy]) # final vy 
y_f = np.array([y[-1] for y in Y]) # final y 

# max height
Ind = y_f<=h_max
y_m = np.full(y_f.shape,np.nan)
y_m[Ind] = y_f[Ind]
y_m[~Ind] = y_f[~Ind] + vy_f[~Ind]**2/2/g

# motion in vacuum 
y_m_vaccum = vy0**2/(2*g) # [m]
vy0_vaccum = (2*g*y_m)**.5

# data for plotting
note1 = ('ICAO Std. Atm. (1993)\n'+
        r'$\rho_{\rm s}=%.1f\rm\ g\ cm^{-3}$'%(rho_s/1e3)+'\n'+
        r'$d=%s\rm\ m$'%d+'\n'+
        r'$g=%.1f\rm\ m\ s^{-2}$'%g+'\n')
note2 = (r'$v_{y0}=10^{%d}-10^{%d}\rm\ m\ s^{-1}$'%(np.log10(vy0[0]),
    np.log10(vy0[-1]))+'\n')
color = matplotlib.cm.get_cmap('coolwarm')(np.log(vy0/vy0[0])/
    np.log(vy0[-1]/vy0[0]))

# vy vs H ......................................................................

plt.figure(figsize=(6,5))
plt.xscale('log')
plt.yscale('log')
for i in range(len(vy0)):
    plt.plot(Vy[i],Y[i]/1e3,color=color[i])
plt.text(.65,.65,note1+note2,fontsize=12,transform=plt.gca().transAxes)
plt.grid()
plt.xlabel(r'$v_{y0}\rm\ (m\ s^{-1})$')
plt.ylabel('Height (km)')
plt.tight_layout()
plt.savefig('image/vertical_cannon/vy_vs_H_d_%s_rho_s_%s.pdf'%(d,rho_s))
plt.close()

# vy0 vs vy_f ..................................................................

plt.figure(figsize=(6,5))
plt.xscale('log')
plt.scatter(vy0,vy_f,color='k',s=50)
plt.plot(vy0,vy_f,color='k')
plt.text(.65,.65,note1,fontsize=12,transform=plt.gca().transAxes)
plt.grid()
plt.xlabel(r'$v_{y0}\rm\ (m\ s^{-1})$')
plt.ylabel(r'$v_{y,\rm final}\rm\ (m\ s^{-1})$')
plt.tight_layout()
plt.savefig('image/vertical_cannon/vy0_vs_vyf_d_%s_rho_s_%s.pdf'%(d,rho_s))
plt.close()

# vy0 vs max height ............................................................

plt.figure(figsize=(6,5))
plt.xscale('log')
plt.yscale('log')
plt.scatter(vy0,y_m/1e3,color='k',s=10)
plt.plot(vy0,y_m/1e3,color='k')
plt.plot(vy0_vaccum,y_m/1e3,color='r',ls='--',label='No air')
plt.axhline(h_max/1e3,color='k',ls=':',label='Atm. upper bound')
plt.text(.05,.55,note1,fontsize=12,transform=plt.gca().transAxes)
plt.legend()
plt.grid()
plt.xlabel(r'$v_{y0}\rm\ (m\ s^{-1})$')
plt.ylabel(r'$H_{\rm max}\rm\ (km)$')
plt.tight_layout()
plt.savefig('image/vertical_cannon/max_height_d_%s_rho_s_%s.pdf'%(d,rho_s))
plt.close()
#'''






