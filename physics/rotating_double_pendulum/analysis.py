'''
The rotating double-pendulum problem. See knowledge/double_pendulum.md for more 
information.
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tan, arcsin
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


def eq_p2a(x,lamda): 
    # The equation for solving theta2 for 0<=p1<=pi/2 
    w = 4/lamda*tan(x)-sin(x)
    return 4*tan(arcsin(w)) + lamda*sin(x)-8*tan(x)



def eq_p2b(x,lamda): 
    # The equation for solving theta2 for pi/2<p1<=pi 
    w = 4/lamda*tan(x)-sin(x)
    return -4*tan(arcsin(w)) + lamda*sin(x)-8*tan(x)



def solver(P2,lamda):
    # To solve theta1,2 given lamda

    ya = eq_p2a(P2,lamda)
    Inda = np.where(np.sign(ya[:-1])*np.sign(ya[1:])<0)[0]

    yb = eq_p2b(P2,lamda)
    indb = np.where(np.sign(yb[:-1])*np.sign(yb[1:])<0)[0][0]

    # for 0<=p1<=pi/2 
    p2a = [np.nan]*3
    if lamda>lamda_cr4:
        try:
            p2a = [(P2[ind]+P2[ind+1])/2 for ind in Inda[[5,0,2]]]
        except:
            None

    elif lamda>lamda_cr2:
        try:
            p2a = [(P2[ind]+P2[ind+1])/2 for ind in Inda[[3,0]]]+[np.nan]
        except:
            None

    elif lamda>lamda_cr1:
        try:
            p2a = [(P2[ind]+P2[ind+1])/2 for ind in [Inda[2]]]+[np.nan]*2
        except:
            None

    # for pi/2<p1<=pi 
    p2b = np.nan
    if lamda>lamda_cr3:
        try:
            p2b = (P2[indb]+P2[indb+1])/2
        except:
            None

    p2a = p2a
    p1a = [arcsin(4/lamda*tan(i)-sin(i)) for i in p2a]
    p1b = np.pi-arcsin(4/lamda*tan(p2b)-sin(p2b))

    p1 = p1a.copy()
    p2 = p2a.copy()
    p1.insert(2,p1b)
    p2.insert(2,p2b)
    p1 = np.array(p1)
    p2 = np.array(p2)

    return p1,p2


def derivatives(p1,p2,lamda):
    s1 = sin(p1)
    c1 = cos(p1)
    s2 = sin(p2)
    c2 = cos(p2)

    # 1st-order derivatives
    V1 = .5*s1 - lamda/8*(2*s1+s2)*c1
    V2 = .25*s2 - lamda/8*(s1+s2)*c2

    # 2nd-order derivatives
    V11 = .5*c1 - lamda/8*(2*cos(2*p1)-s1*s2)
    V12 = -lamda/8*c1*c2
    V22 = .25*c2 - lamda/8*(cos(2*p2)-s1*s2)

    # Hessian determinant
    D = V11*V22-V12**2 

    return V1, V2, V11, V12, V22, D


def stability_analysis(p1,p2,lamda):

    V1, V2, V11, V12, V22, D = derivatives(p1,p2,lamda)

    Min_loc = (D>0)*(V11>0)
    Max_loc = (D>0)*(V11<0)
    Saddle = D<0 

    Unstable = Max_loc+Saddle
    Unsure = D==0. 
    Stable = np.where(Min_loc,1,np.where(Unsure,0,np.where(Unstable,-1,np.nan)))

    return Stable 



# constants
lamda_cr1 = 6-2*5**.5 
lamda_cr2 = 2*5**.5-2
lamda_cr3 = 2*5**.5+2
lamda_cr4 = 6+2*5**.5 
p1A = np.pi/6 # some points on curve 4
p2A = -np.pi/2
p1B = .546
p2B = -1.317 
lamda_B = 34.4
Name_s = [r'$0$-$0_1$',r'$0$-$\pi$',r'$\pi$-$0$',r'$0$-$0_2$']
Name_s1 = [i+' branch' for i in Name_s]
Name_st = [r'$0$-$0$',r'$0$-$\pi$',r'$\pi$-$0$',r'$\pi$-$\pi$']


# parameters
Lamda = np.logspace(0,np.log10(150),1000)
P2_series = np.linspace(-np.pi+1e-5,np.pi-1e-5,100000) 
    # N=1e5 for quick results, 1e6 for better results


# solve the problem (nontrivial solutions)
P1 = []
P2 = []
for lamda in Lamda:
    p1,p2 = solver(P2_series,lamda)
    P1.append(p1)
    P2.append(p2)
P1 = np.transpose(P1) # shape (4,n)
P2 = np.transpose(P2)


# trivial solutions 
P1t = np.transpose([[0,0,np.pi,np.pi]]*len(Lamda))
P2t = np.transpose([[0,np.pi,0,np.pi]]*len(Lamda))


# dimensionless potential energy V/(mgl)
V=-.25*(2*cos(P1)+cos(P2))-Lamda/16*(2*sin(P1)**2+sin(P2)**2+2*sin(P1)*sin(P2))
Vt = (-.25*(2*cos(P1t)+cos(P2t))-
    Lamda/16*(2*sin(P1t)**2+sin(P2t)**2+2*sin(P1t)*sin(P2t)) )

# derivatives
V1, V2, V11, V12, V22, D = derivatives(P1,P2,Lamda)
V1t, V2t, V11t, V12t, V22t, Dt = derivatives(P1t,P2t,Lamda)

# stability analysis 
Stable = stability_analysis(P1,P2,Lamda)
Stable_t = stability_analysis(P1t,P2t,Lamda)


# symmetrical solutions for lamda<0 
P1n = np.full(P1.shape,np.nan)
P2n = P1n.copy()

P1n = np.pi-P1
P2n[0] = np.pi-P2[0]
P2n[[1,2,3]] = -np.pi-P2[[1,2,3]]


# demo =========================================================================

# global parameters 
tick_h = np.pi*np.arange(0,1.1,.5) # half
ticklabel_h = ['0',r'$\frac{\pi}{2}$',r'$\pi$']
tick = np.pi*np.arange(-1,1.1,.5)
ticklabel = [r'$-\pi$',r'$-\frac{\pi}{2}$','0',r'$\frac{\pi}{2}$',r'$\pi$']


''' angle vs lamda
plt.figure(figsize=(8.5,6))
plt.xscale('log')
plt.xlim(.5,100)

# nontrivial solutions
for i in range(4):
    plt.plot(Lamda,P1[i],color='C%d'%i,label=Name_s1[i])
    plt.plot(Lamda,P2[i],color='C%d'%i,ls='--')
plt.plot(0,0,color='k',label=r'$\theta_1$')
plt.plot(0,0,color='k',ls='--',label=r'$\theta_2$')

# trivial solutions 
plt.axhline(0,color='C4',label='Trivial '+r'$\theta_{1,2}$')
plt.axhline(np.pi,color='C4')

plt.axvline(lamda_cr1,color='k',ls=':')
plt.axvline(lamda_cr2,color='k',ls=':')
plt.axvline(lamda_cr3,color='k',ls=':')
plt.axvline(lamda_cr4,color='k',ls=':')
plt.text(lamda_cr1,-3.3,r'$\lambda_1$',fontsize=18)
plt.text(lamda_cr2,-3.3,r'$\lambda_2$',fontsize=18)
plt.text(lamda_cr3,-3.3,r'$\lambda_3$',fontsize=18)
plt.text(lamda_cr4,-3.3,r'$\lambda_4$',fontsize=18)

# some points 
plt.scatter(lamda_B,p1B,fc='w',ec='C3',s=20,zorder=40)
plt.text(lamda_B,p1B+.1,r'$B(34.4,31.3^\circ)$',fontsize=14)

plt.grid()
plt.legend(fontsize=14,loc='upper left')
ax = plt.gca()
ax.set_yticks(tick)
ax.set_yticklabels(ticklabel)
plt.xlabel(r'$\lambda=\omega^2l/g$')
plt.ylabel('Angles')
plt.tight_layout()
plt.savefig('image/theta_vs_lamda.pdf')
plt.close()
#'''


''' potential energy vs lamda
plt.figure(figsize=(8.5,6))
plt.xscale('log')
plt.ylim(-6,1)

# nontrivial solutions
for i in range(4):
    plt.plot(Lamda,V[i],color='C%d'%i,label=Name_s1[i])
    plt.plot(Lamda,Vt[i],color='C4')

# trivial solutions 
plt.plot(0,0,color='C4',label='Trivial solutions')

plt.axvline(lamda_cr1,color='k',ls=':')
plt.axvline(lamda_cr2,color='k',ls=':')
plt.axvline(lamda_cr3,color='k',ls=':')
plt.axvline(lamda_cr4,color='k',ls=':')
plt.text(lamda_cr1,-3.3,r'$\lambda_1$',fontsize=18)
plt.text(lamda_cr2,-3.3,r'$\lambda_2$',fontsize=18)
plt.text(lamda_cr3,-3.3,r'$\lambda_3$',fontsize=18)
plt.text(lamda_cr4,-3.3,r'$\lambda_4$',fontsize=18)

plt.grid()
plt.legend(fontsize=14)
plt.xlabel(r'$\lambda=\omega^2l/g$')
plt.ylabel(r'$\hat{V}=V/(mgl)$')
plt.tight_layout()
plt.savefig('image/energy_vs_lamda.pdf')
plt.close()
#'''


''' stability analysis 

# parameters 
Y = [-.3,-1.3,-2.3,.3]
Yt = [0,-1,-2,-3]

plt.figure(figsize=(8,5))
plt.xscale('log')
plt.xlim(Lamda.min()/10,Lamda.max()*3)

for i in range(len(Y)):

    # colors
    c = np.array(['white']*len(Lamda))
    c[Stable[i]==1] = 'C2'
    c[Stable[i]==0] = 'C1'
    c[Stable[i]==-1] = 'C3'

    c_t = np.array(['white']*len(Lamda))
    c_t[Stable_t[i]==1] = 'C2'
    c_t[Stable_t[i]==0] = 'C1'
    c_t[Stable_t[i]==-1] = 'C3'

    plt.scatter(Lamda,[Y[i]]*len(Lamda),c=c,s=.3)
    plt.scatter(Lamda,[Yt[i]]*len(Lamda),c=c_t,s=.3)
    plt.text(Lamda.max(),Y[i],Name_s1[i])
    plt.text(Lamda.max(),Yt[i],Name_st[i])

    plt.axvline(lamda_cr1,color='k',ls=':',lw=1)
    plt.axvline(lamda_cr2,color='k',ls=':',lw=1)
    plt.axvline(lamda_cr3,color='k',ls=':',lw=1)
    plt.axvline(lamda_cr4,color='k',ls=':',lw=1)
    plt.text(lamda_cr1,-2.8,r'$\lambda_1$',fontsize=14)
    plt.text(lamda_cr2,-2.8,r'$\lambda_2$',fontsize=14)
    plt.text(lamda_cr3,-2.8,r'$\lambda_3$',fontsize=14)
    plt.text(lamda_cr4,-2.8,r'$\lambda_4$',fontsize=14)

# legend
plt.plot(0,0,color='C2',label='Linearly stable')
plt.plot(0,0,color='C3',label='Linearly unstable')
plt.plot(0,0,color='C1',label='Unsure')

plt.legend(loc='upper left')
plt.xlabel(r'$\lambda=\omega^2l/g$')
plt.yticks([])
plt.tight_layout()
plt.savefig('image/stability.pdf')
plt.close()
#'''


''' theta1 vs theta2
plt.figure(figsize=(3.7,6))
plt.axis('equal')

# nontrivial solutions
for i in range(4):
    plt.plot(P1[i],P2[i],color='C%d'%i)

# trivial solutions
plt.scatter([0,np.pi,np.pi,0],[0,0,np.pi,np.pi],s=20,fc='C4',ec='none',
    zorder=40)

# curve order 
plt.text(.4,.8,Name_s[0],fontsize=16,color='C0')
plt.text(.6,-2.6,Name_s[1],fontsize=16,color='C1')
plt.text(2.3,-.8,Name_s[2],fontsize=16,color='C2')
plt.text(.6,-.9,Name_s[3],fontsize=16,color='C3')

# some points 
plt.scatter(p1B,p2B,fc='w',ec='C3',s=20,zorder=40)
plt.text(p1B+.1,p2B,r'$B(31.3^\circ,-75.5^\circ)$',fontsize=12)
plt.scatter(p1A,p2A,fc='w',ec='C3',s=20,zorder=40)
plt.text(p1A+.1,p2A,r'$A(30^\circ,-90^\circ)$',fontsize=12)

plt.grid()
ax = plt.gca()
ax.set_xticks(tick_h)
ax.set_xticklabels(ticklabel_h)
ax.set_yticks(tick)
ax.set_yticklabels(ticklabel)
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.tight_layout()
plt.savefig('image/theta1_vs_theta2.pdf')
plt.close()
#'''


''' theta1 vs theta2 (extended)
plt.figure(figsize=(6.5,6))
plt.axis('equal')

# nontrivial solutions
for i in range(4):
    plt.plot(P1[i],P2[i],color='C%d'%i)
    plt.plot(P1n[i],P2n[i],color='C%d'%i,ls='--')
    plt.plot(-P1[i],-P2[i],color='C%d'%i)
    plt.plot(-P1n[i],-P2n[i],color='C%d'%i,ls='--')
plt.plot(0,0,color='k',label=r'$\lambda>0$')
plt.plot(0,0,color='k',ls='--',label=r'$\lambda<0$')

# trivial solutions
x = np.linspace(-np.pi,np.pi,3)
y = x.copy()
X, Y = np.meshgrid(x,y)
plt.scatter(X,Y,s=20,fc='C4',ec='none',label='Trivial',zorder=5)

plt.legend()
plt.grid()
ax = plt.gca()
ax.set_xticks(tick)
ax.set_xticklabels(ticklabel)
ax.set_yticks(tick)
ax.set_yticklabels(ticklabel)
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')
plt.tight_layout()
plt.savefig('image/theta1_vs_theta2_ext.pdf')
plt.close()
#'''


''' interactive pendulum configurations

# parameters
lamda = 12.5
xmin, xmax = -.7, 1.3
ymin, ymax = -1.2, .6


# solution for initial plot
p1,p2 = solver(P2_series,lamda)
x1 = .5*sin(p1)
x2 = .5*(sin(p1)+sin(p2))
y1 = -.5*cos(p1)
y2 = .5*(-cos(p1)-cos(p2))


# figure parameters
num2in = 1.5 # value-to-inch conversion factor 
g_p = 0
rl_p = .2
b_p = .2
t_p = .5
h_s = .2
rl_s = 1.2
b_s = .4

# derived figure parameters
x_p = num2in*(xmax-xmin)
y_p = num2in*(ymax-ymin)

fig_x = 4*x_p+3*g_p+2*rl_p
fig_y = t_p+y_p+b_p+h_s+b_s 

fx_p1 = rl_p/fig_x 
fx_p2 = (rl_p+x_p+g_p)/fig_x 
fx_p3 = (rl_p+2*x_p+2*g_p)/fig_x 
fx_p4 = (rl_p+3*x_p+3*g_p)/fig_x 

fy_p = (b_p+h_s+b_s)/fig_y 
fw_p = x_p/fig_x
fh_p = y_p/fig_y

fx_s = rl_s/fig_x 
fy_s = b_s/fig_y
fw_s = 1-2*rl_s/fig_x 
fh_s = h_s/fig_y


fig = plt.figure(figsize=(fig_x,fig_y))

ax1 = plt.axes([fx_p1,fy_p,fw_p,fh_p])
ax1.set_title(r'$0$'+'-'+'$0_1$'+' branch')
ax1.axis('equal')
ax1.axis('off')
ax1.set_xlim(xmin,xmax)
ax1.set_ylim(ymin,ymax)
ax1.axvline(0,color='k',ls=':')
ax1.scatter(0,0,ec='k',fc='w',s=20,zorder=40)
l1,=ax1.plot([0,x1[0],x2[0]],[0,y1[0],y2[0]],color='C0')
dot1=ax1.scatter([x1[0],x2[0]],[y1[0],y2[0]],ec='C0',fc='w',s=20,zorder=40)

ax2 = plt.axes([fx_p2,fy_p,fw_p,fh_p])
ax2.set_title(r'$0$'+'-'+'$\pi$'+' branch')
ax2.axis('equal')
ax2.axis('off')
ax2.set_xlim(xmin,xmax)
ax2.set_ylim(ymin,ymax)
ax2.axvline(0,color='k',ls=':')
ax2.scatter(0,0,ec='k',fc='w',s=20,zorder=40)
l2,=ax2.plot([0,x1[1],x2[1]],[0,y1[1],y2[1]],color='C1')
dot2=ax2.scatter([x1[1],x2[1]],[y1[1],y2[1]],ec='C1',fc='w',s=20,zorder=40)

ax3 = plt.axes([fx_p3,fy_p,fw_p,fh_p])
ax3.set_title(r'$\pi$'+'-'+'$0$'+' branch')
ax3.axis('equal')
ax3.axis('off')
ax3.set_xlim(xmin,xmax)
ax3.set_ylim(ymin,ymax)
ax3.axvline(0,color='k',ls=':')
ax3.scatter(0,0,ec='k',fc='w',s=20,zorder=40)
l3,=ax3.plot([0,x1[2],x2[2]],[0,y1[2],y2[2]],color='C2')
dot3=ax3.scatter([x1[2],x2[2]],[y1[2],y2[2]],ec='C2',fc='w',s=20,zorder=40)

ax4 = plt.axes([fx_p4,fy_p,fw_p,fh_p])
ax4.set_title(r'$0$'+'-'+'$0_2$'+' branch')
ax4.axis('equal')
ax4.axis('off')
ax4.set_xlim(xmin,xmax)
ax4.set_ylim(ymin,ymax)
ax4.axvline(0,color='k',ls=':')
ax4.scatter(0,0,ec='k',fc='w',s=20,zorder=40)
l4,=ax4.plot([0,x1[3],x2[3]],[0,y1[3],y2[3]],color='C3')
dot4=ax4.scatter([x1[3],x2[3]],[y1[3],y2[3]],ec='C3',fc='w',s=20,zorder=40)


# lamda slider
ax_s = plt.axes([fx_s,fy_s,fw_s,fh_s])
sld = Slider(ax=ax_s,label=r'$\lambda=\omega^2l/g$',
    valmin=0,valmax=20,valinit=lamda,orientation='horizontal')
sld.vline._linewidth = 0

tick_s = [lamda_cr1,lamda_cr2,lamda_cr3,lamda_cr4]
ticklabel_s = [r'$\lambda_1$',r'$\lambda_2$',r'$\lambda_3$',r'$\lambda_4$']
sld.ax.xaxis.set_ticks(tick_s)
sld.ax.xaxis.set_ticklabels(ticklabel_s)


# The function to be called anytime a slider's value changes
def update(val,ax1=ax1,ax2=ax2,ax3=ax3,ax4=ax4):

    # new slider values
    lamda = sld.val 

    # new solutions
    p1,p2 = solver(P2_series,lamda)
    x1 = .5*sin(p1)
    x2 = .5*(sin(p1)+sin(p2))
    y1 = -.5*cos(p1)
    y2 = .5*(-cos(p1)-cos(p2))

    # upload plots
    l1.set_xdata([0,x1[0],x2[0]])
    l1.set_ydata([0,y1[0],y2[0]])
    dot1.set_offsets([[x1[0],y1[0]],[x2[0],y2[0]]])

    l2.set_xdata([0,x1[1],x2[1]])
    l2.set_ydata([0,y1[1],y2[1]])
    dot2.set_offsets([[x1[1],y1[1]],[x2[1],y2[1]]])

    l3.set_xdata([0,x1[2],x2[2]])
    l3.set_ydata([0,y1[2],y2[2]])
    dot3.set_offsets([[x1[2],y1[2]],[x2[2],y2[2]]])

    l4.set_xdata([0,x1[3],x2[3]])
    l4.set_ydata([0,y1[3],y2[3]])
    dot4.set_offsets([[x1[3],y1[3]],[x2[3],y2[3]]])

    fig.canvas.draw_idle()

# register the update function with each slider
sld.on_changed(update)
plt.savefig('image/interactive_pendulum.pdf')
plt.show()
#'''

