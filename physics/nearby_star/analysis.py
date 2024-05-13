import dill
import sys
import colour
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patheffects as path_effects
from scipy.optimize import curve_fit

sys.path.append('../module')
import phunction 


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def fun_Teff(BR,ct_T,ct_alpha,ct_br):
    # the Teff-BR relation
    return ct_T*(BR-ct_br)**ct_alpha


# read Gaia dr3 catalog (sorted by parallax) ...................................

f = open('data/gaia_dr3.csv','r')
f.readline()
data = f.readlines()
f.close()

Id,Ra,Dec,Parallax,Mg,BR,Teff,Logg,Fe,Ag = [],[],[],[],[],[],[],[],[],[]
    # with GSP-Phot params
Id0,Ra0,Dec0,Parallax0,Mg0,BR0 = [],[],[],[],[],[] 
    # without GSP-Phot params

for line in data:
    line = line.split(',')
    if not line[6]=='':
        line = [float(i) for i in line]
        Id.append(line[0])
        Ra.append(line[1])
        Dec.append(line[2])
        Parallax.append(line[3])
        Mg.append(line[4])
        BR.append(line[5])
        Teff.append(line[6])
        Logg.append(line[7])
        Fe.append(line[8])
        Ag.append(line[9])

    else:
        Id0.append(float(line[0]))
        Ra0.append(float(line[1]))
        Dec0.append(float(line[2]))
        Parallax0.append(float(line[3]))
        Mg0.append(np.nan if line[4]=='' else float(line[4]))
        BR0.append(np.nan if line[5]=='' else float(line[5]))

Id,Ra,Dec,Parallax,Mg,BR,Teff,Logg,Fe,Ag = [np.array(i) for i in 
    [Id,Ra,Dec,Parallax,Mg,BR,Teff,Logg,Fe,Ag]]
Id0,Ra0,Dec0,Parallax0,Mg0,BR0 = [np.array(i) for i in 
    [Id0,Ra0,Dec0,Parallax0,Mg0,BR0]]

print('%d/%d (%.1f%%) sources have GSP-Phot parameters.'%(len(Id),len(data),
    100*len(Id)/len(data)))

# calculations .................................................................

# distance
Dist = 1e3/Parallax # [pc]
Dist0 = 1e3/Parallax0

# Equatorial XYZ [pc]
X = Dist*np.cos(Dec*np.pi/180)*np.cos(Ra*np.pi/180)
Y = Dist*np.cos(Dec*np.pi/180)*np.sin(Ra*np.pi/180)
Z = Dist*np.sin(Dec*np.pi/180)
X0 = Dist0*np.cos(Dec0*np.pi/180)*np.cos(Ra0*np.pi/180)
Y0 = Dist0*np.cos(Dec0*np.pi/180)*np.sin(Ra0*np.pi/180)
Z0 = Dist0*np.sin(Dec0*np.pi/180)

# Teff0 
p0 = [7000,-.8,-.5]
res = curve_fit(fun_Teff,BR,Teff,p0)
ct_T,ct_alpha,ct_br = res[0]
Teff0 = fun_Teff(BR0,ct_T,ct_alpha,ct_br)
Teff0[Teff0>3e4] = np.nan

# Luminosity
MMg = Mg - 5*(np.log10(Dist)-1) # abs mag 
MMg0 = Mg0 - 5*(np.log10(Dist0)-1)
L = ((Teff/5800)**4 * 10**(.4*(4.83-MMg)) * (np.exp(22135/Teff)-1) /
    (np.exp(22135/5800)-1)) # [L_sun]
L0 = ((Teff0/5800)**4 * 10**(.4*(4.83-MMg0)) * (np.exp(22135/Teff0)-1) /
    (np.exp(22135/5800)-1)) 

# Radius 
R = (Teff/5800)**-2*L**.5 # [R_sun]
R0 = (Teff0/5800)**-2*L0**.5

# colors
RGB = phunction.T2color(Teff)
RGB0 = phunction.T2color(Teff0)

# demo -------------------------------------------------------------------------

''' x vs y
plt.figure(figsize=(6,6))
plt.axis('equal')
plt.scatter(X,Y,s=4,label='With GSP-Phot params')
plt.scatter(X0,Y0,s=4,label='Without GSP-Phot params')
plt.legend()
plt.xlabel('Equatorial x (pc)')
plt.ylabel('Equatorial y (pc)')
plt.grid()
plt.tight_layout()
plt.savefig('image/scatters/x_y.pdf')
plt.close()
#'''


''' Teff vs Logg
plt.figure(figsize=(5,4))
plt.scatter(Teff,Logg,s=4,label='With GSP-Phot params')
plt.legend()
plt.xlabel(r'$T_{\rm eff}\rm\ (K)$')
plt.ylabel(r'${\rm Log}g\rm\ (cm\ s^{-2})$')
plt.grid()
plt.tight_layout()
plt.savefig('image/scatters/teff_logg.pdf')
plt.close()
#'''


''' BR vs Teff 

# parameters
ct_T0 = 7000
ct_alpha0 = -1
ct_br0 = -.35
x = np.linspace(-.4,5.5,100)

y0 = fun_Teff(x,ct_T0,ct_alpha0,ct_br0)
y = fun_Teff(x,ct_T,ct_alpha,ct_br)

plt.figure(figsize=(5,4))
# plt.xscale('log')
plt.yscale('log')
plt.scatter(BR,Teff,s=2,label='With GSP-Phot params')
plt.plot(x,y0,color='k',label='Eye fitting')
plt.plot(x,y,color='r',label='Fitting')
plt.legend()
plt.xlabel(r'$B-R$')
plt.ylabel(r'$T_{\rm eff}\rm\ (K)$')
plt.grid()
plt.tight_layout()
plt.savefig('image/scatters/br_teff.pdf')
plt.show()
#'''


''' Teff vs L
plt.figure(figsize=(5,4))
plt.yscale('log')
plt.scatter(Teff,L,s=4,alpha=.5,label='With GSP-Phot params')
plt.scatter(Teff0,L0,s=4,alpha=.5,label='Without GSP-Phot params')
plt.legend()
plt.xlabel(r'$T_{\rm eff}\rm\ (K)$')
plt.ylabel(r'$L\ (L_\odot)$')
plt.grid()
plt.tight_layout()
plt.savefig('image/scatters/teff_L.pdf')
plt.close()
#'''


''' distributions of params

# parameters 
Labels = ['With GSP-Phot params','Without GSP-Phot params']
Data = {
    'Dist': {
        'data': [Dist,Dist0],
        'xlabel': 'Distance (pc)',
    },
    'Mg': {
        'data': [Mg,Mg0],
        'xlabel': r'$m_G\rm\ (mag)$',
    },
    'BR': {
        'data': [BR,BR0],
        'xlabel': 'B-R color index',
    },
    'L': {
        'data': [L,L0],
        'xlabel': r'$L\ (L_\odot)$',
    },
    'MMg': {
        'data': [MMg,MMg0],
        'xlabel': r'$M_G\rm\ (mag)$',
    },
    'R': {
        'data': [R,R0],
        'xlabel': r'$R\ (R_\odot)$',
    },
    'Teff': {
        'data': [Teff,Teff0],
        'xlabel': r'$T_{\rm eff}\rm\ (K)$',
    },
    'Logg': {
        'data': [Logg],
        'xlabel': r'${\rm Log}g\rm\ (cm\ s^{-2})$',
    },
    'Fe': {
        'data': [Fe],
        'xlabel': '[Fe/H]',
    },
    'Ag': {
        'data': [Ag],
        'xlabel': 'Extinction '+r'$A_G$',
    },    
}

for name in Data:
    data = Data[name]['data']
    xlabel = Data[name]['xlabel']

    plt.figure(figsize=(5,4))
    for i in range(len(data)):
        plt.hist(data[i],bins=int(len(data[i])**.5),label=Labels[i],alpha=.6)
    plt.legend()
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('Numbers')
    plt.tight_layout()
    plt.savefig('image/hist/%s.pdf'%name)
    plt.close()
#'''


''' star diagram 

# parameters
a_R = .1


fig = plt.figure(figsize=(6,6))
ax = plt.gca()
ax = fig.add_axes([0,0,1,1])
plt.xlim(-20,20)
plt.ylim(-20,20)

# dark background
c = 'k'
fig.patch.set_facecolor(c)
ax.set_facecolor(c)
ax.spines['bottom'].set_color('w') 
ax.spines['top'].set_color('w') 
ax.spines['right'].set_color('w')
ax.spines['left'].set_color('w')
ax.tick_params(axis='x',colors='w') 
ax.tick_params(axis='y',colors='w')
ax.yaxis.label.set_color('w')
ax.xaxis.label.set_color('w')

# scale circles
for r in [5,10,15,20]:
    ax.add_artist(plt.Circle((0,0),r,fc='none',ec='w',lw=.5,ls=':',zorder=1))
    plt.text(r,.1,'%d pc'%r,color='w',fontsize=10)
plt.axvline(0,color='w',lw=.5,ls=':',zorder=1)
plt.axhline(0,color='w',lw=.5,ls=':',zorder=1)

# the Sun 
ax.add_artist(plt.Circle((0,0),a_R,fc=phunction.T2color([5800])[0],ec='none'))
plt.text(.1,.1,'Sun',color='w',fontsize=10)

# stars
for i in range(len(R)):
    ax.add_artist(plt.Circle((X[i],Y[i]),a_R*R[i],fc=RGB[i],ec='none'))
# plt.savefig('image/star_diagram_GSP.pdf')
for i in range(len(R0)):
    ax.add_artist(plt.Circle((X0[i],Y0[i]),a_R*R0[i],fc=RGB0[i],ec='none'))

plt.savefig('image/star_diagram.pdf')
plt.close()
#'''










