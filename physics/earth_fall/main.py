"""
To dig a hole through the earth and jump in.
"""
import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.append('/Users/yuecao/Documents/coding/module')
import phunction

# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13

# parameters
r_earth = 6371e3  # [m]
N = 1000  # number of sampling

d = np.linspace(0, r_earth, N)  # depth, [m]
r = r_earth - d
g = phunction.r2g_earth(r/1e3)  # [m/s^2]

v = np.full(N, np.nan)
v[0] = 0  # initial velocity
t = np.full(N, np.nan)  # time passed
t[0] = 0
for i in range(N-1):
    v[i+1] = (v[i]**2 + 2*g[i]*(d[i+1]-d[i])) **.5
    t[i+1] = (v[i+1]-v[i]) / g[i] + t[i]

print(f'Needs {t[-1]/60:.2f} min to reach the earth center.')
print(f'Final v is {v[-1]/1e3:.2f} km/s.')

# v vs t
plt.figure()
plt.plot(t/60, v/1e3, color='k')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.grid()
plt.xlabel(r'$t\rm\ (min)$')
plt.ylabel(r'$v\rm\ (km\ s^{-1})$')
plt.tight_layout()
plt.savefig('v_vs_t.pdf')
plt.close()

# depth vs t
plt.figure()
plt.plot(t/60, d/1e3, color='k')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.grid()
plt.xlabel(r'$t\rm\ (min)$')
plt.ylabel('Depth '+r'$\rm (km)$')
plt.tight_layout()
plt.savefig('depth_vs_t.pdf')
plt.close()

