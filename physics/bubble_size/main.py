"""
To calculate soap bubble size at different altitude.
"""
import sys
import numpy as np 
import matplotlib.pyplot as plt 

sys.path.append('../../../module')
import phunction
import math_tk

# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


# parameters
h = np.linspace(-5, 80, 1000)  # [km]
eta = np.array([0, 1e-3, 1e-2, .1, 1, 10, 1e2])  
    # 4*gamma/(p0*r0)

p = phunction.h2property_atm(h, 'p') / phunction.h2property_atm(0, 'p')
T = ((phunction.h2property_atm(h, 'T') + 273.15) / 
     (phunction.h2property_atm(0, 'T') + 273.15))

# solve the cubic equation
eta1 = eta[..., np.newaxis]  # (eta, h)
b = eta1 / p 
a = np.ones_like(b)
c = np.zeros_like(b)
d = -(1+eta1)*T/p 
res = math_tk.cubic_solver(a, b, c, d)

ind = 0
if not np.all(res[ind].real > 0):
    print(f'[Error] not all values in res[{ind}] > 0.')
print(f'[Info] max(res[{ind}].imag) = {np.max(res[ind].imag):.2e}.')
r = abs(res[ind])

# solution when eta = inf 
eta = np.concatenate((eta, [np.inf]))
r = np.concatenate((r, T[np.newaxis]**.5))

# color map 
colors = plt.get_cmap('rainbow')(np.arange(len(eta))/(len(eta)-2))
colors[-1] = np.array([0,0,0,1])

plt.figure()
plt.yscale('log')
for i in range(len(eta)):
    plt.plot(h, r[i], color=colors[i], lw=2, label=f'$\eta={eta[i]}$')
plt.legend()
plt.grid()
plt.xlabel('Height (km)')
plt.ylabel(r'$r/r_0$')
plt.tight_layout()
plt.savefig('bubble_size.pdf')
plt.close()