import numpy as np
import matplotlib.pyplot as plt


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


#''' 散光
# parameters 
d_theta = 10 # [deg]

Theta = np.arange(0,180,d_theta)*np.pi/180 
X0 = -np.cos(Theta)
X1 = -X0 
Y0 = -np.sin(Theta)
Y1 = -Y0

X = np.array([X0,X1])
Y = np.array([Y0,Y1])

plt.figure(figsize=(7,7)) 
plt.plot(X,Y,color='k',lw=1)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('SanGuang.pdf')
plt.show()
#'''







