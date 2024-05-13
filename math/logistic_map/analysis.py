import itertools 
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


''' bifurcation diagram

# parameters 
r = np.linspace(3,4,100000)
n = 10000 # number of iterations
X = np.random.rand(len(r))

for i in range(n):
    X *= r*(1-X)

plt.figure(figsize=(11,5))
plt.xlim(r[0],r[-1])
plt.scatter(r,X,fc='k',ec='none',s=.2)
plt.grid()
plt.xlabel(r'$r$')
plt.ylabel(r'$x_n\ (n=%d)$'%n)
plt.tight_layout()
plt.savefig('bifurcation.pdf')
plt.show()
#'''


#''' hist of x_n

# parameters
x0 = .5 
n = 1000000
r = 3.6

X = np.zeros(n) 
X[0] = x0
for i in range(n-1):
    X[i+1] = r*X[i]*(1-X[i])


plt.figure(figsize=(11,5))
plt.hist(X,int(n**.5),fc='none',ec='k')
plt.grid()
plt.xlabel(r'$x_n\ (n=%d)$'%n)
plt.ylabel('Numbers')
plt.tight_layout()
plt.savefig('hist_xn.pdf')
plt.show()
#'''










