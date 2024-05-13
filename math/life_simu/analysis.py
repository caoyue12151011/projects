import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


np.random.seed(0)
t0 = time.time()


# parameters 
N = 100000 # number of people 
T = 40 # int, career span
p = 8**-1 # probability of one lucky event per year 


t = np.arange(T) # time
# A = np.random.rand(N) # abilities
A = np.random.normal(.6,.1,N)
A[A>1] = 1
A[A<0] = 0



Lc = np.random.binomial(1,p,(N,T)) # lucky event matrix 
Catch = np.random.binomial(np.ones((T,N),dtype=int),A).transpose() 
Suc = Lc*Catch # success event matrix
num_suc = np.sum(Suc,axis=1)

# demo -------------------------------------------------------------------------

text = 'N=%d\nT=%d\np=%.2f'%(N,T,p)


#''' distribution of num_suc 
bins = np.arange(min(num_suc)-.5,max(num_suc)+1.5)

plt.figure()
plt.hist(num_suc,bins,color='purple')
plt.text(.75,.7,text,fontsize=14,transform=plt.gca().transAxes)
plt.grid()
plt.xlabel('num_suc')
plt.ylabel('# of people')
plt.tight_layout()
plt.savefig('image/distr_num_suc_%s.pdf'%text)
plt.close()
#'''


#''' A vs num_suc  
plt.figure()
plt.scatter(A,num_suc,fc='r',ec='none',s=6)
plt.grid()
plt.xlabel('Ability')
plt.ylabel('num_suc')
plt.tight_layout()
plt.savefig('image/num_suc_vs_A.pdf')
plt.close()
#'''



print('Time = %.1f min.'%((time.time()-t0)/60))








