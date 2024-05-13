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

def learning(a0,t_learn):
    return erf((erfinv(a0)*t_learn+1)/t_learn)


# parameters 
N = 1000000 # number of people 
T = 40 # career span, int
p = 8**-1 # probability of one lucky event per year
a0 = .01 # initial ability
t_learn = 3 # learning time factor in learning curve 

t = np.arange(T) # time
Opp = np.random.rand(N) # probability of being opportunistic per year


# learning event matrix, shape (N,T)
Ln = np.random.binomial(np.ones((T,N),dtype=int),1-Opp).transpose()


# ability matrix 
A = np.zeros((N,T))
A[:,0] = a0
for j in range(1,T):
    A[:,j] = A[:,j-1]
    Ind = np.where(Ln[:,j])[0]
    A[Ind,j] = learning(A[Ind,j],t_learn) 


# lucky event matrix 
Lc = np.random.binomial(1,p,(N,T)) 
Lc1 = Lc*(1-Ln) # noted lucky event matrix 
Lc2 = Lc1*np.random.binomial(1,A) # caught lucky event matrix 
num_Lc2 = np.sum(Lc2,axis=1)

# demo -------------------------------------------------------------------------

text = 'N=%d\np=%.2f\na0=%.2f\nt_learn=%.1f'%(N,p,a0,t_learn)

''' ability vs time 

# colors
cm = plt.get_cmap('inferno')

plt.figure()
plt.xlim(0,T-1)
plt.ylim(0,1)
for i in range(N):  
    plt.plot(t,A[i],color=(1-Opp[i],0,0,.8),lw=1)
plt.grid()
plt.xlabel('Time (year)')
plt.ylabel('Ability')
plt.tight_layout()
plt.savefig('image/ability_vs_time.pdf')
plt.close()
#'''


#''' distribution of num_Lc2 
bins = np.arange(min(num_Lc2)-.5,max(num_Lc2)+1.5)

plt.figure()
plt.hist(num_Lc2,bins,color='purple')
# plt.yscale('log')
plt.text(.75,.7,text,fontsize=14,transform=plt.gca().transAxes)
plt.grid()
plt.xlabel('num_Lc2')
plt.ylabel('# of people')
plt.tight_layout()
plt.savefig('image/distr_%s.pdf'%text)
plt.close()
#'''


''' num_Lc2 vs Opp 
plt.figure()
plt.scatter(Opp,num_Lc2,fc='r',ec='none',s=6)
plt.grid()
plt.xlabel('Degree of opportunism')
plt.ylabel('# of caught lucky events')
plt.tight_layout()
plt.savefig('image/num_Lc2_vs_Opp.pdf')
plt.close()
#'''


''' num_Lc2 vs mean A
plt.figure()
plt.scatter(np.mean(A,axis=1),num_Lc2,fc='r',ec='none',s=6)
plt.grid()
plt.xlabel('Mean ability during career')
plt.ylabel('# of caught lucky events')
plt.tight_layout()
plt.savefig('image/num_Lc2_vs_meanA.pdf')
plt.close()
#'''


#''' distribution of Opp per num_Lc2 
bins = np.linspace(0,1,11) 
cen = (bins[1:]+bins[:-1])/2 

plt.figure(figsize=(5,8)) 
plt.xlim(0,1)
for num in range(int(max(num_Lc2))+1):
    opp = Opp[num_Lc2==num]
    No = np.histogram(opp,bins)[0]
    No = .8*No/max(No)

    plt.step(cen,num+No,where='mid',color='r')
    plt.fill_between(cen,num,num+No,step='mid',alpha=0.5,color='r')
    plt.text(.75,num,'%.4f%%'%(100*len(opp)/N),fontsize=14)

plt.text(.05,.85,text,fontsize=16,transform=plt.gca().transAxes)

plt.grid()
plt.xlabel('Degree of opportunism')
plt.ylabel('# of caught lucky events')
plt.tight_layout()
plt.savefig('image/distr_Opp_%s.pdf'%text)
plt.close()
#'''


#''' distribution of num_Lc2 per Opp

bins_n = np.arange(min(num_Lc2)-.5,max(num_Lc2)+1.5) # bins for num_Lc2
cen_n = (bins_n[1:]+bins_n[:-1])/2

plt.figure(figsize=(5,8))
for i in np.arange(.05,1,.1):
    Ind = (i-.5<=Opp) & (Opp<i+.5) 
    num = num_Lc2[Ind] 

    No = np.histogram(num,bins_n)[0] 
    No = .08*No/max(No) 

    plt.step(cen_n,i+No,where='mid',color='orange')
    plt.fill_between(cen_n,i,i+No,step='mid',alpha=0.5,color='orange')

    # mid num_Lc2
    plt.text(8,i,'%.2f'%np.mean(num),fontsize=14)

plt.text(.05,.85,text,fontsize=16,transform=plt.gca().transAxes)

plt.grid()
plt.xlabel('# of caught lucky events')
plt.ylabel('Degree of opportunism')
plt.tight_layout()
plt.savefig('image/distr_num_Lc2_%s.pdf'%text)
plt.close()
#'''



print('Time = %.1f min.'%((time.time()-t0)/60))








