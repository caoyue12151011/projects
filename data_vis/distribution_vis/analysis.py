'''
To visualize different number distributions with circles of different sizes.
'''

import sys
import time
import matplotlib   
import numpy as np 
import matplotlib.pyplot as plt 
from importlib import reload
from scipy.stats import loguniform, lognorm, pareto

sys.path.append('../module')
import imf
import phunction
import circ_pack 
reload(imf)
reload(phunction)
reload(circ_pack)


t = time.time()


# demo -------------------------------------------------------------------------

''' lognormal distribution

# parameters
N = 1000 # number of circles
sigma = 1 # ln(standard deviation)
mu = 0 # ln(mean)
eta = 1.5 
f = .5
figx = 8 # [in]


# cirle packing
S = lognorm.rvs(s=sigma,scale=np.exp(mu),size=N)
R = (S/np.pi)**.5
Xc, Yc, w, h = circ_pack.circ_pack(S,eta,f)


# demo
plt.figure(figsize=(figx,figx/eta))
plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
plt.axis('equal')
plt.xlim(0,w)
plt.ylim(0,h)
ax = plt.gca()

lnS = np.log(S)
Rgba = matplotlib.cm.get_cmap('Spectral')((lnS-lnS.min())/(lnS.max()-lnS.min()))
for i in range(len(S)):
    circle = plt.Circle((Xc[i],Yc[i]),R[i],facecolor=Rgba[i],edgecolor='none',
        alpha=.7)
    ax.add_artist(circle)

text = 'Lognormal\n'+r'$N=%s$'%N+'\n'+r'$\mu=%s$'%mu+'\n'+r'$\sigma=%s$'%sigma
plt.text(.05,.8,text,transform=ax.transAxes,fontsize=14,
    bbox=dict(facecolor='w',edgecolor='k',alpha=.5))

plt.xticks([])
plt.yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.savefig('image/lognorm_N%s_mu%s_sigma%s.pdf'%(N,mu,sigma))
plt.close()
#'''



''' Pareto distribution (truncated powerlaw)

# parameters
N = 10000 # number of circles
p = 4.8 # powerlaw index, f(x)~x^-p
eta = 1.5 
f = .3
figx = 8 # [in]


# cirle packing
S = pareto.rvs(p-1,size=N)
R = (S/np.pi)**.5
Xc, Yc, w, h = circ_pack.circ_pack(S,eta,f)
print('Circle packing finished')


# demo
plt.figure(figsize=(figx,figx/eta))
plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
plt.axis('equal')
plt.xlim(0,w)
plt.ylim(0,h)
ax = plt.gca()

lnS = np.log(S)
Rgba = matplotlib.cm.get_cmap('Spectral')((lnS-lnS.min())/(lnS.max()-lnS.min()))
for i in range(len(S)):
    circle = plt.Circle((Xc[i],Yc[i]),R[i],facecolor=Rgba[i],edgecolor='none',
        alpha=.8)
    ax.add_artist(circle)

text = 'Powerlaw\n'+r'$N=%s$'%N+'\n'+r'$p=%s$'%p
plt.text(.05,.8,text,transform=ax.transAxes,fontsize=14,
    bbox=dict(facecolor='w',edgecolor='k',alpha=.5))

plt.xticks([])
plt.yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_facecolor('k')
plt.savefig('image/powerlaw_N%s_p%s_k.pdf'%(N,p))
plt.close()
#'''




t = time.time() - t
print('Time: %.1f min.'%(t/60))





