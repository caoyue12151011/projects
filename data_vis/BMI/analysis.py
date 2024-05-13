import numpy as np 
import matplotlib.pyplot as plt 


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


# parameters
BMI_cr = np.array([18.5,24,28])
alpha = .3


for h in [1.6,1.83,1.71]: # [m] height
     
    w_cr = BMI_cr*h**2 # [kg]
    if h in [1.6,1.71]:
        w_cr *= 2
   
    plt.figure(figsize=(6,2))
    if h in [1.6,1.71]:
        plt.xlim(w_cr[0]-30,w_cr[-1]+30)
    else:
        plt.xlim(w_cr[0]-15,w_cr[-1]+15)
    for i in w_cr:
        plt.axvline(i,color='k',lw=.5)
        plt.text(i-2,1.05,'%.1f'%i,fontsize=12)
    ax = plt.gca()
    plt.axvspan(0,w_cr[0],fc='r',ec='none',alpha=alpha)
    plt.axvspan(w_cr[0],w_cr[1],fc='green',ec='none',alpha=alpha)
    plt.axvspan(w_cr[1],w_cr[2],fc='orange',ec='none',alpha=alpha)
    plt.axvspan(w_cr[2],1e4,fc='r',ec='none',alpha=alpha)
    if h not in [1.6,1.71]:
        plt.text(.02,.8,'Height=%s cm'%(100*h),fontsize=12,
            transform=ax.transAxes)
    plt.grid()
    # plt.xlabel('Body weight (kg)')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('BMI_h%d.pdf'%(100*h))
    plt.close()



