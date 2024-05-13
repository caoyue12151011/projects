import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patheffects as path_effects


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


# load data
f = open('data.txt','r')
for i in range(4):
    f.readline()
data = np.loadtxt(f.readlines())
f.close()

Bin = np.concatenate((np.array([100]),data[:,0]))
Pct = data[:,1]

Cen = (Bin[:-1]+Bin[1:])/2
Acc = np.cumsum(Pct)
Yticklabel = ['%d'%b for b in Bin]
Yticklabel[0] = '<0'
Yticklabel[-1] = '20000+'


#''' income distributions  
plt.figure(figsize=(4,6))
plt.title('            Income distribution (2019)',fontsize=16,weight='bold')
plt.gca().axis('off')
plt.barh(np.arange(len(Cen)),Pct,height=1,fc='r',ec='w',lw=0,alpha=.7)
for i in range(len(Cen)):
    plt.text(Pct[i],i-.1,'%.1f%%'%Pct[i],fontsize=14,color='k',weight='bold')
for i in range(len(Bin)):
    plt.axhline(i-.5,color='k',ls='--',lw=1)
    plt.text(25,i-.55,'¥'+Yticklabel[i],fontsize=12,color='k')
plt.tight_layout()
plt.savefig('income.pdf')
plt.show()
#'''


plt.figure()
plt.plot(Bin[1:],Acc)
plt.grid()
plt.show()


