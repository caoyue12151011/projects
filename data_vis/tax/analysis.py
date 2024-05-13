import itertools
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


# tax rate of monthly income
Node = np.array([0,5000,8000,17000,30000,40000,60000,85000,np.inf])
Rate = np.array([0,      .03,   .1,   .2,  .25,   .3,  .35,   .45])


def calc_tax(Inc):
    tax = 0 
    for i in range(len(Rate)):
        if Inc < Node[i+1]:
            tax += (Inc-Node[i])*Rate[i]
            break 
        else:
            tax += (Node[i+1]-Node[i])*Rate[i]

    return Inc-tax



# parameters 
Inc = np.linspace(0,1e5,1000) # Monthly income before tax 

# after-tax income 
Inc1 = Inc.copy()
for i in range(len(Inc)):
    Inc1[i] = calc_tax(Inc[i])

Tax = Inc - Inc1

Node1 = Node.copy()
for i in range(len(Node)):
    Node1[i] = calc_tax(Node[i])



# after-tax income 
plt.figure()
plt.plot(Inc,Inc1,color='k')
plt.plot(Inc,Inc,color='k',ls=':')
for i in range(1,len(Rate)):
    plt.scatter(Node[i],Node1[i],fc='w',ec='k',s=30,zorder=45)
    plt.text(Node[i]+1e3,Node1[i]-1e3,'%d'%Node[i],fontsize=14)
plt.grid()
plt.xlabel('Monthly income before tax (¥)')
plt.ylabel('Monthly income after tax (¥)')
plt.tight_layout()
plt.savefig('monthly.pdf')
plt.close()

# tax
plt.figure()
plt.plot(Inc,Tax,color='k')
plt.grid()
plt.xlabel('Monthly income before tax (¥)')
plt.ylabel('Tax (¥)')
plt.tight_layout()
plt.savefig('monthly_tax.pdf')
plt.show()























