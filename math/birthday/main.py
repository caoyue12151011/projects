'''
Chance of same birthdays in M people.
'''
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


# parameters 
N = 365
M = np.arange(1, 50)

pct = [100*(1 - factorial(N)/factorial(N-int(m))/N**int(m)) for m in M]

plt.figure(figsize=(10,5))
plt.bar(M, pct, fc='gray', ec='none')
plt.grid()
plt.xlabel('Number of people')
plt.ylabel('Probability (%)')
plt.tight_layout()
plt.savefig('birthday_prob.pdf')
plt.close()
