import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.patches import Rectangle


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


Beliefs = ['Ruism','Daoism','Fajia','Mohism','Bingjia']

# Note: -300 = 300BC, when plotting, add one. no '0'
People = {
    'Laozi': {
        'birth-death': [-571,-471],
        'belief': 'Daoism',
        'position': 0, 
    },
    'Confucius': {
        'birth-death': [-551,-479],
        'belief': 'Ruism',
        'position': 1, 
    },
    'Sunzi': {
        'birth-death': [-544,-496],
        'belief': 'Bingjia',
        'position': 2, 
    },
    'Mozi': {
        'birth-death': [-470,-391],
        'belief': 'Mohism',
        'position': 0, 
    },
    'Sun Bin': {
        'birth-death': [-382,-316],
        'belief': 'Bingjia',
        'position': 0, 
    },
    'Mencius': {
        'birth-death': [-372,-289],
        'belief': 'Ruism',
        'position': 1, 
    },
    'Zhuangzi': {
        'birth-death': [-369,-286],
        'belief': 'Daoism',
        'position': 2, 
    },
    'Xunzi': {
        'birth-death': [-310,-238],
        'belief': 'Ruism',
        'position': 0, 
    },
    'Han Fei': {
        'birth-death': [-280,-233],
        'belief': 'Fajia',
        'position': 2, 
    },
}


plt.figure(figsize=(10,4))
plt.ylim(-2,4)
ax = plt.gca()
for people in People:
    tmp = People[people]
    birth, death = tmp['birth-death']
    belief = tmp['belief']
    position = tmp['position']

    ind = Beliefs.index(belief)

    if birth<0:
        birth += 1 
    if death<0:
        death += 1 

    plt.barh(position,death-birth,height=.8,left=birth,fc='C%d'%ind,ec='none')
    plt.text(.8*birth+.2*death,position-.2,people,color='w',weight='bold',
        fontsize=18)

xticks, xlabel0 = plt.xticks()
xlabel = []
for i in range(len(xticks)):
    text = xlabel0[i].get_text()
    if xticks[i]<0:
        xlabel.append(text[1:]+'BE')
        xticks[i] += 1
    else:
        xlabel.append(text+'AD')

ax.set_xticks(xticks)
ax.set_xticklabels(xlabel)
plt.xlabel('Year')
plt.yticks([])
plt.grid()
plt.tight_layout()
plt.savefig('people.pdf')
plt.show()
