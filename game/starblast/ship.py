'''
To analyze the ship stats.

Outputs
-------
Para_ship: OrderedDict

    ship: 'Fly', 'A-Speedster', etc.

        'Tier'
        'Mass'
        'MinShieldRegen'
        'MaxShieldRegen'
        'MinShieldCap'
        'MaxShieldCap'
        'MinEnergyRegen'
        'MaxEnergyRegen'
        'MinEnergyCap'
        'MaxEnergyCap'
        'MinShipSpeed'
        'MaxShipSpeed'
        'MinShipRotate'
        'MaxShipRotate'
        'MinShipAccel'
        'MaxShipAccel'
        'MinBurstDPS'
        'MaxBurstDPS'
        'MinBurstDamage'
        'MaxBurstDamage'

        'leaf_ships': list, ships that can be updated to


Para_tier: OrderedDict

    tier: 1,2,...

         'CargoCap'
         'UpgradeNo': number of upgrades
         'UpgradeCost'
         'color'
         'Ships': list of ship names
         'Mass_max':
         'ShieldRegen_max'
         'ShieldCap_max'
         'EnergyRegen_max'
         'EnergyCap_max'
         'BurstDPS_max'
         'BurstDamage_max'
         'ShipSpeed_max'
         'ShipRotate_max'
         'ShipAccel_max'


Colors_rb: list of rainbow color series
calc_trait: function to calculate ship traits
'''

import dill
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


def calc_trait(_min,_max,tier,i):
    '''
    To calculate the upgraded trait values. 

    Inputs
    ------
    _min, _max: min/max trait values
    tier: tier of the ship
    i: i'th upgrade, = 0, 1, ..., tier 

    Returns
    -------
    upgraded trait value
    '''

    return (_max-_min)*i/tier+_min



def kill_time(S0A,rSA,E0A,rEA,bA,S0B,rSB,E0B,rEB,bB):
    '''
    To calculate the time needed for ship A to kill ship B.

    Inputs
    ------
    S0: shield capacity of the ship
    rS: shield regeneration
    E0: energy capacity
    rE: energy regeneration
    b: burst DPS

    Returns
    -------
    t_AB: [s], the time needed for ship A to kill ship B
    '''

    # different conditions considered
    cond_a1 = bA <= rSB 
    cond_a2 = (rEA <= rSB) and (rSB < bA)
    cond_a3 = rSB < rEA  
    cond_b = S0B*(bA-rEA) > E0A*(bA-rSB)

    t_AB = None 
    if cond_a1 or (cond_a2 and cond_b):
        t_AB = np.inf
    elif (cond_a2 or cond_a3) and not cond_b:
        t_AB = S0B/(bA-rSB)
    else:
        t_AB = (S0B-E0A)/(rEA-rSB)

    return t_AB



def ship_match(shipA,shipB,maxA,maxB):
    '''
    To calculate the outcome of a ship match.

    Inputs
    ------
    shipA, shipB: subdicts Para_ship[ship]
    maxA, maxB: T/F, whether to use maxed trait values

    Returns
    -------
    t_AB: [s], the time needed for ship A to kill ship B
    win_AB: winning function. >0 A wins, <0 A loses, =0 or nan ties 
    '''

    # retrieve data
    prefixA = 'Max' if maxA else 'Min'
    prefixB = 'Max' if maxB else 'Min'

    S0A = shipA[prefixA+'ShieldCap']
    rSA = shipA[prefixA+'ShieldRegen']
    E0A = shipA[prefixA+'EnergyCap']
    rEA = shipA[prefixA+'EnergyRegen']
    bA = shipA[prefixA+'BurstDPS']

    S0B = shipB[prefixB+'ShieldCap']
    rSB = shipB[prefixB+'ShieldRegen']
    E0B = shipB[prefixB+'EnergyCap']
    rEB = shipB[prefixB+'EnergyRegen']
    bB = shipB[prefixB+'BurstDPS']

    # initial checking 
    if bA<=rEA or bB<=rEB:
        print('Error: bA<=rEA or bB<=rEB.')
     
    # calculate killing time
    t_AB = kill_time(S0A,rSA,E0A,rEA,bA,S0B,rSB,E0B,rEB,bB) # A kills B
    t_BA = kill_time(S0B,rSB,E0B,rEB,bB,S0A,rSA,E0A,rEA,bA)
    win_AB = np.log10(t_BA/t_AB)

    return t_AB, win_AB



# global parameters
Traits = ['Mass','ShieldRegen','ShieldCap','EnergyRegen','EnergyCap',
    'BurstDPS','BurstDamage','ShipSpeed','ShipRotate','ShipAccel']
TraitsAll = ['Tier','Mass','MinShieldRegen','MaxShieldRegen',
    'MinShieldCap','MaxShieldCap','MinEnergyRegen','MaxEnergyRegen',
    'MinEnergyCap','MaxEnergyCap','MinShipSpeed','MaxShipSpeed',
    'MinShipRotate','MaxShipRotate','MinShipAccel','MaxShipAccel',
    'MinBurstDPS','MaxBurstDPS','MinBurstDamage','MaxBurstDamage']
Colors_rb = ['brown','red','orange','y','green','c','blue','purple','black']
    # rainbow colors


# read ship statistics
f = open('data_in/ship_stats.txt','r')
header = f.readline()
header = header[:-1].split(',')[2:] # ship traits 
data = f.readlines()
f.close()


# create ship dict
Para_ship = OrderedDict()
for line in data:
    line = line.split('\n')[0].split(',') 

    name = line.pop(0)
    ClsName = line.pop(0)
    line = [int(i) for i in line]

    Para_ship[name] = {}
    for i in range(len(header)):
        Para_ship[name][header[i]] = line[i]


# ship tree info
f = open('data_in/ship_tree.txt','r')
f.readline()
f.readline()
data = f.readlines()
f.close()

for line in data:
    ship, leaves = line[:-1].split(';')
    leaves = leaves.split(',')
    Para_ship[ship]['leaf_ships'] = leaves


# read tier statistics
f = open('data_in/tier_stats.txt','r')
header = f.readline().split()[1:] # ['CargoCap','UpgradeNo','UpgradeCost']
data = f.readlines()
f.close()


# create tier dict
Para_tier = OrderedDict()
for line in data:
    line = line.split()
    line = [int(i) for i in line]
    tier = line.pop(0)

    Para_tier[tier] = {}
    for i in range(len(header)):
        Para_tier[tier][header[i]] = line[i]


# tier colors for demo
for tier in Para_tier:
    Para_tier[tier]['color'] = Colors_rb[tier]


# other tier properties
for tier in Para_tier:

    # ship list of the same tier 
    Names = []
    for name in Para_ship:
        if Para_ship[name]['Tier']==tier:
            Names.append(name)
    Para_tier[tier]['Ships'] = Names

    # find max traits values 
    for trait in Traits:
        _max = -np.inf 
        for name in Names:
            for keys in Para_ship[name]:
                if trait in keys:
                    if _max < Para_ship[name][keys]:
                        _max = Para_ship[name][keys]
        Para_tier[tier][trait+'_max'] = _max


# output -----------------------------------------------------------------------

dill.dump(Para_tier,open('variable/Para_tier.p','wb'))
dill.dump(Para_ship,open('variable/Para_ship.p','wb'))
dill.dump(Colors_rb,open('variable/Colors_rb.p','wb'))
dill.dump(calc_trait,open('variable/calc_trait.p','wb'))


# demonstration ================================================================

''' ship traits rose map 

# parameters 
sv1 = 1 # [in], vertical margins from top to bottom
sv2 = .4
sv3 = .4
sh1 = .8 # left/right horizontal margins
hl  = .4 # legend axis height 
wb1 = 4 # left polar box width
hb = 4 # polar box height

# angle 
Theta = np.arange(0,2*np.pi,2*np.pi/len(Traits))
Theta = np.array(list(Theta)+[Theta[0]])

for tier in Para_tier:
    # if tier not in [6]:
    #     continue
    if tier==1:
        continue

    # data 
    Names = Para_tier[tier]['Ships']
    Data_min = []
    Data_max = []

    for trait in Traits+[Traits[0]]:
        _max = Para_tier[tier][trait+'_max']

        if trait=='Mass':
            data = np.array([Para_ship[name][trait] for name in Names])/_max
            Data_min.append(data)
            Data_max.append(data)
        else:
            data_min = np.array([Para_ship[name]['Min'+trait] for name in 
                Names])/_max
            data_max = np.array([Para_ship[name]['Max'+trait] for name in 
                Names])/_max
            Data_min.append(data_min)
            Data_max.append(data_max)
    Data_min = np.array(Data_min)
    Data_max = np.array(Data_max)


    # variable figure parameters
    sh2 = 1.5 # middle horizontal margins
    wb2 = 4 # right polar box width
    if tier==7:
        sh2 = 0
        wb2 = 0

    # demo
    fig_x = 2*sh1+sh2+wb1+wb2
    fig_y = hl+sv1+sv2+sv3+hb 
    plt.figure(figsize=(fig_x,fig_y))


    # left polar plot
    left = sh1/fig_x 
    bottom = (hl+sv2+sv3)/fig_y
    width = wb1/fig_x 
    height = wb1/fig_y
    ax = plt.axes([left,bottom,width,height],polar=True)

    for i in range(len(Names)):
        plt.plot(Theta,Data_min[:,i],color=Colors_rb[i],linewidth=1)
        plt.fill_between(Theta,np.zeros(len(Theta)),Data_min[:,i],
            color=Colors_rb[i],alpha=.1,label=Names[i])
    plt.thetagrids(Theta[:-1]*180/np.pi,Traits,fontsize=14,weight='bold')
    plt.rgrids([],[])
    ax.spines['polar'].set_alpha(.3)
    if tier==7:
        plt.title('Tier %d ships'%tier,fontsize=26,weight='bold',y=1.12)
    else:
        plt.title('Tier %d ships (Min.)'%tier,fontsize=26,weight='bold',y=1.12)


    # right polar plot
    if not tier==7:
        left = (sh1+sh2+wb1)/fig_x 
        bottom = (hl+sv2+sv3)/fig_y
        width = wb2/fig_x 
        height = wb1/fig_y
        ax = plt.axes([left,bottom,width,height],polar=True)

        for i in range(len(Names)):
            plt.plot(Theta,Data_max[:,i],color=Colors_rb[i],linewidth=1)
            plt.fill_between(Theta,np.zeros(len(Theta)),Data_max[:,i],
                color=Colors_rb[i],alpha=.1)
        plt.thetagrids(Theta[:-1]*180/np.pi,Traits,fontsize=14,weight='bold')
        plt.rgrids([],[])
        ax.spines['polar'].set_alpha(.3)
        plt.title('Tier %d ships (Max.)'%tier,fontsize=26,weight='bold',y=1.12)


    # legend axis
    left = 0
    bottom = sv3/fig_y
    width = 1
    height = hl/fig_y
    ax = plt.axes([left,bottom,width,height])

    for i in range(len(Names)):
        plt.fill_betweenx([0,1],[i,i],[i+.7]*2,color=Colors_rb[i],alpha=.1,
            linewidth=1)
        plt.text(i,.3,Names[i],fontsize=12,weight='bold')
    plt.xticks([])
    plt.yticks([])
    ax.spines['left'].set_alpha(0)
    ax.spines['right'].set_alpha(0)
    ax.spines['bottom'].set_alpha(0)
    ax.spines['top'].set_alpha(0)

    plt.savefig('data_out/image/ship_rose/tier%d.pdf'%tier)
    plt.close()
#'''


''' ship trait bar plot

# parameters 
s = 36 # dot size

for trait in Traits:
    # if trait not in ['BurstDamage']:
    #     continue

    plt.figure(figsize=(8,5))
    ax = plt.gca()

    i = 0
    for ship in Para_ship:
        tmp = Para_ship[ship]
        tier = tmp['Tier']
        color = Para_tier[tier]['color']

        # data
        value0 = value1 = None
        if trait in tmp:
            value0 = value1 = tmp[trait]
        else:
            value0 = tmp['Min'+trait]
            value1 = tmp['Max'+trait]

        # plot
        if value0==value1:
            plt.scatter(i,value0,s=s,fc=color,ec=color)
        else:
            plt.bar(i,value1-value0,.3,bottom=value0,fc=color,ec='none')
        plt.plot([i,i],[0,value0],lw=1,ls=':',c=color)

        i += 1


    # add upgrade yields 
    i = 0
    ymin, ymax = plt.ylim()

    for ship in Para_ship:
        tmp = Para_ship[ship]
        tier = tmp['Tier']
        color = Para_tier[tier]['color']

        if trait not in tmp:
            value0 = tmp['Min'+trait]
            value1 = tmp['Max'+trait]

            if not value0==value1:
                pct = 100*(value1/value0-1)
                plt.text(i-.4,value1+.01*(ymax-ymin),'+%.0f%%'%pct,fontsize=7,
                    color=color,weight='bold')
        i += 1


    plt.xlim(-.5,i-.5)
    plt.ylim(bottom=0)
    plt.grid(axis='y')
    ticks = np.arange(len(Para_ship))
    labels = list(Para_ship.keys())
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels,rotation='vertical',weight='bold')
    plt.title(trait,fontsize=20,weight='bold')
    plt.tight_layout()
    plt.savefig('image/ship/ship_traits/%s.pdf'%trait)
    plt.close()
#'''


''' ship match diagram 

# constructing winning function matrix (A is offensive, B defensive)
for maxA, maxB in [[True,True],[True,False],[False,True],[False,False]]:

    Name_ship = list(Para_ship)
    Name_ship.pop(Name_ship.index('Barracuda'))
    num_ship = len(Name_ship)
    Win_AB = np.full((num_ship,num_ship),np.nan)
    T_AB = np.full((num_ship,num_ship),np.nan)

    for i in range(num_ship):
        # if not i==2:
        #     continue 
        for j in range(num_ship):
            # if not j==20:
            #     continue 

            nameA = Name_ship[i]
            nameB = Name_ship[j]
            t_AB, win_AB = ship_match(Para_ship[nameA],Para_ship[nameB],
                maxA,maxB) 
            T_AB[i,j] = t_AB
            Win_AB[i,j] = win_AB


    # T_AB map
    plt.figure(figsize=(7,6))
    ax = plt.gca()
    plt.imshow(np.ones_like(T_AB),cmap='gray',zorder=19,vmin=10,vmax=11)
    plt.imshow(np.log10(T_AB),cmap='Blues',origin='lower',zorder=20)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'$\rm log_{10}(killing\ time [s])$')

    if maxB:
        plt.xlabel('Defensive ships (max traits)',fontsize=14,weight='bold')
    else:
        plt.xlabel('Defensive ships (min traits)',fontsize=14,weight='bold')
    if maxA:
        plt.ylabel('Offensive ships (max traits)',fontsize=14,weight='bold')
    else:
        plt.ylabel('Offensive ships (min traits)',fontsize=14,weight='bold')
    ax.set_xticks(np.arange(num_ship))
    ax.set_xticklabels(Name_ship,rotation=90,fontsize=8)
    ax.set_yticks(np.arange(num_ship))
    ax.set_yticklabels(Name_ship,fontsize=8)
    plt.box(on=None)
    plt.tight_layout()
    plt.savefig('data_out/image/ship_match/kill_time_maxX%s_maxY%s.pdf'%(maxB,
        maxA))
    plt.close()


    # Win_AB map
    Win_AB[Win_AB==np.inf] = np.max(Win_AB[~np.isinf(Win_AB)])
    Win_AB[Win_AB==-np.inf] = np.min(Win_AB[~np.isinf(Win_AB)])

    plt.figure(figsize=(7,6))
    ax = plt.gca()
    plt.imshow(Win_AB,cmap='bwr',origin='lower')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Blue wins'+' '*20+'Tie'+' '*20+'Red wins',
        fontsize=14)
    cbar.ax.set_yticklabels('')
    cbar.ax.tick_params(axis=u'both',which=u'both',length=0)

    ax.set_xticks(np.arange(num_ship))
    ax.set_xticklabels(Name_ship,rotation=90,fontsize=8,color='blue')
    ax.set_yticks(np.arange(num_ship))
    ax.set_yticklabels(Name_ship,fontsize=8,color='red')
    if maxB:
        plt.xlabel('Max traits',fontsize=14,weight='bold',color='blue')
    else:
        plt.xlabel('Min traits',fontsize=14,weight='bold',color='blue')
    if maxA:
        plt.ylabel('Max traits',fontsize=14,weight='bold',color='red')
    else:
        plt.ylabel('Min traits',fontsize=14,weight='bold',color='red')
    plt.box(on=None)
    plt.tight_layout()
    plt.savefig('data_out/image/ship_match/win_func_maxX%s_maxY%s.pdf'%(maxB,
        maxA))
    plt.close()
#'''


''' UpgradeCost vs EnergyRegen
plt.figure(figsize=(8,5))

Ships = []
Marginal = []
for tier in Para_tier:
    if tier==7:
        continue

    UpgradeCost = Para_tier[tier]['UpgradeCost'] 
    c = Colors_rb[tier]

    for ship in Para_tier[tier]['Ships']:
        MinEnergyRegen = Para_ship[ship]['MinEnergyRegen']
        MaxEnergyRegen = Para_ship[ship]['MaxEnergyRegen']
        marginal = (MaxEnergyRegen-MinEnergyRegen)/tier 

        plt.scatter(UpgradeCost,marginal,fc='none',ec=c,s=40)
        plt.text(UpgradeCost,marginal,ship,fontsize=12,color=c)

        Ships.append(ship)
        Marginal.append(marginal)
Ships = np.array(Ships)
Marginal = np.array(Marginal)

plt.grid()
plt.xlabel('Gem cost for upgrading')
plt.ylabel('Increase of EnergyRegen for each upgrade')
plt.tight_layout()
plt.savefig('data_out/image/optimal_mining/UpgradeCost_vs_EnergyRegen.pdf')
plt.close()


# sort Marginal
Ind = np.argsort(Marginal)[::-1]
Ships = Ships[Ind]
Marginal = Marginal[Ind]

plt.figure(figsize=(7,5))
for i in range(len(Ships)):
    ship = Ships[i]
    tier = Para_ship[ship]['Tier']
    c = Colors_rb[tier]

    plt.bar(i,Marginal[i],fc=c,ec='none',zorder=45,alpha=.8)

plt.grid(axis='y')
ax = plt.gca()
ax.set_xticks(np.arange(len(Ships)))
ax.set_xticklabels(Ships,rotation='vertical')
plt.ylabel('EnergyRegen per gem cost')
plt.tight_layout()
plt.savefig('data_out/image/optimal_mining/EnergyRegen_per_cost.pdf')
plt.close()
#'''


''' best mining strategy

# parameters
method = ['ram','fire']
n_col = 6
figsize = (23,14) # [in]

# derived parameters 
Ships = [ship for ship in list(Para_ship) if Para_ship[ship]['Tier']<7]
n_row = int(np.ceil(len(Ships)/n_col))


for md in method:
    # if md not in ['fire']:
    #     continue

    YPD = yield_per_HP_av if md=='fire' else 1 # yield per damage

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=.03,right=.99,bottom=.04,top=.98,wspace=.15,
        hspace=.15)

    for k in range(len(Ships)):
        ship = Ships[k]
        i, j = divmod(k,n_col)

        # load data
        tmp = Para_ship[ship]
        tier = tmp['Tier']
        DPS1 = tmp['MinShieldRegen'] if md=='ram' else tmp['MinEnergyRegen']
        DPS2 = tmp['MaxShieldRegen'] if md=='ram' else tmp['MaxEnergyRegen']

        # derived parameters
        CargoCap = 20*tier**2
        DPS = calc_trait(DPS1,DPS2,tier,np.arange(tier+1))


        # demo
        fig.add_subplot(n_row,n_col,k+1)
        for i_max in range(tier+1):
            t, y = mining_yield(CargoCap,i_max,tier,DPS,YPD)
            c = Para_asteroid[i_max+1]['color']
            plt.plot(t,y,color=c)

        plt.axhline(CargoCap,color='gray',ls='--')
        if k==0:
            plt.text(0,1.02*CargoCap,'Cargo capacity',color='gray',fontsize=12)
        txt = plt.text(.1,.85,ship,fontsize=20,weight='bold',
            color=Para_tier[tier]['color'],transform=plt.gca().transAxes)
        # txt.set_path_effects([path_effects.Stroke(linewidth=2,foreground='k'),
        #     path_effects.Normal()])


        plt.grid()
        if i==n_row-1:
            plt.xlabel('Time (s)')
        if j==0:
            plt.ylabel('Gem yield')


    # legends
    fig.add_subplot(n_row,n_col,n_row*n_col)
    for k in range(7):
        plt.plot(0,0,color=Colors_rb[k],label='%d update(s)'%k)
    plt.legend(loc='lower right',fontsize=14)
    plt.text(-.5,.6,'%s yield/damage=%.2f'%(md,YPD),fontsize=16,weight='bold',
        transform=plt.gca().transAxes)
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.spines['left'].set_alpha(0)
    ax.spines['right'].set_alpha(0)
    ax.spines['bottom'].set_alpha(0)
    ax.spines['top'].set_alpha(0)

    plt.savefig('data_out/image/mining/mining_%s.pdf'%md)
    plt.close()
#'''















