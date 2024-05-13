'''
To analyze the asteroid stats and mining strategies.

Outputs 
-------
MiningLab:

    experiment: 'Exp1', 'Exp2', ...

        'crystal_value': gem yield factor

        'asteroids'

            size: 1,2,3,...,9 

                'yields': n*m array, gem yields of asteroids. n is the number 
                    of asteroids, m is the number of max fragments. =nan if 
                    there are no corresponding fragments. 

                'total_yield': 1D array of total gem yield of each asteroid
                'frag_No': 1D array of gem fragmentation numbers

Para_asteroid

    'size': [1,2,3,...,9]
    'HP': 1D array of asteroid HP when crystal_value=1
'''


import dill
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import factorial


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def mining_yield(CargoCap,i_max,tier,DPS,YPD):
    '''
    To calculate a ship's gem yield as a function of mining time.

    Inputs
    ------
    CargoCap: cargo capacity
    i_max: max number of upgrades, can be 0,1,2,...,tier
    tier: tier of the ship 
    DPS: 1D array of size tier, damage per sec after each upgrades
    YPD: yield per damage for asteroids

    Returns
    -------
    t: [s], 1D array of time, 
        = [0, t1, t1, t2, t2, ..., t_i_max, t_i_max, 1.1*CargoCap/YPS[0]]
    y: 1D array of the gem yield
    '''

    YPS = DPS[:i_max+1]*YPD # yield per sec
    rYPS = 1/YPS

    # time 
    t = 5*tier*np.cumsum(rYPS[:-1])
    t = np.array([0] + list(np.repeat(t,2)) + [1.1*CargoCap/YPS[0]])

    y = np.full(len(t),np.nan)
    for i in range(int(len(t)/2)):
        y[2*i:2*i+2] = YPS[i]*(t[2*i:2*i+2]-5*tier*np.sum(rYPS[:i]))

    return t, y


def order_str(i):
    y = 'th'
    if i==1:
        y = 'st'
    elif i==2:
        y = 'nd'
    elif i==3:
        y = 'rd'

    return y


# load variables
Para_ship = dill.load(open('variable/Para_ship.p','rb'))
Para_tier = dill.load(open('variable/Para_tier.p','rb'))
calc_trait = dill.load(open('variable/calc_trait.p','rb'))
Colors_rb = dill.load(open('variable/Colors_rb.p','rb'))


# parameters
frag_upper_lim = 5 # upper limit of frag #, affects the shape of 'yields' array
Size = [1,2,3,4,5,6,7,8,9] # asteroid sizes
Exp = ['Exp1','Exp2']

# Para_asteroid 
HP = np.loadtxt('data_in/asteriod_stats.txt')[:,1]
Para_asteroid = {
    'size': Size,
    'HP': HP,
}


# data of mining experiments ---------------------------------------------------

MiningLab = {}

for exp in Exp:

    # find crystal_value data 
    f = open('data_in/mining/%s/readme.txt'%exp,'r')
    f.readline()
    crystal_value = int(f.readline().split(':')[-1])
    f.close()

    MiningLab[exp] = {
        'crystal_value': crystal_value,
        'asteroids': {},
    }

    for size in Size:

        # yield data
        f = open('data_in/mining/%s/size%d.txt'%(exp,size),'r')
        data = f.readlines()
        f.close()

        yields = np.full((len(data),frag_upper_lim),np.nan)
        for i in range(len(data)):
            val = [int(j) for j in data[i].split(',')]
            yields[i][:len(val)] = val

        total_yield = np.nansum(yields,axis=1)
        frag_No = np.nansum(~np.isnan(yields),axis=1)


        # save
        MiningLab[exp]['asteroids'][size] = {
            'yields': yields,
            'total_yield': total_yield,
            'frag_No': frag_No,
        }

# output 
dill.dump(MiningLab,open('variable/MiningLab.p','wb'))
dill.dump(Para_asteroid,open('variable/Para_asteroid.p','wb'))


# Outputs ======================================================================

# terminal ---------------------------------------------------------------------

''' total gem yield per HP
# seems that mean(total_yield) per asteroid is prop to crystal_value^.5

for size in Size:
    HP = Para_asteroid['HP'][size-1]
    total_yield = np.array([np.mean(
        MiningLab[exp]['asteroids'][size]['total_yield']) for exp in Exp])
    sp_yield = total_yield/HP

    print('%s size: %.2f.'%(size,sp_yield[1]/sp_yield[0]))
#'''


# figures ----------------------------------------------------------------------

''' asteroid HP vs size

# parameters 
x = np.linspace(1,9,500)
y = 378*(x/9)**3 # approximate model

plt.figure()
plt.xlim(0,10)
plt.yscale('log')
plt.scatter(Size,HP,color=Colors_rb,zorder=45)
for i in range(len(HP)):
    plt.text(Size[i]+.1,1.05*HP[i],'%d'%HP[i],color=Colors_rb[i],fontsize=14)
plt.plot(x,y,color='k',label=r'$\rm HP\propto size^3$')
plt.text(.3,.1,'asteroids_strength=1',fontsize=16,transform=plt.gca().transAxes)
plt.legend()
plt.grid()
plt.xlabel('Asteroid size')
plt.ylabel('Asteroid HP')
plt.tight_layout()
plt.savefig('image/mining/HP_vs_size.pdf')
plt.close()
#'''


''' gem number distribution

gem_No = np.arange(1,frag_upper_lim+1)

for exp in Exp:
    plt.figure()
    # plt.yscale('log')

    for i in range(len(Size)):
        size = Size[i]

        tmp = MiningLab[exp]['asteroids'][size]
        crystal_value = MiningLab[exp]['crystal_value']
        frag_No = tmp['frag_No']
        yields = tmp['yields']

        frag_prob = [100*np.sum(frag_No==j)/len(frag_No) 
            for j in range(1,frag_upper_lim+1)]

        plt.scatter(gem_No,frag_prob,color=Colors_rb[i],s=16)
        plt.plot(gem_No,frag_prob,color=Colors_rb[i],lw=.5,
            label='Size %d (%d asteroids, %d gems)'%(size,len(yields),
                sum(frag_No)))

    plt.text(.1,.9,'crystal_value=%.1f'%crystal_value,fontsize=16,
        transform=plt.gca().transAxes)
    plt.grid()
    plt.xticks(gem_No)
    plt.legend()
    plt.xlabel('# of gems per asteroid')
    plt.ylabel('Frequency of occurrence (%)')
    plt.tight_layout()
    plt.savefig('image/mining/%s/gem_num_distr.pdf'%exp)
    plt.close()
#'''


''' distributions of gem sizes
for exp in Exp:
    # if not exp=='Exp2':
    #     continue 

    crystal_value = MiningLab[exp]['crystal_value']

    for size in Size:
        # if size not in [3]:
        #     continue 
 
        tmp = MiningLab[exp]['asteroids'][size] 
        yields = tmp['yields'] 
        frag_No = tmp['frag_No'] 

        # bins 
        _min = np.nanmin(yields)
        _max = np.nanmax(yields)
        bw = np.ceil((_max-_min)/len(frag_No)**.5)
        bins = np.arange(_min-.5,_max+bw+.5,bw)

        Y = np.zeros(len(bins)-1)
        YY = Y.copy()

        plt.figure(figsize=(7,5))

        # gems with different frag No.
        for i in range(1,max(frag_No)+1):
            ind = frag_No==i 

            # the j-th largest gems 
            for j in range(i):
                data = yields[ind][:,j]
                label = '%d%s gem of %d (%d gems)'%(j+1,order_str(j+1),i,
                    len(data))

                y = plt.hist(data,bins,bottom=Y,histtype='stepfilled',
                    fc=Colors_rb[i],ec='none',alpha=1-j/i,label=label)[0]
                Y += y

            Data = yields[ind][:,:i].flatten()
            yy = plt.hist(Data,bins,bottom=YY,histtype='step',lw=1,ec='k',
                zorder=49)[0]
            YY += yy

        # Poisson distribution test
        # lamda = 4.5
        # x = np.arange(1,8)
        # y = 626* lamda**x*np.exp(-lamda)/factorial(x)
        # plt.scatter(x,y,color='k',zorder=49)

        text = '%d asteroids\n%d gems\ncrystal_value=%.1f'%(len(yields),
            sum(frag_No),crystal_value)
        plt.text(.4,.8,text,fontsize=14,transform=plt.gca().transAxes)
        plt.legend(fontsize=10)
        plt.grid()
        plt.xlabel('Gem size')
        plt.ylabel('# of gems (stacked)')
        plt.tight_layout()
        plt.savefig('image/mining/%s/gem_distr/size%d.pdf'%(exp,size))
        plt.close()
#'''


''' distributions of total yields 
for exp in Exp:
    # if not exp=='Exp2':
    #     continue 

    crystal_value = MiningLab[exp]['crystal_value']

    for size in Size:
        # if size not in [3]:
        #     continue

        tmp = MiningLab[exp]['asteroids'][size]
        total_yield = tmp['total_yield']
        frag_No = tmp['frag_No']

        _min = np.nanmin(total_yield)
        _max = np.nanmax(total_yield)
        bw = int(np.ceil((_max-_min)/len(frag_No)**.5))
        bins = np.arange(_min-.5,_max+bw+.5,bw)

        Y = np.zeros(len(bins)-1)

        plt.figure(figsize=(7,5))

        # total_yield with different frag No.
        for i in range(1,max(frag_No)+1):
            data = total_yield[frag_No==i]

            label = '%d gem'%i 
            if i>1:
                label += 's'
            label += ' (%d asteroids)'%len(data)

            y = plt.hist(data,bins,bottom=Y,histtype='stepfilled',
                fc=Colors_rb[i],ec='k',alpha=.8,lw=1,label=label)[0]
            Y += y

        text = '%d asteroids\n%d gems\ncrystal_value=%.1f'%(len(total_yield),
            sum(frag_No),crystal_value)
        plt.text(.05,.8,text,fontsize=14,transform=plt.gca().transAxes)
        plt.legend()
        plt.grid()
        plt.xlabel('Total gem yield per asteroid')
        plt.ylabel('# of asteroids (stacked)')
        plt.tight_layout()
        plt.savefig('image/mining/%s/tot_yield_distr/size%d.pdf'%(exp,size))
        plt.close()
#'''


''' gem size correlations
for exp in Exp:
    if not exp=='Exp2':
        continue 

    crystal_value = MiningLab[exp]['crystal_value']

    for size in Size:
        if size not in [3]:
            continue

        tmp = MiningLab[exp]['asteroids'][size]
        yields = tmp['yields']
        frag_No = tmp['frag_No']

        for n_g in range(2,frag_No.max()+1):
            # if not n_g==4:
            #     continue

            data = yields[frag_No==n_g]

            plt.figure(figsize=(6,5))

            for i in range(1,n_g):
                for j in range(i+1,n_g+1): # i'th-j'th gem pair
                    # if not (i==1 and j==4):
                    #     continue

                    # correlation matrix, shape: (j-axis, i-axis)
                    col_i = data[:,i-1]
                    col_j = data[:,j-1]  
                    axis_i = np.arange(col_i.min(),col_i.max()+1)
                    axis_j = np.arange(col_j.min(),col_j.max()+1)

                    Corr = np.zeros((len(axis_j),len(axis_i)))
                    for c_i, c_j in zip(col_i,col_j): 
                        ii = np.where(c_i==axis_i)[0][0]
                        jj = np.where(c_j==axis_j)[0][0]
                        Corr[jj,ii] += 1
                    # print(Corr)

                    # demo
                    plt.subplot(n_g-1,n_g-1,i+(n_g-1)*(j-2))
                    ax = plt.gca()                        
                    plt.imshow(Corr,origin='lower',aspect='auto',
                        extent=[axis_i[0]-.5,axis_i[-1]+.5,
                                axis_j[0]-.5,axis_j[-1]+.5])

                    # text
                    if i==1 and j==n_g:
                        text = 'crystal_value=%.1f\n%d asteroids'%(
                            crystal_value,len(col_i))
                        plt.text(.03,.7,text,fontsize=10,color='w', 
                            transform=ax.transAxes)

                    if j==n_g:
                        ax.set_xticks(axis_i)
                        ax.set_xticklabels(['%d'%s for s in axis_i])
                        plt.xlabel('%d%s gem'%(i,order_str(i)))
                    else:
                        ax.axes.xaxis.set_ticklabels([])

                    if i==1:
                        ax.set_yticks(axis_j)
                        ax.set_yticklabels(['%d'%s for s in axis_j])
                        plt.ylabel('%d%s gem'%(j,order_str(j)))
                    else:
                        ax.axes.yaxis.set_ticklabels([])

            plt.tight_layout()
            plt.savefig('image/mining/%s/gem_corr/size%d_%dgems.pdf'
                %(exp,size,n_g))
            plt.close()
#'''


''' Portrait of all asteroids

# parameters
size_fct_a = 2500
size_fct_g = .06
figsize = 6 # [in]

for exp in Exp:
    # if not exp=='Exp2':
    #     continue 

    for size in Size:
        # if size not in [3]:
        #     continue

        tmp = MiningLab[exp]['asteroids'][size]
        yields = tmp['yields']
        frag_No = tmp['frag_No']

        n_a = len(yields) # number of asteroids 
        max_gem = np.nanmax(yields)
        n_col = int(n_a**.5) # number of columns 

        # asteroid parameters
        s_a = size_fct_a*figsize**2/n_a # dot size of asteroids
        X_a = np.arange(n_a)%n_col 
        Y_a = -(np.arange(n_a)//n_col)   

        # gem parameters
        S_g = [] 
        X_g = [] 
        Y_g = []

        for i in range(n_a):
            data = yields[i][:frag_No[i]]

            s_g = size_fct_g*s_a*data/max_gem

            x_a = X_a[i]
            y_a = Y_a[i]
            dx = .13*np.arange(frag_No[i])
            dx -= dx[-1]/2
            x_g = x_a + dx
            y_g = y_a + np.zeros(frag_No[i])

            S_g.extend(list(s_g))
            X_g.extend(list(x_g))
            Y_g.extend(list(y_g))

        # demo
        fig = plt.figure(figsize=(figsize,figsize))
        ax = fig.add_axes([0,0,1,1])
        plt.scatter(X_a,Y_a,s=s_a,fc='k',ec='none',alpha=.3) # asteroids
        plt.scatter(X_g,Y_g,s=S_g,fc='r',ec='none') # gems
        ax.spines['left'].set_alpha(0)
        ax.spines['right'].set_alpha(0)
        ax.spines['bottom'].set_alpha(0)
        ax.spines['top'].set_alpha(0)
        plt.savefig('image/mining/%s/portrait/size%d.pdf'%(exp,size))
        plt.close()
#'''

