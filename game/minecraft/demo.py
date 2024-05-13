"""
Demonstrate the results from analysis.py.
"""
import os
import dill
import numpy as np
import matplotlib.pyplot as plt

# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13

# load data 
World = dill.load(open('variable/World.p','rb'))
block2id = dill.load(open('variable/block2id.p','rb'))
Block_all = dill.load(open('variable/Block_all.p','rb'))
Color_all = dill.load(open('variable/Color_all.p','rb'))
y_max_mt = dill.load(open('variable/y_max_mt.p','rb'))

MC_seed = World['MC_seed']
MC_version = World['MC_version']
rx1 = World['rx1']
rx2 = World['rx2']
rz1 = World['rz1']
rz2 = World['rz2']
shape_y = World['shape_y']
shape_z = World['shape_z']
shape_x = World['shape_x']
surface_RGBA = World['surface_RGBA']
surface_y = World['surface_y']
block_num = World['block_num']
block_y_num = World['block_y_num']

# .pdf ------------------------------------------------------------------------

''' block y distribution for a group of blocks

# parameters
top_fct = .9 # height of each curve
ref_fct = .9 # height of the reference line
y = np.arange(-64,y_max_mt+1)
Block_kind = {
    'ore': ['coal_ore','copper_ore','iron_ore','redstone_ore','gold_ore',
        'lapis_ore','diamond_ore','emerald_ore'],

    # 'tree': ['acacia_log','birch_log','dark_oak_log','jungle_log',
    #     'oak_log','spruce_log','bamboo'],#'mangrove_log',

    'mineral': ['stone','deepslate','andesite','diorite','granite','tuff',
        'sandstone','terracotta','ice','water'],#'red_sandstone',

    'dirt': ['dirt','coarse_dirt','mycelium','podzol','gravel',
        'sand','clay','snow_block'],#'red_sand',,'mud'

    'amethyst': ['amethyst_cluster'],
}

for kind in Block_kind:
    if kind not in ['amethyst']:
        continue

    block_list = Block_kind[kind]

    # total number of ore blocks
    num_tot = 0
    for block in block_list:
        num_tot += sum(World['block_y_num'][block])
        if kind=='ore':
            num_tot += sum(World['block_y_num']['deepslate_'+block]) 

    # demo
    plt.figure(figsize=(10,7))

    Num = [] # for pie chart
    colors_pie = [] # for pie chart
    x_min, x_max = np.inf, -np.inf
    bottom = 0

    for block in block_list:

        c = Color_all[block2id[block]]/255
        y_num = World['block_y_num'][block]

        # x limits
        ind_non0 = np.where(y_num)[0]-64
        if x_min > ind_non0[0]:
            x_min = ind_non0[0]
        if x_max < ind_non0[-1]:
            x_max = ind_non0[-1]

        if kind=='ore':
            block_d = 'deepslate_'+block
            c_d = Color_all[block2id[block_d]]/255
            y_num_d = World['block_y_num'][block_d]
            y_num_all = y_num+y_num_d

            y_den = top_fct*y_num/max(y_num_all) + bottom
            y_den_d = top_fct*y_num_d/max(y_num_all) + bottom
            y_den_all = top_fct*y_num_all/max(y_num_all) + bottom

            # x limits
            ind_non0 = np.where(y_num_all)[0]-64
            if x_min > ind_non0[0]:
                x_min = ind_non0[0]
            if x_max < ind_non0[-1]:
                x_max = ind_non0[-1]

            plt.step(y,y_den,where='post',lw=1,color=c_d)
            plt.step(y,y_den_d,where='post',lw=1,color=c_d)
            plt.step(y,y_den_all,where='post',lw=1,color=c_d)
            plt.fill_between(y,bottom,y_den,step="post",alpha=0.5,color=c)
            plt.fill_between(y,bottom,y_den_d,step="post",alpha=0.5,color=c_d)

            # ref line 
            # y_ref = ref_fct*top_fct + bottom
            # plt.axhline(y_ref,color='k')

            # text
            text = ' '.join(block.split('_')).capitalize() 
            text += ' (%.3f%%)'%(100*sum(y_num_all)/num_tot)
            plt.text(-63,bottom+.05,text,fontsize=14,weight='bold')

            # for pie chart
            Num.extend([sum(y_num),sum(y_num_d)])
            colors_pie.extend([c,c_d])

        else:
            y_den = top_fct*y_num/max(y_num) + bottom

            plt.step(y,y_den,where='post',lw=1,color=c)
            plt.fill_between(y,bottom,y_den,step="post",alpha=0.5,color=c)

            # ref line 
            # y_ref = ref_fct*top_fct + bottom
            # plt.axhline(y_ref,color='k')

            # text
            text = ' '.join(block.split('_')).capitalize() 
            text += ' (%.3f%%)'%(100*sum(y_num)/num_tot)
            plt.text(-63,bottom+.05,text,fontsize=14,weight='bold')
            
            # for pie chart
            Num.extend([sum(y_num)])
            colors_pie.extend([c])

        bottom += 1

    ax = plt.gca()
    plt.xlim(x_min-3,x_max)

    # text
    text = ('MC %s Overworld\nSeed: %d\n%d'%(MC_version,MC_seed,512*(rx2-rx1+1))
        + r'$\times$' + '%d blocks'%(512*(rz2-rz1+1)))
    t = plt.text(.74,.1,text,transform=ax.transAxes,fontsize=16,weight='bold')
    t.set_bbox(dict(facecolor='w',alpha=0.5,edgecolor='gray',
        boxstyle='round,pad=.5'))

    # xticks
    ticks = np.arange(10*np.ceil(x_min/10),x_max,10)
    labels = ['%d'%i for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    plt.grid(axis='x')
    plt.xlabel('Height (block)')
    plt.yticks([])
    plt.ylabel('Relative abundance')

    # pie chart
    plt.axes([.55,.3,.5,.4])
    patches = plt.pie(Num,colors=colors_pie,startangle=90,counterclock=False,
        radius=1.1,pctdistance=1,textprops={'fontsize':10})[0]
    [patches[i].set_alpha(.7) for i in range(len(block_list))]

    plt.tight_layout()
    plt.savefig('image/y_distr/%s.pdf'%kind)
    plt.close()
#'''


''' block y abundance for all blocks
# parameters
top_fct = .9  # height of each curve
ref_fct = .9  # height of the reference line
y = np.arange(-64, y_max_mt+1)
fig_x = 9

# select non-zero blocks 
Ind = block_num != 0
block_demo = Block_all[Ind]
block_num_demo = block_num[Ind]
block_y_num_demo = block_y_num[Ind]

fig_y = len(block_num_demo)

# sort blocks by abundance
Ind = np.argsort(block_num_demo)
block_demo = block_demo[Ind]
block_num_demo = block_num_demo[Ind]
block_y_num_demo = block_y_num_demo[Ind]

# demo
plt.figure(figsize=(fig_x, fig_y))
plt.xlim(-64, y_max_mt)
plt.ylim(0, len(block_demo))
ax = plt.gca()

bottom = 0
for block, num, y_num in zip(block_demo, block_num_demo, block_y_num_demo):
    c = Color_all[block2id[block]]/255
    y_den = top_fct * y_num/max(y_num) + bottom

    plt.step(y, y_den, where='post', lw=1, color='k')
    plt.fill_between(y, bottom, y_den, step='post', alpha=0.5, color=c)
    plt.axhline(bottom, color='k', lw=.5)

    text = f'{block} ({num:,})'
    plt.text(-63, bottom+.05, text, fontsize=14, weight='bold')
    bottom += 1

# text
text = (f'MC {MC_version} overworld\nSeed: {MC_seed}\n'
        f'{shape_x}' + r'$\times$' + f'{shape_z} blocks')
t = plt.text(165, bottom-1, text, fontsize=16, weight='bold')
t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='gray',
                boxstyle='round,pad=.5'))

# xticks
ticks = np.arange(-60, y_max_mt, 10)
labels = [str(i) for i in ticks]
ax.set_xticks(ticks)
ax.set_xticklabels(labels)

# additional xticks
separation = 5 
x_off = -3
y_off = -.15
for y in np.arange(separation, bottom, separation):
    for tick, label in zip(ticks, labels):
        plt.text(tick+x_off, y+y_off, label, fontsize=10)

plt.grid(axis='x')
plt.xlabel('Height (block)')
plt.yticks([])
plt.tight_layout()
plt.savefig('image/y_distr/all.pdf')
plt.close()
#'''

#''' cumulative block y abundance for all blocks
# parameters
y = np.arange(-64, y_max_mt+1)

# select non-zero blocks 
Ind = block_num != 0
block_demo = Block_all[Ind]
block_num_demo = block_num[Ind]
block_y_num_demo = block_y_num[Ind]

# sort blocks by abundance
Ind = np.argsort(block_num_demo)[::-1]
block_demo = block_demo[Ind]
block_num_demo = block_num_demo[Ind]
block_y_num_demo = block_y_num_demo[Ind]

# place 'air' to the last [1, ..., 0]
Ind = np.concatenate((np.arange(1, len(block_demo)), [0]))
block_demo = block_demo[Ind]
block_num_demo = block_num_demo[Ind]
block_y_num_demo = block_y_num_demo[Ind]


# demo
plt.figure(figsize=(10, 6))
ax = plt.gca()

bottom = 0
for block, num, y_num in zip(block_demo, block_num_demo, block_y_num_demo):
    c = Color_all[block2id[block]]/255
    plt.fill_between(y, bottom, bottom+y_num, step='post', alpha=.8, fc=c,
                     ec='none')
    bottom += y_num

# text
# text = (f'MC {MC_version} overworld\nSeed: {MC_seed}\n'
#         f'{shape_x}' + r'$\times$' + f'{shape_z} blocks')
# t = plt.text(165, bottom-1, text, fontsize=16, weight='bold')
# t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='gray',
#                 boxstyle='round,pad=.5'))

# xticks
ticks = np.arange(-60, y_max_mt, 10)
labels = [str(i) for i in ticks]
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
    
plt.xlim(-64, 150)
plt.ylim(0, bottom[0])

plt.grid()
plt.xlabel('Height (block)')
plt.ylabel('Number of blocks')
plt.tight_layout()
plt.savefig('image/y_distr/cumulative.pdf')
plt.close()
#'''


''' abs/stone abundance for each block

# parameters
y = np.arange(-64,y_max_mt+1)
Block_kind = {
    'ore': ['coal_ore','copper_ore','iron_ore','redstone_ore','gold_ore',
        'lapis_ore','diamond_ore','emerald_ore'],
}


for kind in Block_kind:
    if not os.path.isdir('image/y_distr/%s'%kind):
        os.mkdir('image/y_distr/%s'%kind)

    block_list = Block_kind[kind]

    # demo
    for block in block_list:
        # if block not in ['iron_ore']:
        #     continue

        block_d = 'deepslate_'+block

        c = Color_all[block2id[block]]/255
        c_d = Color_all[block2id[block_d]]/255

        y_num = World['block_y_num'][block]
        y_num_d = World['block_y_num'][block_d]
        y_num_all = y_num+y_num_d

        y_s = World['block_y_num']['stone']+World['block_y_num']['deepslate']

        abs_abun = y_num/4096**2*1e3 # [per thousand]
        abs_abun_d = y_num_d/4096**2*1e3 
        sto_abun = y_num/(y_s+y_num)*1e3
        sto_abun_d = y_num_d/(y_s+y_num_d)*1e3

        if block=='emerald_ore':
            abs_abun *= 1e3 # [ppm]
            abs_abun_d *= 1e3
            sto_abun *= 1e3
            sto_abun_d *= 1e3


        # x limits
        ind_non0 = np.where(y_num_all)[0]-64
        x_min = ind_non0[0]-3
        x_max = ind_non0[-1]+3


        # demo
        plt.figure(figsize=(8,4))

        plt.step(y,sto_abun,where='post',lw=1,color='C3',
            label='Stone abundance')
        plt.step(y,sto_abun_d,where='post',lw=1,color='darkred')

        plt.step(y,abs_abun,where='post',lw=1,color='k',label='Abs. abundance')
        plt.fill_between(y,0,abs_abun,step="post",alpha=0.5,color=c)

        plt.step(y,abs_abun_d,where='post',lw=1,color='k')
        plt.fill_between(y,0,abs_abun_d,step="post",alpha=0.5,color=c_d)

        ax = plt.gca()

        # text
        text = ('MC %s Overworld\nSeed: %d\n%d'%(MC_version,MC_seed,
            512*(rx2-rx1+1)) + r'$\times$' + '%d blocks'%(512*(rz2-rz1+1)))
        t = plt.text(.68,.75,text,transform=plt.gca().transAxes,fontsize=14,
            weight='bold')
        t.set_bbox(dict(facecolor='w',alpha=0.5,edgecolor='gray',
            boxstyle='round,pad=.5'))

        plt.legend()

        # limits
        plt.xlim(x_min,x_max)
        plt.ylim(0,plt.ylim()[1])

        # xticks
        xticks = np.arange(10*np.ceil(x_min/10),x_max,10)
        xlabels = ['%d'%i for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        plt.xlabel('Height (block)')

        if block=='emerald_ore':
            plt.ylabel('Abundance (ppm)')
        else:
            plt.ylabel('Abundance (‰)')
        plt.grid()

        plt.tight_layout()
        plt.savefig('image/y_distr/%s/abun_%s.pdf'%(kind,block))
        plt.close()
#'''


''' average area to find one block (unmodified)

# parameters
y = np.arange(-64,320)


# demo
plt.figure(figsize=(9,5))
plt.xlim(-65,175)   
plt.yscale('log')
for block in Block_ore:
    c = Para_blk[block]['color']/255
    y_num = World['block_y_num'][block]
    y_num_d = World['block_y_num']['deepslate_'+block]

    y_num_all = y_num+y_num_d
    size_blk = ((rx2-rx1+1)*(rz2-rz1+1)*512**2/y_num_all)**.5
    size_blk_all = ((rx2-rx1+1)*(rz2-rz1+1)*512**2/sum(y_num_all))**.5

    label = block.split('_')[0].capitalize()
    plt.plot(y,size_blk,lw=2,color=c,label=label)

# text
text = ('MC %s Overworld\nSeed: %d\n%d'%(MC_version,MC_seed,512*(rx2-rx1+1))
    + r'$\times$' + '%d blocks'%(512*(rz2-rz1+1)))
t = plt.text(.75,.1,text,transform=plt.gca().transAxes,fontsize=14,
    weight='bold')
t.set_bbox(dict(facecolor='w',alpha=0.5,edgecolor='gray',
    boxstyle='round,pad=.5'))

plt.grid()
plt.yticks([10,100,1000,1e4])
plt.legend()
plt.xlabel('Height (block)')
plt.ylabel('Size of square (block)')
plt.tight_layout()
plt.savefig('image/ave_area_one_block.pdf')
plt.close()
#'''


''' world map
# parameters
panel_x = 10 # [in]
marg1 = .5 
marg2 = .1

panel_y = panel_x*shape_z/shape_x
dpi = shape_x/panel_x
fig_x = marg1+marg2+panel_x
fig_y = marg1+marg2+panel_y

# map range 
bx1 = 512*rx1
bx2 = 512*(rx2+1)-1
bz1 = 512*rz1
bz2 = 512*(rz2+1)-1

plt.figure(figsize=(fig_x,fig_y))
plt.axes([marg1/fig_x,marg1/fig_y,panel_x/fig_x,panel_y/fig_y])
plt.imshow(World['RGBA'],extent=[bx1,bx2,bz2,bz1],interpolation='none')
ax = plt.gca()

# text
text = 'MC %s Overworld\nSeed: %d'%(MC_version,MC_seed)
t = plt.text(.8,.95,text,transform=plt.gca().transAxes,fontsize=14,
    weight='bold')
t.set_bbox(dict(facecolor='w',alpha=0.5,edgecolor='gray',
    boxstyle='round,pad=.5'))

# grids of regions 
xticks = 512*np.arange(rx1,rx2+2)
xlabels = ['%d'%i for i in xticks]
yticks = 512*np.arange(rz1,rz2+2)
ylabels = ['%d'%i for i in yticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
plt.grid()

# grids of chunks
for bx in range(bx1,bx2,16):
    plt.axvline(bx,lw=.1,color='gray')
for bz in range(bz1,bz2,16):
    plt.axhline(bz,lw=.1,color='gray')

plt.savefig('image/map.png',dpi=dpi)
plt.close()
#'''


''' height map
# parameters
panel_x = 10 # [in]
marg1 = .5 
marg2 = 0

panel_y = panel_x*shape_z/shape_x
dpi = shape_x/panel_x
fig_x = marg1+marg2+panel_x + .5
fig_y = marg1+marg2+panel_y 

# map range 
bx1 = 512*rx1
bx2 = 512*(rx2+1)-1
bz1 = 512*rz1
bz2 = 512*(rz2+1)-1

plt.figure(figsize=(fig_x, fig_y))
plt.axes([marg1/fig_x, marg1/fig_y, panel_x/fig_x, panel_y/fig_y])
plt.imshow(surface_y-64, 'jet', extent=[bx1,bx2,bz2,bz1], interpolation='none',
           vmax=200)
plt.colorbar(fraction=0.046, pad=0.04)
ax = plt.gca()

# text
text = 'MC %s Overworld\nSeed: %d'%(MC_version,MC_seed)
t = plt.text(.8,.95,text,transform=plt.gca().transAxes,fontsize=14,
    weight='bold')
t.set_bbox(dict(facecolor='w',alpha=0.5,edgecolor='gray',
    boxstyle='round,pad=.5'))

# grids of regions 
xticks = 512*np.arange(rx1,rx2+2)
xlabels = ['%d'%i for i in xticks]
yticks = 512*np.arange(rz1,rz2+2)
ylabels = ['%d'%i for i in yticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels)
plt.grid()

# grids of chunks
for bx in range(bx1,bx2,16):
    plt.axvline(bx,lw=.1,color='gray')
for bz in range(bz1,bz2,16):
    plt.axhline(bz,lw=.1,color='gray')

plt.savefig('image/height.png',dpi=dpi)
plt.close()
#'''

''' height distribution
data = surface_y.flatten() - 64
bins = np.arange(-64, y_max_mt+1)

plt.figure()
plt.yscale('log')
plt.hist(data, bins, histtype='stepfilled', fc='whitesmoke', ec='k', lw=1)
plt.xlabel('Surface height')
plt.ylabel('Number of blocks')
plt.grid()
plt.tight_layout()
plt.savefig('image/height_distr.pdf')
plt.close()
#'''