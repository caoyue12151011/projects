'''
To analyze & demo the Minecraft world. 

Output
------
World: dict of a Minecraft cuboid, derived from the r_x_x.p files
    'MC_seed'
    'MC_version'
    'rx1': region coord ranges of the world
    'rx2'
    'rz1'
    'rz2'
    'shape_y': world shape in blocks
    'shape_z'
    'shape_x'

    'surface_RGBA': RGBA of surface blocks, (shape_z,shape_x,4)
    'surface_y': y of surface blocks, (shape_z,shape_x)
    'block_num': 1darray of len block_id, sorted by block Ids. Block numbers
    'block_y_num': np.ndarray of shape (block_id, y), sorted by block Ids.
                   block numbers in each level
Notes
-----
    # ores
    # 'coal_ore','deepslate_coal_ore',
    # 'copper_ore','deepslate_copper_ore',
    # 'iron_ore','deepslate_iron_ore',
    # 'gold_ore','deepslate_gold_ore',
    # 'redstone_ore','deepslate_redstone_ore',
    # 'diamond_ore','deepslate_diamond_ore',
    # 'lapis_ore','deepslate_lapis_ore',
    # 'emerald_ore','deepslate_emerald_ore',

    # trees
    # 'acacia_log','birch_log','dark_oak_log','jungle_log','mangrove_log',
    # 'oak_log','spruce_log','bamboo',

    # flowers
    # 'allium','azalea','azure_bluet','blue_orchid','cornflower','dandelion',
    # 'dead_bush','lilac_bottom','lily_of_the_valley','lily_pad','orange_tulip',
    # 'oxeye_daisy','peony_bottom','pink_tulip','poppy','red_tulip',

    # vegetations
    # 'melon','pumpkin','red_mushroom'

    # minerals
    # 'amethyst_block','andesite','deepslate','diorite','dripstone_block',
    # 'granite','ice','magma_block','red_sandstone','sandstone','stone',
    # 'terracotta','tuff','water',

    # # dirt
    # 'clay','coarse_dirt','dirt','gravel','mud','mycelium','podzol','red_sand',
    # 'sand','sculk','slime_block','snow_block',

    # 'amethyst_cluster',
'''
import os
import time
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
block2id = dill.load(open('variable/block2id.p','rb'))
Block_all = dill.load(open('variable/Block_all.p','rb'))
Color_all = dill.load(open('variable/Color_all.p','rb'))
y_max_mt = dill.load(open('variable/y_max_mt.p','rb'))

# constants
Yield = { # mean drops without Fortune enhancement
    'coal_ore': 1,
    'copper_ore': 3.5, # 2-5, 1406/400~3.5
    'iron_ore': 1,
    'redstone_ore': 4.5, # 4-5, 1801/400~4.5
    'gold_ore': 1,
    'lapis_ore': 6.5, # 4–9, 1972/300~6.5
    'diamond_ore': 1,
    'emerald_ore': 1,
}
Durability =  {
    'wood': 59,
    'stone': 131,
    'iron': 250,
    'gold': 32,
    'diamond': 1561,
    'netherite': 2031,
}

# world parameters 
MC_seed = 1
MC_version = '1.19.3'

# find r-coord range
rx1 = np.inf 
rx2 = -np.inf 
rz1 = np.inf 
rz2 = -np.inf 
for fname in os.listdir('variable/regions'):
    if '.p' in fname:
        fname = fname.split('.')[0].split('_')
        rx = int(fname[1])
        rz = int(fname[2])
        if rx1>rx: 
            rx1 = rx
        if rx2<rx: 
            rx2 = rx
        if rz1>rz: 
            rz1 = rz
        if rz2<rz: 
            rz2 = rz

# world cube shape
shape_y = y_max_mt + 65
shape_z = 512*(rz2-rz1+1)
shape_x = 512*(rx2-rx1+1)

World = {
    'MC_seed': MC_seed,
    'MC_version': MC_version,
    'rx1': rx1,
    'rx2': rx2,
    'rz1': rz1,
    'rz2': rz2,
    'shape_y': shape_y,
    'shape_z': shape_z,
    'shape_x': shape_x,
}

# analysis --------------------------------------------------------------------

surface_RGBA = np.zeros((shape_z,shape_x,4), np.uint8) 
surface_y = np.zeros((shape_z,shape_x), np.uint8)  # [0-255]
block_num = np.zeros(len(Block_all), int)
block_y_num = np.zeros((len(Block_all),shape_y), int)

# gather r_x_x.p data
t0 = time.time()
for rx in range(rx1,rx2+1):
    for rz in range(rz1,rz2+1):
        # block id cube
        ID = dill.load(open('variable/regions/r_%d_%d.p'%(rx,rz),'rb'))

        # block coord in World
        bx1 = 512*(rx-rx1)
        bz1 = 512*(rz-rz1)

        # block counter
        for Id in range(len(Block_all)):
            msk = ID==Id
            block_num[Id] += np.sum(msk)
            block_y_num[Id] += np.sum(msk, axis=(1,2))

        # surface 
        Msk = (ID==16)|(ID==137)|(ID==854)  # air, cave_air, void_air
        dMsk = np.diff(Msk, axis=0) # (z,x)
        for bx_r in range(512): 
            bx = bx1 + bx_r 
            for bz_r in range(512):
                bz = bz1 + bz_r
                ind = np.where(dMsk[:,bz_r,bx_r]==1)[0][-1] 
                Id = ID[ind, bz_r, bx_r]
                surface_y[bz, bx] = ind - 64
                surface_RGBA[bz, bx] = Color_all[Id]
        
        t = (time.time() - t0)/60  # [min]
        print(f'Region {rx},{rz} done. Time: {t:.1f} min.')

World['surface_RGBA'] = surface_RGBA
World['surface_y'] = surface_y
World['block_num'] = block_num
World['block_y_num'] = block_y_num

# outputs =====================================================================

# dills
dill.dump(World,open('variable/World.p', 'wb'))

# block num 
Ind = np.argsort(block_num)[::-1]
f = open('block_num.tab', 'w')
for ind in Ind:
    f.write(f'{Block_all[ind]:40s}  {block_num[ind]}\n')
f.close()

''' ore counter
print('In the %d*%d minecraft world:'%((rx2-rx1+1)*512,(rz2-rz1+1)*512))
for block in Yield:
    tmp = World['block_y_num']

    num = sum(tmp[block])
    num_d = sum(tmp['deepslate_'+block])
    num_tot = num+num_d 
    num_tot_drop = num_tot*Yield[block]

    print('{:s}\t{:,}\t{:,}\t{:,}'.format(block.split('_')[0],num_tot,
        num,num_d))
#'''


