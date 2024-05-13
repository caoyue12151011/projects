'''
Define some variables used later for the analysis.

Output
------
Block_all: all-block 1d array, sorted by Id's
Color_all: RGBA colors of all the blocks, sorted by Id's, n*4 array
block2id: block-to-id converter dict
Para_bio
    
    biome

        'color': RGBA values, 0-255 int
'''
import os 
import dill
import numpy as np
import matplotlib.image as mpimg

# load block file
f = open('data/block_list.txt','r')
f.readline()
data = f.readlines()
f.close()

# parameters 
dire = 'data/block_texture'
Works = np.array(['_fence_gate', '_pressure_plate', '_wood',
                  '_wall_sign', '_sign', '_wall', '_stairs', '_slab',
                  '_button', '_carpet', '_bed', '_banner', '_candle_cake'])

Block_all = []
Color_all = []
block2id = {}
for i in range(len(data)):
    block = data[i].split()[0]
    Block_all.append(block)
    block2id[block] = i

    # match block name with image filenames (hard-coded) ......................
    if 'waxed_' in block:
        block = block[6:]

    fname = None
    if os.path.isfile('%s/%s.png'%(dire,block)):
        fname = block

    elif os.path.isfile('%s/%s_top.png'%(dire,block)):
        fname = block+'_top'

    elif np.any([work in block for work in Works]):
        wk = Works[np.array([work in block for work in Works])][0]
        fn = block.split(wk)[0]

        if os.path.isfile('%s/%s.png'%(dire,fn)):
            fname = fn 
        elif os.path.isfile('%s/%s_planks.png'%(dire,fn)):
            fname = fn+'_planks'
        elif os.path.isfile('%s/%s_log.png'%(dire,fn)):
            fname = fn+'_log'
        elif os.path.isfile('%s/%ss.png'%(dire,fn)):
            fname = fn+'s'
        elif os.path.isfile('%s/%s_concrete.png'%(dire,fn)):
            fname = fn+'_concrete'
        else:
            print('Warning: no texture for block %s.'%block)
            continue 

    elif 'potted_' in block:
        fname = block[7:]

    elif 'infested_' in block:
        fname = block[9:]

    elif 'dripstone' in block:
        fname = 'dripstone_block'

    elif block in ['air','barrier','cave_air','light','structure_void',
                   'void_air']:
        fname = 'void'

    else:
        print('Warning: no texture for block %s.'%block)
        continue 

    # find block color
    img = mpimg.imread('%s/%s.png'%(dire,fname))*255  # 0-255

    rgba = None
    if img.shape[2]==4:
        msk = np.ones((img.shape[0],img.shape[1]))
        msk[img[:,:,3]==0.] = np.nan 
        img_c = img*msk[...,np.newaxis]
        rgba = np.nanmean(img_c,axis=(0,1)).astype(int)

    elif img.shape[2]==3:
        rgba = np.nanmean(img,axis=(0,1)).astype(int)
        rgba = np.array(list(rgba)+[255])

    Color_all.append(rgba)

Block_all = np.array(Block_all)
Color_all = np.array(Color_all)


# Para_bio 
Biome = ['Badlands','Beach','Desert','Ice Spikes','Jungle','Mushroom Fields',
         'Plains','Savanna','Snowy Plains','Swamp']
Para_bio = {}
for biome in Biome:
    rgba = np.mean(mpimg.imread('data/biome_color/%s.png'%biome),axis=(0,1))
    Para_bio[biome] = {'color': rgba}


# output
dill.dump(Block_all,open('variable/Block_all.p','wb'))
dill.dump(Color_all,open('variable/Color_all.p','wb'))
dill.dump(block2id,open('variable/block2id.p','wb'))
dill.dump(Para_bio,open('variable/Para_bio.p','wb'))