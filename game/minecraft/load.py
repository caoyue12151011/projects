'''
Load Minecraft mca files. To save time, open multiple terminals & type 
'run load <rx>' (see the corresponding code).

Output
------
y_max_mt: max height of non-air blocks
r_rx_rz: np.ndarray of shape (y,z,x). cube of block ID's of a Minecraft region
'''
import sys
import dill
import anvil
import time
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

# parameters
dire = '/Users/yuecao/Documents/curseforge/minecraft/Instances/1.19.3/saves'
y_max_mt = 256  # max height of non-air blocks

#''' loading MC world data -----------------------------------------------------
t0 = time.time()

# parameters
rx1 = rx2 = int(sys.argv[1]) # region coordinates, 'run load -4'
rz1, rz2 = -4,3 

for rx in range(rx1,rx2+1):
    for rz in range(rz1,rz2+1):

        # chunk coordinates
        cx1 = 32*rx
        cx2 = 32*(rx+1)-1
        cz1 = 32*rz
        cz2 = 32*(rz+1)-1

        # load region files
        region = anvil.Region.from_file('%s/r.%d.%d.mca'%(dire,rx,rz))

        # region cube data, id=16 for air
        ID = np.full((y_max_mt+65,512,512),16,np.int16) 
            
        for cx in range(cx1,cx2+1): # chunk coordinates
            # if not cx==cx1:
            #     continue

            for cz in range(cz1,cz2+1):
                # if not cz==cz1:
                #     continue

                print('Loading world %s,%s chuck %s,%s.'%(rx,rz,cx,cz))

                try:
                    chunk = anvil.Chunk.from_region(region,cx,cz)
                except:
                    print('Error: chunk %d,%d not found in region %d,%d.'%(cx,
                        cz,rx,rz))

                else: # load blocks
                    fail = 0
                
                    for bx_r in range(16): # relative coords in a chunk
                        bx_i = 16*(cx-cx1) + bx_r # relative coords in region 

                        for bz_r in range(16):
                            bz_i = 16*(cz-cz1) + bz_r

                            for by in range(-64,y_max_mt+1):
                                by_i = by+64
                            
                                try:
                                    block = chunk.get_block(bx_r,by,bz_r)
                                except:
                                    fail += 1
                                else:
                                    ID[by_i,bz_i,bx_i] = block2id[block.id]

                    if fail>0:
                        fail *= 100/256/384
                        print('Error: %.1f%% of chunk %d,%d not loaded.'%(fail,
                            cx,cz))
        # save
        dill.dump(ID,open('variable/regions/r_%d_%d.p'%(rx,rz),'wb'))


t = (time.time()-t0)/60 
print('All done. Time = %.1f min.'%t)
#'''

# save
dill.dump(y_max_mt, open('variable/y_max_mt.p','wb'))


# demo -------------------------------------------------------------------------

''' region map 

# parameters
y = 60+64
rx = -4 
rz = -4 

for fname in os.listdir('variable/regions'):
    if '.p' not in fname:
        continue 
    
    rx, rz = fname.split('.')[0].split('_')[1:]
    ID = dill.load(open('variable/regions/r_%s_%s.p'%(rx,rz),'rb'))

    # RGBA data 
    RGBA = np.zeros((512,512,4),np.int16)
    for x in range(512):
        for z in range(512):
            block = id2block[ID[y,z,x]]
            if 'air' not in block:
                RGBA[z,x] = Para_blk[block]['color']

    # demo
    plt.figure(figsize=(6,6))
    plt.imshow(RGBA,extent=[0,511,511,0],interpolation='none')
    plt.text(.1,.1,'y=%d'%(y-64),fontsize=12,transform=plt.gca().transAxes)
    plt.savefig('image/region_map/r%s_r%s.png'%(rx,rz))
    plt.tight_layout()
    plt.close()
#'''

