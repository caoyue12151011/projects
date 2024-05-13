import dill 
import sys
import numpy as np 
import matplotlib.pyplot as plt 
 
sys.path.append('/Users/yuecao/Documents/coding/module')
import phunction


# array of mass & RGB for interpolating 
M_color = np.logspace(-1.2,2.2,100)
RGB_color = phunction.mass2color(M_color)

# demo 
# plt.figure(figsize=(12,3))
# plt.xscale('log')
# plt.scatter(M_color,np.zeros(len(M_color)),color=RGB_color)
# plt.tight_layout()
# plt.show()


# save 
dill.dump(M_color,open('variable/M_color.p','wb'))
dill.dump(RGB_color,open('variable/RGB_color.p','wb'))



