'''
To illustrate colors.
'''
import sys
import colour
import matplotlib   
import numpy as np 
import matplotlib.pyplot as plt 
from importlib import reload

sys.path.append('../../../module')
from phunction import phunction
reload(phunction)

# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


''' blackbody color 
# parameters
T = np.linspace(500, 12200, 3000)  # [K]
aspect_r = 6 # width/heigh of image
fig_x = 11 # [in]

# derived parameters
T_c = T - 273.15
fig_y = 1.4*fig_x/aspect_r
ymax = (T.max()-T.min())/aspect_r

# derive the color
RGB = phunction.T2color_bb(T)  # (T,3)

# demo of RGB
plt.figure()
plt.plot(T, RGB[:,0], color='r', label='R')
plt.plot(T, RGB[:,1], color='g', label='G')
plt.plot(T, RGB[:,2], color='b', label='B')
plt.legend()
plt.grid()
plt.xlabel('Blackbody temperature (K)')
plt.tight_layout()
plt.savefig('image/bb_color_space.pdf')
plt.close()

RGB = RGB[np.newaxis]  # shape: (y,T,3), add y-axis

# demo of colors
plt.figure(figsize=(fig_x,fig_y))
plt.ylim(0,ymax)
plt.imshow(RGB,extent=(T.min(),T.max(),0,ymax))
plt.text(.03,.85,'sRGB gamut',fontsize=12,transform=plt.gca().transAxes)
plt.xlabel('Blackbody temperature (K)')
# plt.xlabel('Blackbody temperature '+ r'$\rm(^\circ C)$')
plt.yticks([])
plt.tight_layout()
plt.savefig('image/bb_color.pdf')
plt.close()
#'''

''' main-sequence star color 

# parameters
M = np.arange(.1,3,.005) # [Msun] stellar mass
aspect_r = 6 # width/heigh of image
fig_x = 9 # [in]

# derived parameters
fig_y = 1.4*fig_x/aspect_r
ymax = (M.max()-M.min())/aspect_r

# RGB values
RGB = phunction.m2color_MS(M)
RGB = RGB[np.newaxis] # add y axis

plt.figure(figsize=(fig_x,fig_y))
plt.ylim(0,ymax)
plt.imshow(RGB,extent=(M.min(),M.max(),0,ymax))
plt.text(.03,.85,'sRGB gamut',fontsize=12,transform=plt.gca().transAxes)
# plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(.2))

ticks = np.arange(M.min(),M.max(),.1)
plt.xticks(ticks)
plt.xlabel(r'$M_{\rm star}\ (M_{\odot})$')
plt.yticks([])
plt.tight_layout()
plt.savefig('image/color_star.pdf')
plt.close()
#'''


