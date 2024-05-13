'''
To illustrate colors.
'''

import sys
import colour
import matplotlib   
import numpy as np 
import matplotlib.pyplot as plt 

# sys.path.append('../module')
# import phunction


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'medium'


#''' blackbody color 

# parameters
T = np.linspace(500,12200,1000)  # [K]
aspect_r = 6 # width/heigh of image
fig_x = 11 # [in]

# derived parameters
T_c = T - 273.15
fig_y = 1.4*fig_x/aspect_r
ymax = (T.max()-T.min())/aspect_r

# blackbody XYZ
XYZ = np.array([colour.sd_to_XYZ(colour.sd_blackbody(t)) for t in T])
# XYZ /= np.max(XYZ, axis=1)[...,np.newaxis]

RGB = colour.XYZ_to_sRGB(XYZ)  # shape: (y, 3)
# RGB = colour.XYZ_to_RGB(XYZ, 'sRGB')

# demo of XYZ & RGB
plt.figure(figsize=(5,6))
plt.subplot(211)
plt.plot(T, XYZ[:,0], color='k', ls='-', label='X')
plt.plot(T, XYZ[:,1], color='k', ls='--', label='Y')
plt.plot(T, XYZ[:,2], color='k', ls=':', label='Z')
plt.legend()
plt.grid()
plt.gca().axes.xaxis.set_ticklabels([])
plt.subplot(212)
plt.plot(T, RGB[:,0], color='r', label='R')
plt.plot(T, RGB[:,1], color='g', label='G')
plt.plot(T, RGB[:,2], color='b', label='B')
plt.legend()
plt.grid()
plt.xlabel('Blackbody temperature (K)')
plt.tight_layout()
plt.savefig('image/bb_color_space.pdf')
plt.close()

# scaling
RGB /= np.max(RGB,axis=1)[...,np.newaxis]
RGB = RGB[np.newaxis]  # shape: (y,x,3), add y-axis

# demo of colors
plt.figure(figsize=(fig_x,fig_y))
plt.ylim(0,ymax)
plt.imshow(RGB,extent=(T.min(),T.max(),0,ymax))
# plt.imshow(Msk,extent=(T.min(),T.max(),0,.3*ymax),zorder=1)

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


T = phunction.m2T_MS(M) # [K], surface temperature
fig_y = 1.4*fig_x/aspect_r
ymax = (M.max()-M.min())/aspect_r

# RGB values
XYZ = np.array([colour.sd_to_XYZ(colour.sd_blackbody(t)) for t in T])
RGB = colour.XYZ_to_sRGB(XYZ)

# scaling
_max = np.max(RGB,axis=1)
RGB = RGB/_max[:,np.newaxis]
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


