'''
To illustrate colors.
'''
import sys
import colour
import matplotlib   
import numpy as np 
import matplotlib.pyplot as plt 
from importlib import reload
from matplotlib.widgets import Slider, Button

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

''' dust color 
# parameters
lamda = np.linspace(360, 780, 421)
T = np.linspace(500, 12200, 500)  # [K]
beta = np.linspace(-2, 4, 300)
lgN = 17.5
Gamma = 100

# figure parameters 
figsize = (8, 5)
fx_p = .2  # fraction of the plotting panel
fy_p = .15
fw_p = .75
fh_p = .8
fx_s = .05
fy_s = .1
fw_s = .05
fh_s = .8

# spectrum
T = T[np.newaxis, ..., np.newaxis]  # (beta, T, lamda)
beta = beta[..., np.newaxis, np.newaxis]  # (beta, T, lamda)
I = 10**phunction.SED_dust(lamda/1e3, T, lgN, beta, Gamma)/lamda**2 
    # dI/d(lamda)

# RGB colors
RGB = np.full((I.shape[0], I.shape[1], 3), np.nan)
for i in range(I.shape[0]):
    for j in range(I.shape[1]):
        RGB[i,j] = phunction.SED2color(lamda, I[i,j])

# plotting panel
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([fx_p, fy_p, fw_p, fh_p])
im = plt.imshow(RGB, extent=[T.min(), T.max(), beta.min(), beta.max()],
                origin='lower', aspect='auto')
plt.grid()
plt.xlabel(r'$T\rm\ (K)$')
plt.ylabel(r'$\beta$')

# sliders 
sld = Slider(ax=fig.add_axes([fx_s,fy_s,fw_s,fh_s]),
             label=r'$\lg(N_{\rm H_2}/\rm cm^{-2})$',
             valmin=15, valmax=25, valinit=lgN, orientation='vertical')

# The function to be called anytime a slider's value changes
def update(val):
    # new slider values
    lgN = sld.val

    I = 10**phunction.SED_dust(lamda/1e3, T, lgN, beta, Gamma) / lamda**2

    # RGB colors
    RGB = np.full((I.shape[0], I.shape[1], 3), np.nan)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            RGB[i,j] = phunction.SED2color(lamda, I[i,j])

    # upload plots
    im.set_data(RGB)
    fig.canvas.draw_idle()

# register the update function with each slider
sld.on_changed(update)
plt.savefig('image/dust_color.pdf')
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


