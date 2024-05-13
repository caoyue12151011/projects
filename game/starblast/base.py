'''
To analyze distances of bases in starblast.io.
'''
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def calc_dist(x1, y1, x2, y2):
    return ( (x1-x2)**2 + (y1-y2)**2 )**.5



# parameters
M = 40  # base size, 80 for team mode
T = 60  # [min], orbital period of bases
N = 2  # number of bases
ind_map = np.arange(9)  # indices of warped maps, see bases.key
ind_map_no_self = np.array([0, 1, 2, 3, 5, 6, 7, 8])
video_len = 5  # [sec], video length
disp_len = 60  # [min], time range to be displayed
video_fps = 35

# calculation -----------------------------------------------------------------

t = np.linspace(0, disp_len, video_len*video_fps)  # [min]
speed_fct = 60*disp_len/video_len

ind_base = np.arange(N-1)  # no self referring
color_base = [f'C{i}' for i in ind_base]
r = M/2/2**.5

# the self base
theta0 = 2*np.pi * t/T
x0 = r * np.cos(theta0)
y0 = r * np.sin(theta0)

# the other bases
theta = 2*np.pi * (t/T + (ind_base+1)[...,np.newaxis]/N)  
    # shape = (ind_base,t)
ind_k, ind_j = divmod(ind_map, 3)
x = r * np.cos(theta) + (ind_j[...,np.newaxis,np.newaxis]-1)*M
y = r * np.sin(theta) + (ind_k[...,np.newaxis,np.newaxis]-1)*M  
      # (ind_map,ind_base,t)

distance = calc_dist(x0, y0, x, y)

# find smallest distance 
min_distance = np.nanmin(distance, axis=0)  # (ind_base,t)
ind_map_min = np.nanargmin(distance, axis=0)  # (ind_base,t)
ind_k_min, ind_j_min = divmod(ind_map_min, 3)


# demo ------------------------------------------------------------------------

#''' min distance vs time
av_distance = np.mean(min_distance, axis=0)

plt.figure()
plt.plot(0,0)
for i in ind_base:
    plt.plot(t, min_distance[i], color=color_base[i], lw=1.5, 
             label=f'To base {i+1}')
plt.plot(t, av_distance, color='k', lw=1.5, ls='--', label='Mean')
plt.legend()
plt.grid()
plt.xlabel('Time (min)')
plt.ylabel('Distance to bases')
plt.tight_layout()
plt.savefig(f'image/base/min_distance_{N}bases.pdf')
plt.close()
#'''

''' base animation 

# parameters 
s00 = 150
s = 60
c0 = 'w'  # color of my base

X0 = x0 + (ind_j[...,np.newaxis]-1)*M  # (ind_map, t)
Y0 = y0 + (ind_k[...,np.newaxis]-1)*M

# demo
fig = plt.figure(figsize=(6,6))
ax = plt.gca()
ax.set_facecolor([.2]*3)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.xlim(-1.5*M, 1.5*M)
plt.ylim(-1.5*M, 1.5*M)

# map borders
for i in range(-2,3):
    plt.axvline((i-.5)*M, color='gray', lw=.5)
    plt.axhline((i-.5)*M, color='gray', lw=.5)

# bases
dot00 = plt.scatter(X0[4,0], Y0[4,0], fc=c0, ec='w',s=s00)
dot0 = plt.scatter(X0[ind_map_no_self,0], Y0[ind_map_no_self,0], fc=c0, s=s)
dots = []
for i in ind_base:
    dots.append(plt.scatter(x[:,i,0], y[:,i,0], fc=color_base[i], s=s))

# connection line to nearest bases
ls = []
for i in ind_base:
    ls.append(plt.plot([x[ind_map_min[i,0],i,0], x0[0]], 
                       [y[ind_map_min[i,0],i,0], y0[0]], 
                       color=color_base[i], lw=1)[0])

def make_frame(t1):
    t1 *= speed_fct/60  # animation time to simulation time, sec to min
    ind_t = int(t1/(t[1]-t[0]))

    coord00 = [X0[4,ind_t], Y0[4,ind_t]]
    coord0 = np.transpose([X0[ind_map_no_self,ind_t], 
                           Y0[ind_map_no_self,ind_t]])
    dot00.set_offsets(coord00)
    dot0.set_offsets(coord0)

    for i in ind_base:
        dots[i].set_offsets(np.transpose([x[:,i,ind_t], y[:,i,ind_t]]))

    for i in ind_base:
        ls[i].set_data([[x[ind_map_min[i,ind_t],i,ind_t], x0[ind_t]], 
                        [y[ind_map_min[i,ind_t],i,ind_t], y0[ind_t]]])

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration=video_len)
animation.write_gif(f'image/base/base_{N}bases.gif', fps=video_fps)
#'''
