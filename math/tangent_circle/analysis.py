import itertools
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def circle_triplet_gen(r1,r2,r3):
    '''
    To generate three mutually-tangent circles. By default, the 1st circle 
    locates at the origin, the 2rd circle lies along the x+ axis, the 3nd 
    circle lies in y>0 region.  

    Inputs
    ------
    r1,r2,r3: radii of the 3 circles, can <0 

    Returns 
    -------
    x, y: 1d array of circle coordinates
    '''

    x1 = 0 
    y1 = 0

    x2 = r1+r2 
    y2 = 0 

    s12 = r1+r2
    s23 = r2+r3 
    s13 = r1+r3

    x3 = (s12**2 + s13**2 - s23**2)/2/s12 
    y3 = (s13**2 - x3**2)**.5

    return [x1,x2,x3], [y1,y2,y3]


def fourth_circle(para1,para2,para3):
    '''
    Find the position & radius of the 4th circle that are mutually tangent to 
    the given 3 circles using Descartes' theorem. See
    https://en.wikipedia.org/wiki/Descartes%27_theorem.

    Inputs
    ------
    para_i: [x,y,r] of the i-th circle. r can be negative for internally 
        tangent circles

    Returns
    -------
    para4: [x,y,r] of the two 4th circles
    '''

    x1, y1, r1 = para1
    x2, y2, r2 = para2
    x3, y3, r3 = para3
    k1, k2, k3 = 1/np.array([r1,r2,r3])

    z1 = complex(x1,y1)
    z2 = complex(x2,y2)
    z3 = complex(x3,y3)

    k4_1 = k1+k2+k3 + 2*(k1*k2+k2*k3+k3*k1)**.5
    k4_2 = k1+k2+k3 - 2*(k1*k2+k2*k3+k3*k1)**.5

    r4_1 = 1/k4_1
    r4_2 = 1/k4_2 

    tmp1 = z1*k1+z2*k2+z3*k3 
    tmp2 = 2*(z1*z2*k1*k2+z2*z3*k2*k3+z3*z1*k3*k1)**.5
    z4_1 =(tmp1+tmp2)/k4_1
    # z4_2 =(tmp1-tmp2)/k4_1
    # z4_3 =(tmp1+tmp2)/k4_2 
    z4_4 =(tmp1-tmp2)/k4_2 

    x4_1 = z4_1.real
    y4_1 = z4_1.imag
    # x4_2 = z4_2.real
    # y4_2 = z4_2.imag
    # x4_3 = z4_3.real
    # y4_3 = z4_3.imag
    x4_4 = z4_4.real
    y4_4 = z4_4.imag

    return [x4_1,y4_1,r4_1],[x4_4,y4_4,r4_2]
    # return [x4_1,y4_1,r4_1],[x4_2,y4_2,r4_1],[x4_3,y4_3,r4_2],[x4_4,y4_4,r4_2]


def other_fourth(para1,para2,para3,para4):
    ''' To find the other fourth circle in a circle quadruplet. Returns the 
    other para4. See fourth_circle for the definition of para.
    '''

    x1, y1, r1 = para1
    x2, y2, r2 = para2
    x3, y3, r3 = para3
    x4, y4, r4 = para4
    k1, k2, k3, k4 = 1/np.array([r1,r2,r3,r4])

    z1 = complex(x1,y1)
    z2 = complex(x2,y2)
    z3 = complex(x3,y3)
    z4 = complex(x4,y4)

    k4a = 2*(k1+k2+k3) - k4 
    kz4a = 2*(k1*z1 + k2*z2 + k3*z3) - k4*z4  
    z4a = kz4a/k4a

    x4a = z4a.real
    y4a = z4a.imag
    r4a = 1/k4a

    return [x4a,y4a,r4a]


def other_fourth_fast(k1,k2,k3,k4):
    # Same as other_fourth but focused on radii only.
    return 2*(k1+k2+k3) - k4 


# demo =========================================================================

''' circle quadruplet illustrator 

# parameters 
r1,r2,r3 = 1,.5,1.2

[x1,x2,x3], [y1,y2,y3] = circle_triplet_gen(r1,r2,r3)
res = fourth_circle([x1,y1,r1],[x2,y2,r2],[x3,y3,r3])
[x4_1,y4_1,r4_1],[x4_2,y4_2,r4_1],[x4_3,y4_3,r4_2],[x4_4,y4_4,r4_2] = res


plt.figure(figsize=(6,6))
ax = plt.gca()
plt.xlim(-1,4)
plt.ylim(-2,3)
ax.add_artist(plt.Circle((x1,y1),r1,fc='none',ec='k'))
ax.add_artist(plt.Circle((x2,y2),r2,fc='none',ec='k'))
ax.add_artist(plt.Circle((x3,y3),r3,fc='none',ec='k'))
ax.add_artist(plt.Circle((x4_1,y4_1),r4_1,fc='none',ec='r'))
# ax.add_artist(plt.Circle((x4_2,y4_2),r4_1,fc='none',ec='g'))
# ax.add_artist(plt.Circle((x4_3,y4_3),r4_2,fc='none',ec='r',ls='--'))
ax.add_artist(plt.Circle((x4_4,y4_4),r4_2,fc='none',ec='g',ls='--'))
plt.tight_layout()
plt.show()
#'''


''' Apollonian gasket 

# parameters 
N = 30000 # number of loops
r1,r2,r3 = 1,1,1 # radii of the three circles

N_c = 3*N+5 # number of circles for N>0, also len(Para)

# coordinates of the four circles
[x1,x2,x3], [y1,y2,y3] = circle_triplet_gen(r1,r2,r3)
x4,y4,r4 = fourth_circle([x1,y1,r1],[x2,y2,r2],[x3,y3,r3])[0] 
                                                # any one of the two

# parameters for seed circles 
Para = np.full((N_c,3),np.nan)
Para[:4] = np.array([[x1,y1,r1],[x2,y2,r2],[x3,y3,r3],[x4,y4,r4]])

Sets = np.full((N_c-3,4,3),np.nan)
Sets[0] = Para[:4]


count = 0 
count_c = 1
while count<N:
    Para_t = list(Sets[count])

    i_range = range(1,4) 
    if count==0:
        i_range = range(4)

    for i in i_range:
        para1,para2,para3,para4 = Para_t[i:]+Para_t[:i]
        para4a = other_fourth(para1,para2,para3,para4)

        Para[count_c+3] = para4a
        Sets[count_c] = np.array([para1,para2,para3,para4a])
        count_c += 1

    count += 1
print('# of circles: %d.'%len(Para))


# determine coordinate limits 
R = np.abs(Para[:,2])
ind = np.argmax(R) 
r_max = max(R)
x_ind = Para[ind,0]
y_ind = Para[ind,1]
x_min = x_ind - 1.05*r_max
x_max = x_ind + 1.05*r_max
y_min = y_ind - 1.05*r_max
y_max = y_ind + 1.05*r_max


# circle colors
c_value = np.log10(R)
c_value = (c_value-c_value.min())/(c_value.max()-c_value.min())
colors = matplotlib.cm.get_cmap('coolwarm')(c_value)

# text 
fname = 'A_gasket_%s_%s_%s.png'%(r1,r2,r3)

# demo
fig = plt.figure(figsize=(6.5,6.5))
ax = fig.add_axes([0,0,1,1])
ax.axis('off')
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
# plt.scatter(0,0,s=5,color='r')
for i in range(len(Para)):
    x,y,r = Para[i]
    ax.add_artist(plt.Circle((x,y),r,fc='none',ec='k',lw=.3,alpha=1))
plt.xticks([])
plt.yticks([])
plt.savefig('image/'+fname,dpi=700)
plt.close()
#'''


''' Apollonian gasket curvature distribution

# parameters 
N = 300000 # number of loops
N_c = 3*N+5 # number of circles for N>0, also len(K)
r1, r2, r3 = 1,1,1

# 4th circle 
[x1,x2,x3], [y1,y2,y3] = circle_triplet_gen(r1,r2,r3)
x4,y4,r4 = fourth_circle([x1,y1,r1],[x2,y2,r2],[x3,y3,r3])[0] 

k0 = 1/np.array([r1,r2,r3,r4])

K = np.full(N_c,np.nan)
K[:4] = k0

Sets = np.full((N_c-3,4),np.nan)
Sets[0] = k0


count = 0 
count_c = 1
while count<N:
    K_t = list(Sets[count])

    i_range = range(1,4) 
    if count==0:
        i_range = range(4)

    for i in i_range:
        k1,k2,k3,k4 = K_t[i:]+K_t[:i]
        k4a = other_fourth_fast(k1,k2,k3,k4)

        K[count_c+3] = k4a 
        Sets[count_c] = np.array([k1,k2,k3,k4a])
        count_c += 1

    count += 1
print('# of circles: %d.'%len(K))


# text 
fname = 'R_distr_%s_%s_%s.pdf'%(r1,r2,r3)
text = '%d circles\n'%N_c + r'$r_1,r_2,r_3=%s,%s,%s$'%(r1,r2,r3)

# demo
data = 1/np.abs(K)
bins = np.logspace(np.log10(data.min()),np.log10(data.max()),
    int(len(data)**.5))

plt.figure(figsize=(6.5,5))
ax = plt.gca()
plt.xscale('log')
plt.yscale('log')
plt.hist(data,bins,histtype='step',color='k',density=True)
plt.text(.6,.8,text,fontsize=14,transform=ax.transAxes)
plt.grid()
plt.xlabel('Radius')
plt.ylabel('Frequency density')
plt.tight_layout()
plt.savefig('image/'+fname)
plt.close()
#'''





