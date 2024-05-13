import itertools
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

    y3 = (s23**2 - (s12**2+s23**2-s13**2)**2/4/s12**2)**.5
    x3 = (s13**2 - y3**2)**.5

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


#''' Apollonian gasket 

# parameters 
N = 1 # number of loops
Para = [[0,2/3**.5,1],[-1,-3**-.5,1],[1,-3**-.5,1],[0,0,-(2/3**.5+1)]] 
Para = [[0,2/3**.5,1],[-1,-3**-.5,1],[1,-3**-.5,1],[0,0,.154700538]] 
    # list of [x,y,r]
Sets = [Para.copy()] 


count = 0 
while count<N:
    Para_t = Sets[count]

    i_range = range(1,4) 
    if count==0:
        i_range = range(4)

    for i in i_range:
        para1,para2,para3,para4 = Para_t[i:]+Para_t[:i]
        para4a = other_fourth(para1,para2,para3,para4)

        Para.append(para4a)
        Sets.append([para1,para2,para3,para4a])
    count += 1

Para = np.array(Para)
print('# of circles: %d.'%len(Para))


# demo
fig = plt.figure(figsize=(6.5,6.5))
ax = plt.gca()
ax = fig.add_axes([0,0,1,1])
plt.xlim(-2.2,2.2)
plt.ylim(-2.2,2.2)
for x, y, r in Para:
    ax.add_artist(plt.Circle((x,y),r,fc='none',ec='k',lw=.3,alpha=1))
plt.xticks([])
plt.yticks([])
# plt.savefig('image/Apollonian_gasket.pdf')
plt.show()
#'''


''' Apollonian gasket curvature distribution

# parameters 
N = 300000 # number of loops
N_c = 3*N+5 # number of circles for N>0, also len(K)
# k0 = np.array([1,1,1,-1/(2/3**.5+1)])
k0 = np.array([1,2,3,-6])

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

K = np.array(K)
print('# of circles: %d.'%len(K))


# demo
data = 1/np.abs(K)
bins = np.logspace(np.log10(data.min()),np.log10(data.max()),
    int(len(data)**.5/5))

plt.figure(figsize=(6.5,5))
ax = plt.gca()
plt.xscale('log')
plt.yscale('log')
plt.hist(data,bins,histtype='step',color='k',density=True)
plt.text(.7,.9,'%d circles'%len(data),fontsize=14,transform=ax.transAxes)
plt.grid()
plt.xlabel('Radius')
plt.ylabel('Frequency density')
plt.tight_layout()
plt.savefig('image/Apollonian_gasket_k_distr.pdf')
plt.show()
#'''





