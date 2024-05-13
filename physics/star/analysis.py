import sys
import dill
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patheffects as path_effects

sys.path.append('../module')
import imf
import phunction
import circ_pack 


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


np.random.seed(0)


# load variables 
M_color = dill.load(open('variable/M_color.p','rb'))
RGB_color = dill.load(open('variable/RGB_color.p','rb'))


# generate circle of stars for further demo ....................................

# parameters
N_moninal = 100000 # nomical star number 
m_min = .079 # [Msun], default is .03 for 'kroupa' in imf module
m_max = 120 # [Msun], default is 120 for 'kroupa' in imf module
eta = 1.5 # width/height ratio of the canvas
figx = 8 # [in]

Data_star = {
    'N_moninal': N_moninal,
    'm_min': m_min,
    'm_max': m_max,
    'eta': eta,
    'figx': figx,
}

for massfunc in ['kroupa','kroupa_PMF']:

     # IMF function object
    IMF = None
    if massfunc=='kroupa_PMF':
        IMF = imf.Kroupa_PMF()
    elif massfunc=='kroupa':
        IMF = imf.Kroupa()

    # draw samples
    M_tot = (N_moninal*IMF.m_integrate(m_min,m_max)[0]/
        IMF.integrate(m_min,m_max)[0])
    M = imf.imf.make_cluster(M_tot,massfunc=massfunc,mmin=m_min,mmax=m_max)
    M = np.sort(M)
    N = len(M)

    # stellar properties
    L = phunction.m2L_MS(M) # [L_sun]
    RGB = np.transpose([np.interp(M,M_color,RGB_color[:,i]) for i in range(3)])
        # star color, 'RGB = SED2color(SED)' is time consuming 

    print('Number=%s\nNominal number=%s\nM_min=%.4f Msun\nM_max=%.2f\n'%(N,
        N_moninal,M.min(),M.max()))

    # cirle packing of mass
    R_m = (M/np.pi)**.5
    Xc_m, Yc_m, w_m, h_m = circ_pack.circ_pack(M,eta,.5)
    print('Mass circle packing finished')

    # cirle packing of luminosity
    R_l = (L/np.pi)**.5 # radii of the circles
    Xc_l, Yc_l, w_l, h_l = circ_pack.circ_pack(L,eta,.5)
    print('Luminosity circle packing finished')

    # save 
    name = 'Kroupa_PMF' if massfunc=='kroupa_PMF' else 'Kroupa_IMF'
    Data_star[name] = {
        'M': M,
        'L': L, 
        'RGB': RGB,
        'R_m': R_m,
        'Xc_m': Xc_m,
        'Yc_m': Yc_m,
        'w_m': w_m,
        'h_m': h_m,
        'R_l': R_l,
        'Xc_l': Xc_l,
        'Yc_l': Yc_l,
        'w_l': w_l,
        'h_l': h_l,
    }


# save 
dill.dump(Data_star,open('variable/Data_star.p','wb'))


