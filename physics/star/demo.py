import sys
import dill
import colour
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patheffects as path_effects

sys.path.append('../module')
import imf 
import phunction 


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


# load variables
Data_star = dill.load(open('variable/Data_star.p','rb'))


# demo -------------------------------------------------------------------------

''' stellar distributions 

# parameters
m_bin_str = ['0.08','0.2','0.5','1','2','4','8','16','32','64','120']
massfunc = 'kroupa'
alpha = .3
lw = 1

# derived parameters
m_bin = np.array([float(i) for i in m_bin_str])
m_cen = (m_bin[1:]*m_bin[:-1])**.5 
rgb_cen = phunction.mass2color(m_cen)
m_interval = np.transpose([m_bin[:-1],m_bin[1:]])

# IMF function object
IMF = None
if massfunc=='kroupa_PMF':
    IMF = imf.Kroupa_PMF()
elif massfunc=='kroupa':
    IMF = imf.Kroupa()

# number fraction 
N_tot = IMF.integrate(m_bin[0],m_bin[-1])[0]
pN_cen = np.array([IMF.integrate(m_bin[i],m_bin[i+1])[0]/N_tot*100 
    for i in range(len(m_cen))])

# mass fraction 
M_tot = IMF.m_integrate(m_bin[0],m_bin[-1])[0]
pM_cen = np.array([IMF.m_integrate(m_bin[i],m_bin[i+1])[0]/M_tot*100 
    for i in range(len(m_cen))])

# demo
plt.figure(figsize=(6,4))
plt.xscale('log')
plt.plot(0,0)
ax = plt.gca()

for i in range(len(m_cen)):
    ax.add_patch(plt.Rectangle((m_bin[i],-10),m_bin[i+1]-m_bin[i],pN_cen[i]+10,
        fc=(69/255,144/255,194/255,alpha),ec='w',lw=lw,zorder=2))
    ax.add_patch(plt.Rectangle((m_bin[i],-10),m_bin[i+1]-m_bin[i],pM_cen[i]+10,
        fc=(252/255,141/255,47/255,alpha),ec='w',lw=lw,zorder=1))
    plt.text(m_cen[i]/1.2,pN_cen[i]+.5,'%.1f'%pN_cen[i],color='C0',
        fontsize=12,zorder=2)
    plt.text(m_cen[i]/1.2,pM_cen[i]+.5,'%.1f'%pM_cen[i],color='C1',
        fontsize=12,zorder=1)

# title 
title = None
if massfunc=='kroupa_PMF':
    title = "Kroupa's PMF"
elif massfunc=='kroupa':
    title = "Kroupa's IMF"
title += '\n'+r'$%.2f-%s\ M_{\odot}$'%(m_bin[0],m_bin[-1])
plt.text(.4,.85,title,transform=ax.transAxes,fontsize=14,weight='bold')

plt.xlim((m_bin[0],m_bin[-1]))
plt.ylim(0,1.1*plt.ylim()[1])
ax.add_patch(plt.Rectangle((-2,0),1,1,fc='C0',ec='w',lw=lw,alpha=alpha,
    label='Number'))
ax.add_patch(plt.Rectangle((-2,0),1,1,fc='C1',ec='w',lw=lw,alpha=alpha,
    label='Mass'))
plt.legend()

plt.minorticks_off()
ax.set_xticks(m_bin)
ax.set_xticklabels(m_bin_str)
plt.xlabel('Mass '+r'$(M_\odot)$')
plt.ylabel('Fractions (%)')
plt.tight_layout()
plt.savefig('image/distribution/fraction_%s.pdf'%massfunc)
plt.close()
#'''


''' Star distributions & colors

# parameters
N_moninal = Data_star['N_moninal'] 
m_min = Data_star['m_min']
m_max = Data_star['m_max']
eta = Data_star['eta'] # width/height ratio of the canvas
figx = Data_star['figx'] # [in]

for name in ['Kroupa_PMF','Kroupa_IMF']:
    tmp = Data_star[name]

    # load data 
    M = tmp['M']
    L = tmp['L']
    RGB = tmp['RGB']
    R_m = tmp['R_m']
    Xc_m = tmp['Xc_m']
    Yc_m = tmp['Yc_m']
    w_m = tmp['w_m']
    h_m = tmp['h_m']
    R_l = tmp['R_l']
    Xc_l = tmp['Xc_l']
    Yc_l = tmp['Yc_l']
    w_l = tmp['w_l']
    h_l = tmp['h_l']

    N = len(M)

    # title 
    title_m0 = "Kroupa's PMF" if name=='Kroupa_PMF' else "Kroupa's IMF"
    title_m += '\n'+r'$%.2f-%s\ M_{\odot}$'%(m_min,m_max)
    title_l0 = "Kroupa's PLF" if name=='Kroupa_PMF' else "Kroupa's ILF"
    title_l += '\n'+r'$%.2f-%s\ M_{\odot}$'%(m_min,m_max)

    # MF demo ..................................................................

    plt.figure(figsize=(figx,figx/eta))
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    plt.axis('equal')
    plt.xlim(0,w_m)
    plt.ylim(0,h_m)
    ax = plt.gca()
    ax.set_facecolor('w')

    # circles
    for i in range(N):
        ax.add_artist(plt.Circle((Xc_m[i],Yc_m[i]),R_m[i],fc=RGB[i],ec=None))

    # white space for legends
    plt.axvspan(0,.18*w_m,color='w',alpha=.7,transform=ax.transAxes,zorder=40)

    # texts
    plt.text(.01,.92,title_m,transform=ax.transAxes,fontsize=14,zorder=45,
        weight='bold',bbox=dict(fc='w',ec='none',alpha=.7))

    # scales
    base = 2.
    ind1 = int( np.ceil(np.log(M.min())/np.log(base)) )
    ind2 = int( np.floor(np.log(M.max())/np.log(base)) )
    M_sc = base**np.arange(ind1,ind2+1)
    R_sc = (M_sc/np.pi)**.5
    T_sc = phunction.m2T_MS(M_sc)
    XYZ_sc = np.array([colour.sd_to_XYZ(colour.sd_blackbody(t)) for t in T_sc])
    RGB_sc = colour.XYZ_to_sRGB(XYZ_sc)

    _max = np.max(RGB_sc,axis=1)
    RGB_sc = RGB_sc/_max[:,np.newaxis]
    RGB_sc[RGB_sc<0.] = 0.

    y_sc = np.linspace(.05,.85,len(M_sc))*h_m
    for i in range(len(M_sc))[::-1]:
        ax.add_artist(plt.Circle((.05*w_m,y_sc[i]),R_sc[i],fc=RGB_sc[i],ec='k',
            lw=.3,zorder=43,alpha=1))
        plt.text(.08*w_m,y_sc[i]-0.01*h_m,str(M_sc[i])+r'$\ M_{\odot}$',
            fontsize=12,zorder=45)

    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    name_f = '_'.join(title_m0.split("'s "))
    plt.savefig('image/distribution/dots_%s_N%s.pdf'%(name_f,N_moninal))
    plt.close()

    # LF demo ..................................................................

    plt.figure(figsize=(figx,figx/eta))
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    plt.axis('equal')
    plt.xlim(0,w_l)
    plt.ylim(0,h_l)
    ax = plt.gca()
    ax.set_facecolor('w')

    # circles
    for i in range(N):
        ax.add_artist(plt.Circle((Xc_l[i],Yc_l[i]),R_l[i],facecolor=RGB[i],
            ec=None))

    # white space for legends
    plt.axvspan(0,.18*w_l,color='w',alpha=.7,transform=ax.transAxes,zorder=40)

    # texts
    plt.text(.01,.92,title_l,transform=ax.transAxes,fontsize=14,zorder=45,
        weight='bold',bbox=dict(facecolor='w',edgecolor='none',alpha=.7))

    # scales
    base = 10.
    ind1 = int( np.ceil(np.log(L.min())/np.log(base)) )
    ind2 = int( np.floor(np.log(L.max())/np.log(base)) )
    L_sc = base**np.arange(ind1,ind2+1)
    M_sc = phunction.L2m_MS(L_sc)
    R_sc = (L_sc/np.pi)**.5
    T_sc = phunction.m2T_MS(M_sc)
    XYZ_sc = np.array([colour.sd_to_XYZ(colour.sd_blackbody(t)) for t in T_sc])
    RGB_sc = colour.XYZ_to_sRGB(XYZ_sc)

    _max = np.max(RGB_sc,axis=1)
    RGB_sc = RGB_sc/_max[:,np.newaxis]
    RGB_sc[RGB_sc<0.] = 0.

    y_sc = np.linspace(.05,.85,len(L_sc))*h_l
    for i in range(len(L_sc))[::-1]:
        ax.add_artist(plt.Circle((.05*w_l,y_sc[i]),R_sc[i],fc=RGB_sc[i],ec='k',
            lw=.3,zorder=43))
        plt.text(.08*w_l,y_sc[i]-0.01*h_l,str(L_sc[i])+r'$\ L_{\odot}$',
            fontsize=12,zorder=45)

    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    name_f = '_'.join(title_l0.split("'s "))
    plt.savefig('image/distribution/dots_%s_N%s.pdf'%(name_f,N_moninal))
    plt.close()
#'''


''' mean color of stellar clusters 

# parameters
N0 = 1000 # nomical star number 
m_min = .079 # [Msun], default is .03 for 'kroupa' in imf module
m_max = 120 # [Msun], default is 120 for 'kroupa' in imf module

for massfunc in ['kroupa_PMF','kroupa']: 

     # IMF function object
    IMF = None
    if massfunc=='kroupa_PMF':
        IMF = imf.Kroupa_PMF()
    elif massfunc=='kroupa':
        IMF = imf.Kroupa()

    # draw samples
    M_tot = N0*IMF.m_integrate(m_min,m_max)[0]/IMF.integrate(m_min,m_max)[0]
    M = imf.imf.make_cluster(M_tot,massfunc=massfunc,mmin=m_min,mmax=m_max)
    M = np.sort(M)
    N = len(M)


    # stellar properties
    T = phunction.m2T_MS(M) 
    SED = [colour.sd_blackbody(t) for t in T]
    SED_cl = SED[0]
    for i in range(1,len(SED)):
        SED_cl += SED[i]
    RGB_cl = phunction.SED2color([SED_cl])[0] # mean cluster color


    # demo of cluster color
    plt.figure(figsize=(4,4))
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    ax = plt.gca()
    ax.set_facecolor(RGB_cl)

    # title 
    title = None
    if massfunc=='kroupa_PMF':
        title = "Kroupa's PMF"
    elif massfunc=='kroupa':
        title = "Kroupa's IMF"
    title += '\n'+r'$%.2f-%s\ M_{\odot}$'%(m_min,m_max)
    plt.text(.03,.85,title,transform=ax.transAxes,fontsize=12,zorder=45,
        weight='bold')

    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig('image/color/cluster_%s_N%s.pdf'%(massfunc,N_moninal))
    plt.close()
#'''


