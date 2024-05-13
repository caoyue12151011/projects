import itertools 
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime


# global parameters
time_def = '09:00' # default time of measurement, if there's no data

sys_opt_max = 120 # highest optimal sys value
sys_n_max = 130 # normal
sys_hn_max = 140 # high normal
sys_ht1_max = 160 # Grade 1 hypertension   
sys_ht2_max = 180 

dia_opt_max = 80 # highest optimal dia value
dia_n_max = 85 # normal
dia_hn_max = 90 # high normal
dia_ht1_max = 100 # Grade 1 hypertension   
dia_ht2_max = 110 

pp_norm_min, pp_norm_max = 40,60
hrt_norm_max = 75 # target heat rate for patients with hypertension


# read data --------------------------------------------------------------------

f = open('data/blood_pressure.csv','r')
f.readline()
data = f.readlines()
f.close()

DT = [] # date & time
SYS = []
DIA = []
HRT = []
Arm = [] # 'L' or 'R'
for line in data:
    date, time, sys, dia, hrt, arm, note = line.split(',')

    # fix absent data
    if time=='':
        time = time_def 
    sys,dia,hrt = [np.nan if i=='' else float(i) for i in [sys,dia,hrt]]

    DT.append(datetime.strptime('%s %s'%(date,time),'%Y-%m-%d %H:%M'))
    SYS.append(sys)
    DIA.append(dia)
    HRT.append(hrt)
    Arm.append(arm)

DT  = np.array(DT )
SYS = np.array(SYS)
DIA = np.array(DIA)
HRT = np.array(HRT)
Arm = np.array(Arm)

# data analysis ----------------------------------------------------------------

PP = SYS-DIA 

# mean values
sys_av = np.nanmean(SYS)
dia_av = np.nanmean(DIA)
hrt_av = np.nanmean(HRT)
pp_av = np.nanmean(PP)




# demo =========================================================================

''' bp, hrt vs time

# parameters
yticks = np.arange(40,165,10)

plt.figure(figsize=(14,6))
ax = plt.gca()
plt.plot(DT,SYS,linewidth=.5,color='blue',label='SYS')
plt.plot(DT,DIA,linewidth=.5,color='c',label='DIA')
plt.plot(DT,HRT,linewidth=.5,color='r',label='Heart rate')
plt.axhline(sys_av,linewidth=1,color='blue',linestyle='--')
plt.axhline(dia_av,linewidth=1,color='c',linestyle='--')
plt.axhline(hrt_av,linewidth=1,color='r',linestyle='--')
plt.grid()
plt.legend()
ax.set_yticks(yticks)
ax.set_yticklabels(yticks)
plt.xlabel('Date & time')
plt.tight_layout()
plt.savefig('image/bp_hr_vs_t.pdf')
plt.close()
#'''


''' distributions 

# parameters
alpha = .3


plt.figure()
plt.hist(SYS,int(len(DT)**.5),color='blue',histtype='step',linewidth=2)
plt.axvline(sys_av,color='blue',ls='--')

plt.xlim(plt.xlim())
plt.axvspan(0,sys_opt_max,fc='green',ec='none',alpha=alpha)
plt.axvspan(0,sys_n_max,fc='yellow',ec='none',alpha=alpha)
plt.axvspan(sys_n_max,sys_hn_max,fc='orange',ec='none',alpha=alpha)
plt.axvspan(sys_hn_max,sys_ht1_max,fc='orangered',ec='none',alpha=alpha)
plt.axvspan(sys_ht1_max,sys_ht2_max,fc='red',ec='none',alpha=alpha)
plt.axvspan(sys_ht2_max,300,fc='brown',ec='none',alpha=alpha)

plt.grid()
plt.xlabel('SYS')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('image/hist_sys.pdf')
plt.close()
# ..............................................................................
plt.figure()
plt.hist(DIA,int(len(DT)**.5),color='c',histtype='step',linewidth=2)
plt.axvline(dia_av,color='c',ls='--')

plt.xlim(plt.xlim())
plt.axvspan(0,dia_opt_max,fc='green',ec='none',alpha=alpha)
plt.axvspan(0,dia_n_max,fc='yellow',ec='none',alpha=alpha)
plt.axvspan(dia_n_max,dia_hn_max,fc='orange',ec='none',alpha=alpha)
plt.axvspan(dia_hn_max,dia_ht1_max,fc='orangered',ec='none',alpha=alpha)
plt.axvspan(dia_ht1_max,dia_ht2_max,fc='red',ec='none',alpha=alpha)
plt.axvspan(dia_ht2_max,300,fc='brown',ec='none',alpha=alpha)

plt.grid()
plt.xlabel('DIA')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('image/hist_dia.pdf')
plt.close()
# ..............................................................................
plt.figure()
plt.hist(HRT,int(len(DT)**.5),color='r',histtype='step',linewidth=2)
plt.axvline(hrt_av,color='r',ls='--')

plt.xlim(plt.xlim())
plt.axvspan(0,hrt_norm_max,fc='green',ec='none',alpha=alpha)
plt.axvspan(hrt_norm_max,300,fc='red',ec='none',alpha=alpha)

plt.grid()
plt.xlabel('Heart rate')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('image/hist_hrt.pdf')
plt.close()
# ..............................................................................
plt.figure()
plt.hist(PP,int(len(DT)**.5),color='k',histtype='step',linewidth=2)
plt.axvline(pp_av,color='k',ls='--')

plt.xlim(plt.xlim())
plt.axvspan(pp_norm_min,pp_norm_max,fc='green',ec='none',alpha=alpha)
plt.axvspan(0,pp_norm_min,fc='red',ec='none',alpha=alpha)
plt.axvspan(pp_norm_max,300,fc='red',ec='none',alpha=alpha)

plt.grid()
plt.xlabel('Pulse pressure (SYS-DIA)')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('image/hist_pp.pdf')
plt.close()
#'''


#''' correlations

# parameters
Data = {
    'SYS': SYS, 
    'DIA': DIA, 
    'Pulse pressure': PP, 
    'Heart rate': HRT,
}

for item_x, item_y in itertools.combinations(sorted(Data),2):

    data_x = Data[item_x]
    data_y = Data[item_y]

    plt.figure()
    plt.scatter(data_x,data_y,color='k',s=5)
    plt.grid()
    plt.xlabel(item_x)
    plt.ylabel(item_y)
    plt.tight_layout()
    plt.savefig('image/correlation/%s_%s.pdf'%(item_x,item_y))
    plt.close()
#'''










#'''

