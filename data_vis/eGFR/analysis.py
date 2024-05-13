import itertools 
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def CKD_EPI(Scr,Sex,Age):
    '''
    CKD-EPI Creatinine Equation (2021)
    https://www.kidney.org/content/ckd-epi-creatinine-equation-2021 

    Inputs 
    ------
    Scr: [umol/L], serum creatinine 
    Sex: 'F' or 'M'
    Age: [yr]

    Returns
    -------
    eGFR: [mL/min/1.73m^2]
    '''

    Scr1 = Scr/88.4 # umol/L to mg/dL 
    kappa = .7 if Sex=='F' else .9 
    alpha = -.241 if Sex=='F' else -.302  

    A = np.min(np.array([Scr1/kappa,np.ones_like(Scr)]),axis=0)
    B = np.max(np.array([Scr1/kappa,np.ones_like(Scr)]),axis=0)

    eGFR = 142*A**alpha * B**-1.2 * .9938**Age
    if Sex=='F':
        eGFR *= 1.012

    return eGFR


#''' eGFR vs Scr 

# parameters 
Scr = np.linspace(60,120,500)
Age = 58 

eGFR_M = CKD_EPI(Scr,'M',Age)
eGFR_F = CKD_EPI(Scr,'F',Age)

plt.figure()
plt.plot(Scr,eGFR_M,color='b',label='Male')
plt.plot(Scr,eGFR_F,color='r',label='Female')
plt.ylim(bottom=0)
plt.text(.6,.9,'Age=%s'%Age,fontsize=18,transform=plt.gca().transAxes)
plt.legend()
plt.grid()
plt.xlabel('Serum creatinine '+r'$\rm(\mu mol\ L^{-1})$')
plt.ylabel('eGFR '+r'$\rm(mL/min/1.73m^2)$')
plt.tight_layout()
plt.savefig('eGFR_vs_Scr.pdf')
plt.close()
#'''





















