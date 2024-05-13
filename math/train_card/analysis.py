'''
To analyze the card-train games.

Conclusion: the outcome (winning&game length) is input-sensitive. Any indicators
that are continuous functions of the cards fail to predict the game outcome.
'''

import os
import time
import dill
import numpy as np
import matplotlib.pyplot as plt


# change default matplotlib fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 13


def calc_indicator(hand,kind):
    '''
    Inputs
    ------
    hand: 1d array, a hand of cards.
    kind: str, specifies the indicator type 

    Returns
    -------
    m: value of the indicator

    ''' 

    if kind=='distance':
        array = np.arange(13) 

        m = 0
        num = 0
        for i in range(1,14):
            ind = np.where(hand==i)[0]
            if len(ind)>1:
                m += ind[-1]-ind[0]
                num += len(ind)-1
        m /= num

        return m 


# load 
Hand1 = dill.load(open('variable/Hand1.p','rb'))
Hand2 = dill.load(open('variable/Hand2.p','rb'))
Winner = dill.load(open('variable/Winner.p','rb'))
N_play = dill.load(open('variable/N_play.p','rb'))

Hand1_win = Hand1[Winner]
Hand1_los = Hand1[~Winner]
Hand2_win = Hand2[~Winner]
Hand2_los = Hand2[Winner]


# analysis: how the properties of Hand determine winning/N_play ----------------

# indicator: mean distance of identical cards ..................................

# parameters
kind = 'distance'

M1_win = np.array([calc_indicator(i,kind) for i in Hand1_win])
M1_los = np.array([calc_indicator(i,kind) for i in Hand1_los])
M2_win = np.array([calc_indicator(i,kind) for i in Hand2_win])
M2_los = np.array([calc_indicator(i,kind) for i in Hand2_los])
M1 = np.array([calc_indicator(i,kind) for i in Hand1])
M2 = np.array([calc_indicator(i,kind) for i in Hand2])

print(np.mean(M1_win),np.mean(M1_los))
print(np.median(M1_win),np.median(M1_los))
print(np.mean(M2_win),np.mean(M2_los))
print(np.median(M2_win),np.median(M2_los))


if not os.path.isdir('image/%s'%kind):
    os.mkdir('image/%s'%kind)


''' hist of M1_win/los
bins = np.linspace(min(M1_win.min(),M1_los.min()),
    max(M1_win.max(),M1_los.max()),50)
cen = (bins[1:]+bins[:-1])/2 

plt.figure(figsize=(8,4))
num1_win = plt.hist(M1_win,bins,fc='g',ec='g',alpha=.3,label='Player 1 wins')[0]
num1_los =plt.hist(M1_los,bins,fc='r',ec='r',alpha=.3,label='Player 1 loses')[0]
plt.legend()
plt.grid()
plt.xlabel('Mean distance of identical cards')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('image/%s/hist_M1.pdf'%kind)
plt.close()
#'''


''' hist of M2_win/los
bins = np.linspace(min(M2_win.min(),M2_los.min()),
    max(M2_win.max(),M2_los.max()),50)
cen = (bins[1:]+bins[:-1])/2 

plt.figure(figsize=(8,4))
num2_win = plt.hist(M2_win,bins,fc='g',ec='g',alpha=.3,label='Player 2 wins')[0]
num2_los =plt.hist(M2_los,bins,fc='r',ec='r',alpha=.3,label='Player 2 loses')[0]
plt.legend()
plt.grid()
plt.xlabel('Mean distance of identical cards')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('image/%s/hist_M2.pdf'%kind)
plt.close()
#'''


''' Pct of winning vs distance
pct1_win = num1_win/(num1_win+num1_los)

plt.figure()
plt.plot(cen,pct1_win,color='k')
plt.axhline(.5,color='k',ls='--')
plt.grid()
plt.xlabel('Mean distance of identical cards of player 1')
plt.ylabel('Frequency of 1 winning')
plt.tight_layout()
plt.savefig('image/%s/win1_vs_distance.pdf'%kind)
plt.close()

# Pct of winning vs distance
pct2_win = num2_win/(num2_win+num2_los)

plt.figure()
plt.plot(cen,pct2_win,color='k')
plt.axhline(.5,color='k',ls='--')
plt.grid()
plt.xlabel('Mean distance of identical cards of player 2')
plt.ylabel('Frequency of 2 winning')
plt.tight_layout()
plt.savefig('image/%s/win2_vs_distance.pdf'%kind)
plt.close()
#'''


''' M1/2 vs N_play
plt.figure()
plt.scatter(M1,N_play,fc='k',ec='none')
plt.grid()
plt.xlabel('Mean distance of identical cards of player 1')
plt.ylabel('# of cards played')
plt.tight_layout()
plt.savefig('image/%s/N_play_vs_M1.pdf'%kind)
plt.close()

plt.figure()
plt.scatter(M2,N_play,fc='k',ec='none')
plt.grid()
plt.xlabel('Mean distance of identical cards of player 2')
plt.ylabel('# of cards played')
plt.tight_layout()
plt.savefig('image/%s/N_play_vs_M2.pdf'%kind)
plt.close()
#'''


''' M1 vs M2 
plt.figure()
plt.scatter(M1,M2,fc='k',ec='none')
plt.grid()
plt.xlabel('Mean distance of identical cards of player 1')
plt.ylabel('Ditto for player 2')
plt.tight_layout()
plt.savefig('image/%s/M1_vs_M2.pdf'%kind)
plt.close()
#'''


# demo -------------------------------------------------------------------------

''' hist of N_paly
plt.figure(figsize=(8,4))
plt.hist(N_play,int(n_round**.35),fc='none',ec='k')
plt.text(.6,.8,'Total # of rounds: %d'%n_round,fontsize=14,
    transform=plt.gca().transAxes)
plt.grid()
plt.xlabel('# of plays')
plt.ylabel('# of rounds')
plt.tight_layout()
plt.savefig('image/hist_n_play.pdf')

# log
lg_N_play = np.log10(N_play)

plt.close()
plt.figure(figsize=(8,4))
plt.hist(lg_N_play,int(n_round**.35),fc='none',ec='k')
plt.text(.6,.8,'Total # of rounds: %d'%n_round,fontsize=14,
    transform=plt.gca().transAxes)
plt.grid()
plt.xlabel('lg(# of plays)')
plt.ylabel('# of rounds')
plt.tight_layout()
plt.savefig('image/hist_lg_n_play.pdf')
plt.close()
#'''
