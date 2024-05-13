# To play LaHuoChe (card train).

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


np.random.seed(0)
t = time.time()


def card_train(Card):
    # deal
    np.random.shuffle(Card) # shuffle the cards 
    Hand10 = Card[:26]
    Hand20 = Card[26:]

    Hand1 = Hand10.copy()
    Hand2 = Hand20.copy()

    # play the card-train game  
    Train = []
    n_play = 0
    while 1:
        # player 1 plays
        Train.append(Hand1.pop(0))
        n_play += 1

        # check the train 
        Ind = np.where(np.array(Train[:-1])==Train[-1])[0]
        if len(Ind)>0:
            ind = Ind[0]
            Hand1.extend(Train[ind:])
            Train = Train[:ind]

        elif len(Hand1)==0: # check whether has no cards 
            break 

        # player 2 plays
        Train.append(Hand2.pop(0))
        n_play += 1 

        # check the train 
        Ind = np.where(np.array(Train[:-1])==Train[-1])[0]
        if len(Ind)>0:
            ind = Ind[0]
            Hand2.extend(Train[ind:])
            Train = Train[:ind]

        elif len(Hand2)==0: # check whether has no cards 
            break 

    # end-game analyses 
    winner = len(Hand2)==0

    return Hand10, Hand20, winner, n_play


# parameters
Card = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,
    9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13]
n_round = 1000000 # 1e5 = 2.4 min 

n_card = len(Card)


# the game
Hand1 = np.zeros((n_round,int(n_card/2)),np.int8)
Hand2 = Hand1.copy()
Winner = np.zeros(n_round,bool) # True for 1 wins, False for 2 wins
N_play = np.zeros(n_round,int)

for i in range(n_round):
    Hand1[i], Hand2[i], Winner[i], N_play[i] = card_train(Card)


n_win1 = sum(Winner==True) 
print('Player 1 wins %.2f%% of the games.'%(n_win1/n_round*100))

t = (time.time()-t)/60
print('Time elapsed: %.1f min.'%t)


# save 
dill.dump(Hand1,open('variable/Hand1.p','wb'))
dill.dump(Hand2,open('variable/Hand2.p','wb'))
dill.dump(Winner,open('variable/Winner.p','wb'))
dill.dump(N_play,open('variable/N_play.p','wb'))






