# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:36:11 2018

@author: anaid
"""

from __future__ import print_function
from sklearn.metrics import accuracy_score, recall_score, precision_score
from numpy.random import seed
#seed(1)
from tensorflow import set_random_seed
#set_random_seed(2)
from random import randint
import numpy as np
seed()
#set_random_seed([1,1415])
#np.random.seed([3,1415])


def voting(y_predict1, y_predict2, y_predict3, y_predict4, y_test, x_textReviews_test, y_textReviews_test ):
    seed(randint(0, 50))
    set_random_seed(randint(0, 50))
    np.random.seed(randint(0, 50))
    i=0
    y_voting = []
#    while i < len(y_predict1):
    while i < len(y_predict1):
        #print("Votes: ", y_predict1[i], y_predict2[i], y_predict3[i], y_predict4[i])
        average = (y_predict1[i][0] + y_predict2[i][0] + y_predict3[i][0]+ y_predict4[i][0])/4
        if average == 0.5:
            y = randint(0, 1)
            y_voting.append(y)
            #print("Voting random ", y, "true label", y_test[i])
            #print(y_textReviews_test[i], x_textReviews_test[i])
        else:
            y = int((round(average,0)))
            #print("After voting ", y, "true label", y_test[i])
            #print(y_textReviews_test[i], x_textReviews_test[i])
            y_voting.append(y)
        i += 1 
#    print("y_voting", y_voting)
#    print("y_test", y_test)
        
    acc = accuracy_score(y_test, y_voting) * 100
    recall = recall_score(y_test, y_voting) * 100
    precision = precision_score(y_test, y_voting) * 100
    
    print("****ACCURACY ENSEMBLE", acc)
    print("recall ensemble", recall)
    print("precision ensemble", precision)
    
    return acc, recall, precision, y_voting