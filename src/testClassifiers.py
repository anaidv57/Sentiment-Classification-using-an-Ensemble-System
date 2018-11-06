# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 12:14:26 2018

@author: anaid
"""

from __future__ import print_function
#from keras.preprocessing import sequence
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Embedding
#from keras.layers import Conv1D, GlobalMaxPooling1D
#from keras.datasets import imdb
#from keras.utils.vis_utils import plot_model
#from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime 
import pandas as pd
#from nltk.tokenize import word_tokenize 
#import re
#from sklearn.model_selection import train_test_split, cross_validate
#import matplotlib.pyplot as plt
from numpy.random import seed
from sklearn.metrics import accuracy_score, recall_score, precision_score
import random

from tensorflow import set_random_seed
#seed(1)
#set_random_seed(2)
#import tensorflow as tf
#import pydot
import numpy as np
from load_files import get_train_test
from classifiers import train_classifier1, train_classifier2, train_classifier3, train_classifier4, evaluate_classifier 
from ensemble import voting
from random import randint
from datetime import datetime 
from load_files import load_wordVectors
from ggplot import *
from sklearn.model_selection import train_test_split
seed(randint(0, 50))
set_random_seed(randint(0, 50))
np.random.seed(randint(0, 50))



datasetOld = []

classifier1 = Sequential()
classifier2 = Sequential()
classifier3 = Sequential()
classifier4 = Sequential()


test_size=0.10
#Ensemble
accEnseHistTestOldA = []
accEnseHistTestOld = []
accEnseHistTestNew = []
#--Classifier 1--
#Old
acc1HistTrainOld = []
acc1HistTestOld = []
acc1HistTestOldA = []
#New
acc1HistTrainNew = []
acc1HistTestNew = []
#--Classifier 2--
#Old
acc2HistTrainOld = []
acc2HistTestOld = []
acc2HistTestOldA = []
#New
acc2HistTrainNew = []
acc2HistTestNew = []
#--Classifier 3--
#Old
acc3HistTrainOld = []
acc3HistTestOld = []
acc3HistTestOldA = []
#New
acc3HistTrainNew = []
acc3HistTestNew= []
#--Classifier 4--
#Old
acc4HistTrainOld = []
acc4HistTestOld = []
acc4HistTestOldA = []
#New
acc4HistTrainNew = []
acc4HistTestNew= []


dfALL = pd.DataFrame()
dfA = pd.DataFrame() #Accuracy of the experiments for old datasets
dfN = pd.DataFrame() #Accuracy of the experiments for new datasets
dfAccCl =pd.DataFrame() # Accuracy classifiers
dfC = pd.DataFrame()




xTestArray = []
yTestArray = []
xRTestArray = []
yRTestArray = []

#Return the classifiers that should be train
def classifier_to_train(acc_test1, acc_test2, acc_test3, acc_test4, k, exper, dataset, acc_ensemble): 
    print("ACCURACY Classifier",acc_test1, acc_test2, acc_test3, acc_test4)
    global dfAccCl    
    guilty = ''
    classRandon = ''
    secBest = ''
    classToTrain =[]
    attri = []
    acc = pd.DataFrame()    
    if acc_ensemble > 79:
        print("acc_ensemble", acc_ensemble)
        data = pd.DataFrame({'acc': [acc_test1], 'class': [1]})      
        acc = acc.append(data, ignore_index=True)
        
        data = pd.DataFrame({'acc': [acc_test2], 'class': [2]})       
        acc = acc.append(data, ignore_index=True)
        
        data = pd.DataFrame({'acc': [acc_test3], 'class': [3]})      
        acc = acc.append(data, ignore_index=True)
        
        data = pd.DataFrame({'acc': [acc_test4], 'class': [4]}) 
        acc = acc.append(data, ignore_index=True)         
        acc = acc.sort_values('acc')
        acc = acc.reset_index(drop=True)
        print(acc)
        print(acc['acc'][0])

        if  acc['acc'][0] < 78:
            #Guilty
            guilty = acc['class'][0]
            attri.append(guilty)
            attri.append(proporGuilty)
            attri.append(acc['acc'][0])
            classToTrain.append(attri)
            attri = []

    else:
        data = pd.DataFrame({'acc': [acc_test1], 'class': ['1']})      
        acc = acc.append(data, ignore_index=True)
        
        data = pd.DataFrame({'acc': [acc_test2], 'class': ['2']})       
        acc = acc.append(data, ignore_index=True)
        
        data = pd.DataFrame({'acc': [acc_test3], 'class': ['3']})      
        acc = acc.append(data, ignore_index=True)
        
        data = pd.DataFrame({'acc': [acc_test4], 'class': ['4']})          
        acc = acc.append(data, ignore_index=True)
        
        acc = acc.sort_values('acc')
        acc = acc.reset_index(drop=True)
        print(acc)
#        print(acc['acc'][0])
#        print(acc['acc'][1])

        if  acc['acc'][0] < 78:
            #Guilty
            guilty = acc['class'][0]
            attri.append(guilty)
            attri.append(proporGuilty)
            attri.append(acc['acc'][0])
            classToTrain.append(attri)
            attri = []
            #2nd best  
            secBest =acc['class'][1]
            attri.append(secBest)
            attri.append(propor2Best)
            attri.append(acc['acc'][1])
            classToTrain.append(attri)
            attri = []
        else: #if there is not guilty       
            classRandon = randint(1, 3) #proporRandom 70%
            attri.append(acc['class'][classRandon])
            attri.append(proporRandom)
            attri.append(acc['acc'][classRandon])
            classToTrain.append(attri)
            attri = []
                            
        attri.append(4)#Powerful classifier
        attri.append(proporPowerful)
        attri.append(acc_test4)
        classToTrain.append(attri)
        attri = []
            
    data = pd.DataFrame({'acc_ensemble': [acc_ensemble], 'acc_test1' : [acc_test1], 'acc_test2' : [acc_test2], 'acc_test3' : [acc_test3], 'acc_test4' : [acc_test4], 'k' :[k], 'classGuilty+1' : [guilty], 'val2Best' : [secBest], 'classRandon' : [classRandon], 'exp' : [exper], 'dataset' : [dataset]})
    dfAccCl = dfAccCl.append(data, ignore_index=True)
    print(dfAccCl)
    
    print("classifier to train", classToTrain)            
    return classToTrain


def evaluate_ensemble(new, x_test, y_test, x_textReviews_test, y_textReviews_test, datasetOld= "", d="", rep = "", exper = ""):
    global classifier1, classifier2, classifier3, classifier4, dfA, dfN, dfALL
    global acc1HistTestNew, acc2HistTestNew, acc3HistTestNew, acc4HistTestNew, accEnseHistTestOld
    acc1, y_predict1 = evaluate_classifier(classifier1, x_test, y_test)
    print("acc1",acc1)
    acc2, y_predict2 = evaluate_classifier(classifier2, x_test, y_test)
    print("acc2",acc2)
    acc3, y_predict3 = evaluate_classifier(classifier3, x_test, y_test)
    print("acc3",acc3)
    acc4, y_predict4 = evaluate_classifier(classifier4, x_test, y_test)
    print("acc4",acc4)
    acc_ensemble, recall_ensemble, precision_ensemble, y_voting= voting(y_predict1, y_predict2, y_predict3, y_predict4, y_test, x_textReviews_test, y_textReviews_test)

    if new == True:#New data
        data = pd.DataFrame({'d': [d], 'dataset': [datasetOld], 'acc': [acc_ensemble], 'rep' : [rep], 'exp' : [exper], 'recall_ensemble' : [recall_ensemble], 'precision_ensemble' : [precision_ensemble] })      
        dfN = dfN.append(data, ignore_index=True)
        data = pd.DataFrame({'State' : 'NEW','acc1' : [acc1], 'acc2' : [acc2], 'acc3' : [acc3], 'acc4' : [acc4],'d': [d], 'dataset': [datasetOld], 'acc': [acc_ensemble], 'rep' : [rep], 'exp' : [exper], 'recall_ensemble' : [recall_ensemble], 'precision_ensemble' : [precision_ensemble] })
        dfALL = dfALL.append(data, ignore_index=True)
        acc1HistTestNew.append(acc1)
        acc2HistTestNew.append(acc2)
        acc3HistTestNew.append(acc3)
        acc4HistTestNew.append(acc4)
        accEnseHistTestNew.append(acc_ensemble)
        print("ACCURACY FOR NEW DATA",acc_ensemble)
    else: #Old data
        data = pd.DataFrame({'d': [d], 'dataset': [datasetOld], 'acc': [acc_ensemble], 'rep' : [rep], 'exp' : [exper], 'recall_ensemble' : [recall_ensemble], 'precision_ensemble' : [precision_ensemble]})
        dfA = dfA.append(data, ignore_index=True)
        data = pd.DataFrame({'State' : 'OLD', 'acc1' : [acc1], 'acc2' : [acc2], 'acc3' : [acc3], 'acc4' : [acc4], 'd': [d], 'dataset': [datasetOld], 'acc': [acc_ensemble], 'rep' : [rep], 'exp' : [exper], 'recall_ensemble' : [recall_ensemble], 'precision_ensemble' : [precision_ensemble]})
        dfALL = dfALL.append(data, ignore_index=True)
        acc1HistTestOld.append(acc1)
        acc2HistTestOld.append(acc2)
        acc3HistTestOld.append(acc3)
        acc4HistTestOld.append(acc4)
        accEnseHistTestOld.append(acc_ensemble)
        print("ACCURACY FOR OLD DATA",acc_ensemble)
        
    return acc1, acc2, acc3, acc4, acc_ensemble

        
#Plot of the experiment 
def create_plot(dfmean, exper): 
    title = 'Evolution of the accuracy -Experiment ' + str(exper)
    plotAcc = ggplot(dfmean, aes(x='Classifiers',y='acc_mean', colour='dataset')) + \
    geom_point(size=40)+ \
    ggtitle(title)+ \
    xlim(1, 3)+ \
    ylim(0, 100)+ \
    scale_x_continuous(breaks = (0, 1, 2, 3, 4))+ \
    scale_y_continuous(breaks=[10,20,30,40,50,60,70,80,90,100])    
    print(plotAcc)
    plotAcc.save('plotExp'+str(exper)+'.png')

#Save parameters of the experiment    
def file_params(w):
    
    global exper, dataset, test_size, proporGuilty, proporPowerful,proporRandom, propor2Best, rep
    dfParam =pd.DataFrame()
    data = pd.DataFrame({'Experiment' : [w], 'date' : [datetime.now()], 'rep' : [rep], 'dataset' : [dataset], 'test_size:': [test_size], 'proporGuilty': [proporGuilty], 'proporPowerful': [proporPowerful], 'proporRandom' :[proporRandom], 'propor2Best' :[propor2Best]})
    dfParam = dfParam.append(data, ignore_index=True)
    dfParam = dfParam.T
    save_file(dfParam, 'Experiment'+str(w)+'.xlsx')
    
def save_file(df, namef):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(namef, engine='xlsxwriter')    
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
        
dataset = []



dataset.append('imdb')
dataset.append('yelp')
dataset.append('amazon')
dataset.append('facebook')

start = datetime.now()
print("Staring-> ",start)     
exper = 1#DONT CHANGE
for w in range(0, exper):            
    rep = 10
    for k in range(0, rep):#Repetition
        seed(randint(0, 50))
        set_random_seed(randint(0, 50))
        np.random.seed(randint(0, 50))
        print("***************REPETITION******* ", k)
        i = 0
        d = 0
        classifier1 = Sequential()
        classifier2 = Sequential()
        classifier3 = Sequential()
        classifier4 = Sequential()
        
        embedding_matrix = load_wordVectors()
        while i < len(dataset):
            print("---------------NEXT DATASET-------------", i, dataset[i] )
            seed(randint(0, 50))
            set_random_seed(randint(0, 50))
            np.random.seed(randint(0, 50))
            datasetOld.append(dataset[i])
            previous_weight=False   
     
            x_train, x_test, y_train, y_test, x_textReviews_test, y_textReviews_test = get_train_test(dataset=dataset[i], test_size=test_size, convert=False)
                #Store test set, so I can know if they have fogotten

            #Classifier 1
            classifier1, history1, acc_train1, acc_test1, weights1 = train_classifier1(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight)
            acc1, y_predict1 = evaluate_classifier(classifier1, x_test, y_test)
            print("acc1", acc1)
            recall1 = recall_score(y_test, y_predict1) * 100
            precision1 = precision_score(y_test, y_predict1) * 100
            data = pd.DataFrame({'dataset': [dataset[i]], 'acc': [acc1], 'Classifiers': [1], 'recall':[recall1], 'precision': [precision1]})
            dfC = dfC.append(data, ignore_index=True)
            
            #Classifier 2
            classifier2, history2, acc_train2, acc_test2, weights2 = train_classifier2(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight)
            acc2, y_predict2 = evaluate_classifier(classifier2, x_test, y_test)
            print("acc2", acc2)
            recall2 = recall_score(y_test, y_predict2) * 100
            precision2 = precision_score(y_test, y_predict2) * 100
            data = pd.DataFrame({'dataset': [dataset[i]], 'acc': [acc2], 'Classifiers': [2], 'recall':[recall2], 'precision': [precision2]})
            dfC = dfC.append(data, ignore_index=True)
#    
#                
            #Classifier 3
            classifier3, history3, acc_train3, acc_test3, weights3 = train_classifier3(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight)
            acc3, y_predict3 = evaluate_classifier(classifier3, x_test, y_test)
            print("acc3", acc3)
            recall3 = recall_score(y_test, y_predict3) * 100
            precision3 = precision_score(y_test, y_predict3) * 100
            data = pd.DataFrame({'dataset': [dataset[i]], 'acc': [acc3], 'Classifiers': '3', 'recall':[recall3], 'precision': [precision3]})
            dfC = dfC.append(data, ignore_index=True)
                                
            i = i+1
        end = datetime.now() - start
        print("Duration", end)
        print(datetime.now())
          
    n = 'A'
    #Accuracy old dataset
    print(dfC)
    save_file(dfC,'Experiment'+n+'Classif.xlsx')
    
    dfCmean = dfC.groupby(['Classifiers','dataset']).mean()
    dfCmean = dfCmean.add_suffix('_mean').reset_index()
    print(dfCmean)
    save_file(dfCmean,'Experiment'+n+'Classifmean.xlsx')
    create_plot(dfCmean, n)
    



 
    




       
    
    


