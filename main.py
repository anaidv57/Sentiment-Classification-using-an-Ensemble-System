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


test_size=0.20
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
    plotAcc = ggplot(dfmean, aes(x='d',y='acc_mean', colour='dataset')) + \
    geom_line(size=1.5) + \
    geom_point(size=40)+ \
    ggtitle(title)+ \
    xlim(1, 3)+ \
    ylim(0, 100)+ \
    scale_x_continuous(breaks = (0, 1, 2, 3, 4))+ \
    scale_y_continuous(breaks=[10,20,30,40,50,60,70,80,90,10])    
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



#1- Done 1-6
#dataset.append('amazon')
#dataset.append('yelp')
#dataset.append('imdb')
#dataset.append('facebook')

#2- Done 7-12
#dataset.append('yelp')
#dataset.append('amazon')
#dataset.append('imdb')
#dataset.append('facebook')

#3-done 13-done
#dataset.append('imdb')
#dataset.append('yelp')
#dataset.append('amazon')
#dataset.append('facebook')

#4done
#dataset.append('amazon')
#dataset.append('facebook')
#dataset.append('imdb')
#dataset.append('yelp')

#5 done 25
dataset.append('amazon')
dataset.append('imdb')
dataset.append('facebook')
dataset.append('yelp')


#6 si da tiempo se hace 
#dataset.append('facebook')
#dataset.append('amazon')
#dataset.append('imdb')
#dataset.append('yelp')






start = datetime.now()
print("Staring-> ",start)     
exper = 1#DONT CHANGE
for w in range(0, exper):
    if w == 4:#1Done, 7 good result, 13 good, 19done, 25done, 31 done, 37done, 43 done, 49 done, 55done
        proporGuilty = 0.7
        proporPowerful =  0.8
        proporRandom = 0.6
        propor2Best = 0.5 
    if w == 4:#2b Good result, 8, 14 done, 20 done, 26 done, 32 done, 38 done, 44 done, 50 done, 56 done
        proporGuilty = 0.5
        proporPowerful =  0.8
        proporRandom = 0.6
        propor2Best = 0.7         
    if w == 4:#3done, 9 done, 15 done, 21 done, 27 done, 33 done, 39 done, 45 done, 51 done, 57done
        proporGuilty = 0.7
        proporPowerful =  0.8
        proporRandom = 0.5
        propor2Best = 0.5            
    if w == 4: #4done, 10 done, 16 done, 22 done better, 28 done, 34 done, 40 done, 46 done, 52done, 58 done
        proporGuilty = 0.8
        proporPowerful =  0.9
        proporRandom = 0.5
        propor2Best = 0.7
    if w == 4: #5 done, 11done, 17done, 23 done, 29(prob plot), 35 done, 41 done, 47 done, 53 done, 59 done
        proporGuilty = 0.6
        proporPowerful =  0.8
        proporRandom = 0.7
        propor2Best = 0.6
    if w == 0: #6 done, 12 done, 18 doing, 24 done better, 30done good, 36 done, 42 done, 48done, 54 done, 60 done
        proporGuilty = 0.6
        proporPowerful =  0.8
        proporRandom = 0.6
        propor2Best = 0.7
            
    rep = 10
    for k in range(0, rep):#Repetition
        seed(randint(0, 50))
        set_random_seed(randint(0, 50))
        np.random.seed(randint(0, 50))
        print("***************REPETITION******* ", k)
        i = 0
        d = 0
        datasetOld = []
        xTestArray = []
        yTestArray = []
        xRTestArray = []
        yRTestArray = []
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
            if i == 0:#All classifiers are trained using the same training set      
                x_train, x_test, y_train, y_test, x_textReviews_test, y_textReviews_test = get_train_test(dataset=dataset[i], test_size=test_size, convert=False)
                #Store test set, so I can know if they have fogotten
                xTestArray.append(x_test)
                yTestArray.append(y_test)
                xRTestArray.append(x_textReviews_test)
                yRTestArray.append(y_textReviews_test)
                #Classifier 1
                classifier1, history1, acc_train1, acc_test1, weights1 = train_classifier1(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight)
                acc1, y_predict1 = evaluate_classifier(classifier1, x_test, y_test)
                #print("acc1", acc1)
                acc1HistTrainNew.append(acc_train1)#later decide if it is new or old
                acc1HistTestNew.append(acc_test1)
                
                #Classifier 2
                classifier2, history2, acc_train2, acc_test2, weights2 = train_classifier2(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight)
                acc2, y_predict2 = evaluate_classifier(classifier2, x_test, y_test)
                #print("acc2", acc2)
                acc2HistTrainNew.append(acc_train2)
                acc2HistTestNew.append(acc_test2)
                    
                #Classifier 3
                classifier3, history3, acc_train3, acc_test3, weights3 = train_classifier3(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight)
                acc3, y_predict3 = evaluate_classifier(classifier3, x_test, y_test)
                #print("acc3", acc3)
                acc3HistTrainNew.append(acc_train3)
                acc3HistTestNew.append(acc_test3)
                
                #Classifier 4
                classifier4, history4, acc_train4, acc_test4, weights4 = train_classifier4(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight)
                acc4, y_predict4 = evaluate_classifier(classifier4, x_test, y_test)
                #print("acc4", acc4)
                acc4HistTrainNew.append(acc_train4)
                acc4HistTestNew.append(acc_test4)
                
                #Ensemble testing: using the predictions of all classifiers 
                acc_ensemble, recall, precision, y_voting= voting(y_predict1, y_predict2, y_predict3, y_predict4, y_test, x_textReviews_test, y_textReviews_test)
                evaluate_ensemble(False, x_test, y_test, x_textReviews_test, y_textReviews_test, dataset[i], d=0, rep=k, exper=exper)#old data
                #print(acc_ensemble)
#                accEnseHistTestNew.append(acc_ensemble)
                
    #            classToTrain = classifier_to_train(acc_test1, acc_test2, acc_test3)
    #            print(classToTrain)
                
            else:
                d = d + 1 # arrive new dataset
                print("d", d)
                print("----new dataset-----", dataset[i])
                previous_weight=True
                x_train, x_test, y_train, y_test, x_textReviews_test, y_textReviews_test = get_train_test(dataset=dataset[i], test_size=test_size, convert=False) #convertir en listas
                xTestArray.append(x_test)
                yTestArray.append(y_test)
                xRTestArray.append(x_textReviews_test)
                yRTestArray.append(y_textReviews_test)
                acc1, acc2, acc3, acc4, acc_ensemble = evaluate_ensemble(True,x_test, y_test, x_textReviews_test, y_textReviews_test, datasetOld= dataset[i], d=d, rep=k, exper=exper) #new data
                classToTrain = classifier_to_train(acc1, acc2, acc3, acc4, k, w, dataset[i], acc_ensemble)        
                j = 0
                q = 0                
    #            print("Len classToTrain", len(classToTrain))
                while j < len(classToTrain):
                        #Train classifier
                        nameClassifier = "train_classifier" + str(classToTrain[j][0])
                        nameWeight = "weights" + str(classToTrain[j][0])
                        print(nameClassifier)
                        test_size_clasif = 1 - classToTrain[j][1] #ficticio testing set. I need to create the new training set for this classifier
                        print("test_size_clasif", test_size_clasif)
#                        x_train, _, y_train, _, _, _ = get_train_test(dataset=dataset[i], test_size=test_size_clasif, convert=False)
                        x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=test_size,random_state = randint(0, 50), stratify = y_train)
                        x_train = np.asarray(x_train) 
                        y_train = np.asarray(y_train)
                        classifier, history, acc_train, acc_test, weights = eval(nameClassifier+"(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight, eval(nameWeight))")
                        if classToTrain[j][0] == 1: 
                            classifier1 = classifier
                            weights1 = weights
                        if classToTrain[j][0] == 2: 
                            classifier2 = classifier
                            weights2 = weights
                        if classToTrain[j][0] == 3: 
                            classifier3 = classifier
                            weights3 = weights
                        if classToTrain[j][0] == 4: 
                            classifier4 = classifier
                            weights4 = weights
                        #eval("acc"+str(classToTrain[j][0])+"HistTrainOld.append("+acc_train+")")
                        #eval("acc"+str(classToTrain[j][0])+"HistTestOld.append("+acc_test+")")
                        print("acc_train", acc_train)
                        print("acc_test", acc_test)
                        j = j+1
                 
                #Evaluate actual dataset and previous datasets    
                while q < len(datasetOld): 
                    print("OLD DATASET->", datasetOld[q])
                    evaluate_ensemble(False, xTestArray[q], yTestArray[q], xRTestArray[q], yRTestArray[q], datasetOld[q], d=d, rep=k, exper=exper)#old data
                    q= q+1
                
            i = i+1
        end = datetime.now() - start
        print("Duration", end)
        print(datetime.now())
        
#        print("accEnseHistTestNew", accEnseHistTestNew)
#    print("acc1HistTestOld", acc1HistTestOldA)         
#    print("acc2HistTestOldA", acc2HistTestOldA)       
#    print("acc3HistTestOldA", acc3HistTestOldA)
#    print("acc4HistTestOldA", acc4HistTestOldA)   
    n = 60
    #Accuracy old dataset
#    print(dfA)
    save_file(dfA,'Experiment'+str(n)+'AccAllOld.xlsx')
    
    dfmean = dfA.groupby(['d','dataset']).mean()
    dfmean = dfmean.add_suffix('_mean').reset_index()
#    print(dfmean)
    save_file(dfmean, 'Experiment'+str(n)+'AccOldMean.xlsx')
    create_plot(dfmean, str(n))
    
    #Accuracy new dataset
#    print(dfN)
    save_file(dfN,'Experiment'+str(n)+'AccAllNew.xlsx')
    
    dfmeanN = dfN.groupby(['d','dataset']).mean()
    dfmeanN = dfmeanN.add_suffix('_mean').reset_index()
#    print(dfmeanN)
    save_file(dfmeanN, 'Experiment'+str(n)+'AccNewMean.xlsx')
    #save parameters of the experiment
    file_params(n)
    
#    print(dfAccCl)
    save_file(dfAccCl, 'Classifiers'+str(n)+'AccN.xlsx')
    dfALL
    save_file(dfALL, 'Classifiers'+str(n)+'AccALL.xlsx')
    

#    %logstart -o , ipython_log.py

#    print("Accuracy old dataset")
#    print(dfA)
#    print(dfmean)
#    print("Accuracy new dataset")
#    print(dfN)
#    print(dfmeanN)
    print("Accuracy new dataset")
    print(dfAccCl)
#    print("History All")
#    print(dfALL)
    



 
    




       
    
    


