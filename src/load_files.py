
#https://gist.github.com/vgpena/b1c088f3c8b8c2c65dd8edbe0eae7023
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime 
import pandas as pd
from nltk.tokenize import word_tokenize 
import re
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt
from numpy.random import seed
#seed(1)
from tensorflow import set_random_seed
#set_random_seed(2)
import tensorflow as tf
import pydot
import numpy as np
from autocorrect import spell
from nltk.tokenize import word_tokenize 
import nltk
np.random.seed([3,1415])
from random import randint

count_spelling = 0
incorrect_words = []

#https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

#Index of words
#https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
def load_word_list():
    wordsList = np.load('wordsList.npy')
    #print('Loaded the word list!')
    wordsList = wordsList.tolist() #Originally loaded as numpy array
    wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
    
    return wordsList

def load_wordVectors():
    wordVectors = np.load('wordVectors.npy')
    return wordVectors

#load facebook dataset. The structure of this data is different
def load_fileF(dataset= 'facebook'): 
    print("Reading file->", dataset )
    directory = "D:\Box Sync\Project AI\Project\Dataset\\"
    fb_data = pd.read_csv(directory+dataset+'.txt', header=None, sep=r"\t", lineterminator="\n", engine='python')#Score is either 1 (for positive) or 0 (for negative)
    fb_label = pd.read_csv(directory+'fb_label.txt', header=None, sep=r"\t", lineterminator="\n", engine='python')#Score is either 1 (for positive) or 0 (for negative)
    result = pd.concat([fb_data, fb_label], axis=1)
    result.columns = ['review','sentiment']
    result.loc[(result.sentiment == 'P') | (result.sentiment == 'N')]#644 are positive, total 720
    result = result[result['sentiment'] != 'O']
    result =result.dropna()
    result = result.reset_index(drop=True)
    #
#    positives =result[result['sentiment'] == 'P']
#    positives = positives.reset_index(drop=True) 
#    positives = positives[:][:79]
#    negatives =result[result['sentiment'] == 'N']
#    negatives = negatives.reset_index(drop=True) 
#    result = pd.concat([positives, negatives])
#    result = result.reset_index(drop=True)
    #    
    X = result['review']
    y = result['sentiment'] #labels 
    y = y.replace('P', "1")
    y = y.replace('N', "0")
    y = y.astype(int)
#    print(X[0])
#    print(y[0])  
    return X, y

#Read dataset: reviews of amazon products, movies, and restaurants. 
def load_file(dataset= 'imdb'):    
    print("Reading file->", dataset )
    directory = "D:\Box Sync\Project AI\Project\Dataset\\"
#    print(dataset)
    data = pd.read_csv(directory+dataset+'.txt', header=None, sep=r"\t", engine='python')#Score is either 1 (for positive) or 0 (for negative)
    data.columns = ['review','sentiment']
#    print("Data", len(data))
    y = data['sentiment'] #labels
    X = data['review']
    return X, y

#Auto-correct spelling
def autocorrect_spell(string):
    global count_spelling
    new_string = spell(string).lower()
    if new_string != string:
        count_spelling = count_spelling + 1
        incorrect_words.append(new_string +','+ string)            
    return new_string

def convert_data(wordsList, X, y): 
    textReview = [] 
    embReview = [] 
    labels = [] 
    reviewCounter = 0
    maxSeqLength = 50
    numFiles = 1000
    ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
    for row in X:
        indexCounter = 0
#        cleanedLine = cleanSentences(row)
#        split = cleanedLine.split()
        token = nltk.word_tokenize(row)
        for word in token:
            try:
                word = autocorrect_spell(word)
                ids[reviewCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[reviewCounter][indexCounter] = 399999 #Vector for unkown words
#                count_unkown = count_unkown + 1
#                unknown.append(word)
            indexCounter = indexCounter + 1
            if indexCounter >= maxSeqLength:
                break
        #embReview.append(wordVectors[ids[reviewCounter]])
        labels.append(y[reviewCounter])
        embReview.append(ids[reviewCounter])
#        textReview.append(cleanedLine)
        reviewCounter = reviewCounter + 1

    return embReview, labels

#Save reviews converted in numbers
def save_convert_data_x(embReview, dataset):
    convert_name = 'convert_data_' + dataset +'.npy'
    np.save(convert_name, embReview)
    print("save")
    
#Load reviews converted in numbers
def load_convert_data_x(dataset):
    convert_name = 'convert_data_' + dataset +'.npy'
    return np.load(convert_name)
  
def get_data(dataset, convert = False):
    if dataset == 'facebook':
        X, y = load_fileF(dataset)
    else:
        X, y = load_file(dataset)
    textReview = X
    if convert == True:
        wordsList = load_word_list()
        embReview, labels = convert_data(wordsList, X, y)
        save_convert_data_x(embReview, dataset)
    else:
        embReview = load_convert_data_x(dataset)
    return textReview,embReview, y

def get_train_test(dataset, test_size, convert): 
    np.random.seed([3,1415])
    seed(randint(0, 50))
    set_random_seed(randint(0, 50))
    np.random.seed(randint(0, 50))
    textReview,X, y= get_data(dataset, convert=convert)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state = randint(0, 50), stratify = y) 
    x_textReviews_train, x_textReviews_test, y_textReviews_train, y_textReviews_test = train_test_split(textReview, y, test_size=test_size, stratify = y)

#    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = 413, stratify = y) 
#    x_textReviews_train, x_textReviews_test, y_textReviews_train, y_textReviews_test = train_test_split(textReview, y, test_size=test_size, random_state = 413, stratify = y)
#    print(y_test)
    x_test = np.asarray(x_test) 
    x_train = np.asarray(x_train) 
    y_test = np.asarray(y_test) 
    y_train = np.asarray(y_train)
#    print(y_test)
    x_textReviews_test = np.asarray(x_textReviews_test)
    y_textReviews_test = np.asarray(y_textReviews_test)
    return x_train, x_test, y_train, y_test, x_textReviews_test, y_textReviews_test

