#https://gist.github.com/vgpena/b1c088f3c8b8c2c65dd8edbe0eae7023
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.initializers import Initializer
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
import os
#from load_files import get_train_test
from load_files import load_wordVectors
from random import randint
from keras.callbacks import EarlyStopping
seed()
#set_random_seed([1,1415])
#np.random.seed([3,1415])


def evaluate_classifier(model, x_test, y_test):
    scores = model.evaluate(x=x_test, y=y_test, batch_size=5)
    acc = scores[1]*100
    y_predict = model.predict_classes(x_test)
    #print("y_test", y_test)    
    return acc, y_predict


#CNN 1

def train_classifier1(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight=False, weights=0): 
    seed(randint(0, 50))
    set_random_seed(randint(0, 50))
    np.random.seed(randint(0, 50))
#    Initializer()
   # set parameters:
    maxlen = 50
    batch_size = 50
    filters = 200#250
    kernel_size = 6#3
    hidden_dims = 200#250
    epochs = 9  #9

    
    print('---------------Build Classifier 1...')
#    print("embedding_matrix", embedding_matrix.shape)
#    print("x_train",x_train.shape)
#    print("y_train",y_train.shape)
#    print("x_test",x_test.shape)
#    print("y_test",y_test.shape)
    model = Sequential()    
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(embedding_matrix.shape[0],#400000
                        embedding_matrix.shape[1],
#                        embedding_dims,#10
#                        weights=[embedding_matrix],
                        input_length=maxlen))
    model.add(Dropout(0.7))    
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())   
    # We add a vanilla hidden layer:
    #model.add(Dense(hidden_dims))
#    model.add(Dense(hidden_dims, kernel_initializer= 'random_uniform', bias_initializer='zeros'))
    model.add(Dense(hidden_dims, bias_initializer='zeros'))
    model.add(Dropout(0.7))
    model.add(Activation('relu'))    
    # We project onto a single unit output layer, and squash it with a sigmoid:
#    model.add(Dense(1, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dense(1, bias_initializer='zeros'))
    #model.add(Dense(1))
    model.add(Activation('sigmoid'))    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']) 
    #Load previuos     
    if previous_weight == True:
        model.set_weights(weights)
       # model.load_weights('classifier1.h5', by_name=True) 
        #os.remove('classifier1.h5')
    
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,  
                        validation_data=(x_test, y_test))
#                        callbacks=[early_stopping])
    weights = model.get_weights()
#    model.save_weights('classifier1.h5', overwrite=True)       
    acc_train = history.history['acc'][len(history.history['acc'])-1]*100#accuracy training 
    acc_test = history.history['val_acc'][len(history.history['val_acc'])-1]*100#accuracy testing
    print("Training accuracy", acc_train)
    print("Testing accuracy", acc_test) 
    return model, history, acc_train, acc_test, weights

#CNN  2
 
def train_classifier2(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight=False, weights=0):
    seed(randint(0, 50))
    set_random_seed(randint(0, 50))
    np.random.seed(randint(0, 50))
#    Initializer()
    # set parameters:
    maxlen = 50
    batch_size = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 5  #5

    
    print('---------------Build Classifier 2...')
#    print("embedding_matrix", embedding_matrix.shape)
#    print("x_train",x_train.shape)
#    print("y_train",y_train.shape)
#    print("x_test",x_test.shape)
#    print("y_test",y_test.shape)
    model = Sequential()    
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(embedding_matrix.shape[0],
                        embedding_matrix.shape[1],
#                        embedding_dims,#10
#                        weights=[embedding_matrix],
                        input_length=maxlen))
    
    model.add(Dropout(0.2))   
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    
    # We add a vanilla hidden layer:
    #model.add(Dense(hidden_dims))
    model.add(Dense(hidden_dims, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    model.add(Dense(hidden_dims, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    #Load previuos     
    if previous_weight == True:
        model.set_weights(weights)
       # model.load_weights('classifier1.h5', by_name=True) 
        #os.remove('classifier1.h5')
    
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,  
                        validation_data=(x_test, y_test))
#                        callbacks=[early_stopping])
    weights = model.get_weights()
#    model.save_weights('classifier1.h5', overwrite=True) 
    acc_train = history.history['acc'][len(history.history['acc'])-1]*100#accuracy training 
    acc_test = history.history['val_acc'][len(history.history['val_acc'])-1]*100#accuracy testing
    print("Training accuracy", acc_train)
    print("Testing accuracy", acc_test) 
    return model, history, acc_train, acc_test, weights



#CNN  3

def train_classifier3(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight=False, weights=0):
    seed(randint(0, 50))
    set_random_seed(randint(0, 50))
    np.random.seed(randint(0, 50))
#    Initializer()
#    # set parameters:
    maxlen = 50
    batch_size = 50
    filters = 200#250
    filters2 = 200#250
    kernel_size = 6#3
    hidden_dims = 200#250
    epochs = 12  #12

    
    print('---------------Build Classifier 3...')
#    print("embedding_matrix", embedding_matrix.shape)
#    print("x_train",x_train.shape)
#    print("y_train",y_train.shape)
#    print("x_test",x_test.shape)
#    print("y_test",y_test.shape)
    model = Sequential()    
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(embedding_matrix.shape[0],
                        embedding_matrix.shape[1],
#                        embedding_dims,#10
#                        weights=[embedding_matrix],
                        input_length=maxlen))   
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Dropout(0.8))    
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
       
    model.add(Conv1D(filters2,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D()) 
    
    # We add a vanilla hidden layer:
    #model.add(Dense(hidden_dims))
    model.add(Dense(hidden_dims, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dropout(0.8))
    model.add(Activation('relu'))    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1, kernel_initializer='random_uniform', bias_initializer='zeros'))
    #model.add(Dense(1))
    model.add(Activation('sigmoid'))    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #Load previuos     
    if previous_weight == True:
        model.set_weights(weights)
       # model.load_weights('classifier1.h5', by_name=True) 
        #os.remove('classifier1.h5')
    
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,  
                        validation_data=(x_test, y_test))
#                        callbacks=[early_stopping])
    weights = model.get_weights()
#    model.save_weights('classifier1.h5', overwrite=True)      
    acc_train = history.history['acc'][len(history.history['acc'])-1]*100#accuracy training 
    acc_test = history.history['val_acc'][len(history.history['val_acc'])-1]*100#accuracy testing
    print("Training accuracy", acc_train)
    print("Testing accuracy", acc_test) 
    return model, history, acc_train, acc_test, weights

#CNN  2
  
def train_classifier4(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight=False, weights=0):
    
    seed(randint(0, 50))
    set_random_seed(randint(0, 50))
    np.random.seed(randint(0, 50))
#    Initializer()
    # set parameters:
    maxlen = 50
    batch_size = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 5  #5
    
    print('---------------Build Classifier 4...')
#    print("embedding_matrix", embedding_matrix.shape)
#    print("x_train",x_train.shape)
#    print("y_train",y_train.shape)
#    print("x_test",x_test.shape)
#    print("y_test",y_test.shape)
    model = Sequential()    
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(embedding_matrix.shape[0],
                        embedding_matrix.shape[1],
#                        embedding_dims,#10
#                        weights=[embedding_matrix],
                        input_length=maxlen))
    
    model.add(Dropout(0.2))   
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    
    # We add a vanilla hidden layer:
    #model.add(Dense(hidden_dims))
    model.add(Dense(hidden_dims, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    model.add(Dense(hidden_dims, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    #Load previuos     
    if previous_weight == True:
        model.set_weights(weights)
       # model.load_weights('classifier1.h5', by_name=True) 
        #os.remove('classifier1.h5')
    
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,  
                        validation_data=(x_test, y_test))
#                        callbacks=[early_stopping])
    weights = model.get_weights()
#    model.save_weights('classifier1.h5', overwrite=True) 
    acc_train = history.history['acc'][len(history.history['acc'])-1]*100#accuracy training 
    acc_test = history.history['val_acc'][len(history.history['val_acc'])-1]*100#accuracy testing
    print("Training accuracy", acc_train)
    print("Testing accuracy", acc_test) 
    return model, history, acc_train, acc_test, weights    



