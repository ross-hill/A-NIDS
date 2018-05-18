#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This file defines the EnsemblePartialAutoencoders and PartialAutoencoder classes.
    
    Use of this code applies under an Apache 2.0 licence.
"""

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import random

class PartialAutoencoder:
    
    '''
     This class defines a partial autoencoder neural network, with three hidden layers
     which includes one middle layer. The partial part comes from the fact the input variables are 
     from a set of data X (dim(input) < dim(X)), and the output is this data X.
    '''
    
    def __init__(self, input_layer_n, hidden_nodes, output_layer_n, middle_nodes, n_epochs, activation = 'sigmoid', optimizer = "adadelta", loss = "mean_squared_error"):
        self.input_layer_n = input_layer_n
        self.hidden_nodes = hidden_nodes
        if middle_nodes is None:
            self.stacked = False
        else:
            self.middle_nodes = middle_nodes
            self.stacked = True
        self.output_layer_n = output_layer_n
        self.n_epochs = n_epochs
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        
    '''
     This function creates the neural network architecture, but does NOT train it.
    '''
        
    def create_network(self):
        if self.stacked:
            self.input_data = Input(shape = (self.input_layer_n,))
            enet = Dense(self.hidden_nodes, activation = self.activation, input_shape = (self.input_layer_n,))(self.input_data)
            mnet = Dense(self.middle_nodes, activation = self.activation, input_shape = (self.hidden_nodes,))(enet)
            dmnet = Dense(self.hidden_nodes, activation = self.activation, input_shape = (self.middle_nodes,))(mnet)
            dnet = Dense(self.output_layer_n, activation = self.activation, input_shape = (self.hidden_nodes,))(dmnet)
            self.model = Model(self.input_data, dnet)
            self.model.compile(optimizer = self.optimizer, loss = self.loss)
        else:
            self.input_data = Input(shape = (self.input_layer_n,))
            enet = Dense(self.hidden_nodes, activation = self.activation, input_shape = (self.input_layer_n,))(self.input_data)
            dnet = Dense(self.output_layer_n, activation = self.activation, input_shape = (self.hidden_nodes,))(enet)
            self.model = Model(self.input_data, dnet)
            self.model.compile(optimizer = self.optimizer, loss = self.loss)
            
    '''
     train_network only trains the model
    '''
    
    def train_network(self, X, Y, n_epochs = -1,  printing = 1):
        if n_epochs == -1:
            n_epochs = self.n_epochs
        self.model.fit(X, Y, n_epochs, verbose = printing)
        
    def predict(self, X):
        pred = self.model.predict(X)
        return pred
        
class EnsemblePartialAutoencoders:
    
    def __init__(self, n_features, n_blocks, hidden_nodes, middle_nodes = None, n_epochs = 5,random_state = 0, printing = 1):
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.hidden_nodes = hidden_nodes
        if middle_nodes is None:
            self.stacked = False
        else:
            self.stacked = True
            self.middle_nodes = middle_nodes
        self.n_epochs = n_epochs
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.printing = printing
    
    '''
     The following method creates the network of PartialAutoencoders
    '''
    
    def create_network(self,X_shape):
        n_total_features = X_shape[1]
        block_features = []
        included_features = []
        self.network = []
        
        # Randomly choose n_features for each block, until all features are included and the minimum n_blocks has been reached
        
        while len(included_features) < n_total_features or len(block_features) < self.n_blocks:
            features = np.array(random.sample(range(n_total_features), self.n_features)).astype(int)
            block_features.append(features)
            # Edit included_features
            for feat in features:
                if feat not in included_features:
                    included_features.append(feat)
            # initialise the model for the block
            #input_layer_n, hidden_nodes, middle_nodes, output_layer_n, n_epochs,
            block_model = PartialAutoencoder(input_layer_n = self.n_features, hidden_nodes = self.hidden_nodes, output_layer_n = n_total_features, middle_nodes = self.middle_nodes, n_epochs = self.n_epochs)
            block_model.create_network()
            self.network.append(block_model)
        self.block_features = np.array(block_features)
        self.n_blocks = self.block_features.shape[0]
        print("Final number of blocks: %s" % len(block_features))
        
    '''
     Train each of the networks. This involves looping through each of the block features and training in a for loop.
    '''    
    
    def train_network(self,X):
        for i in range(len(self.network)):
            block = self.block_features[i]
            Xi = X.iloc[:, block]
            self.network[i].train_network(Xi, X, printing = self.printing)
            print("Network %s / %s trained." % (i + 1, len(self.network)))
          
    def euclidean_metric(self, X, Y):
        '''
         Calculates the distance between two data points, X = [x1,..,xn], Y = [y1,...,yn]
         Can be changed easily to any other metric.
        '''
        dist = (X - Y)**2
        esum = np.mean(np.sum(dist))
        return esum
    
    def predict(self, X, numerical_names):
        catindex = np.array([v for v in range(X.shape[1]) if list(X)[v] not in numerical_names])
        self.predictions = np.zeros((X.shape[0], self.n_blocks))
        for (b, block) in zip(range(self.n_blocks), self.block_features):
            Xi = X.iloc[:, block]
            model = self.network[b]
            prediction = np.array(model.predict(Xi))
            for l in range(X.shape[0]):
                predl = prediction[l]
                predl[catindex] = (predl[catindex] > 0.5).astype(int)
                edist = self.euclidean_metric(predl, np.array(X.iloc[l, :]))
                self.predictions[l][b] = edist
    
        
        
