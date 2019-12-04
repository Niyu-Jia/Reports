#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:47:25 2019

@author: niyu
"""

"""
Develope agent for reinforcement learning
"""

from IPython.display import clear_output
from collections import deque
import tensorflow as tf
import numpy as np
import env
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
import random


class Agent:
    def __init__(self, env, optimizer):
        
        # Initialize atributes
        self.length=len(env.r1)
        self.env=env
        self._state_size = self.env.state_size
        self._action_size = self.env.action_size
        self._optimizer = optimizer
        
        self.experience_replay = [0]*100
        self.valid_experience=self.experience_replay.remove(0)
        
        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.epsilon = 0.1
        
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model(self.q_network.get_weights())
        
    
    def store(self, state, action, reward, next_state, terminated):
        tup=(state, action, reward, next_state, terminated)
        No=random.randint(0,99)
        self.experience_replay[No]=tup
        #print(No)
        self.valid_experience=[i for i in self.experience_replay if i!=0]
        
    
    def store_clear(self):
        self.experience_replay = [0]*100
        self.valid_experience=[0]*100
    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Dense(20,input_shape=(7,),activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='/home/niyu/Documents/803/Project')
        return model

    def alighn_target_model(self,weights):
        """
        self.target_network: a keras model
        self.q_network: a keras model
        weights: pervious trained model weights
        """
        self.target_network.set_weights(weights)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            #print('Taking random action this time')
            return self.env.random_act() # randomly choose an action
        #else:
            #print("Taking best action...")
            
        q_values = self.q_network.predict(state.reshape(1,self.env.state_size)) # choose best action according to the Q 
        #print(q_values)
        action_number= np.argmax(q_values[0]) # returning the corresponding action
            
        return self.env.action[action_number]
    
    def retrain(self, batch_size):
        """
        Retrain for 1 epoch and certain batch size
        """
        replay_No=np.random.choice(len(self.valid_experience), batch_size,replace=False)
        minibatch = [self.valid_experience[i] for i in list(replay_No)]
        # sample from experience with a ceratin batch size
        
        for state, action, reward, next_state, terminated in minibatch:
            action_No=self.env.action.index(action)
            
            target = self.q_network.predict(state.reshape(1,self.env.state_size)) # predict q_network result from previous model
            
            if terminated:
                target[0][action_No] = reward
            else:
                t = self.target_network.predict(next_state.reshape(1,self.env.state_size))
                target[0][action_No] = reward + self.gamma * np.max(t)
            
            self.q_network.fit(state.reshape(1,self.env.state_size), target, epochs=1, verbose=0)
        
        w=self.q_network.get_weights()
        keras.backend.clear_session()
        
        return w


            
