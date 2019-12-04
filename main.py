#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:57:24 2019

@author: niyu
"""

"""
Main program for portfolio management with deep Q learning
"""

import os
os.chdir("/home/niyu/Documents/803/Project/")

import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow import keras


import env 
import env_sharpe
import DQN
import case

return1=pd.read_csv('./lowreturn.csv',index_col=0)
return2=pd.read_csv('./highreturn.csv',index_col=0)
price1=pd.read_csv('./lowprice.csv',index_col=0)
price2=pd.read_csv('./highprice.csv',index_col=0)
act=[-0.15,-0.1,-0.05,0,0.05,0.1,0.15]
Return=return1.merge(return2,how='inner', left_index=True, right_index=True)
Price=price1.merge(price2,how='inner', left_index=True, right_index=True)
assert len(Return)==len(Price)

cutpoint=int(0.7*len(Return))
Return_train=Return[0:cutpoint]
Price_train=Price[0:cutpoint]
Return_test=Return[cutpoint:]
Price_test=Price[cutpoint:]

state_size=7
lag=int((state_size-3)/2)
#ticker1=np.random.choice(return1.columns)
#ticker2=np.random.choice(return2.columns)

ticker1='AEP'
ticker2='AAPL'

environment=env.env(act,state_size,ticker1,ticker2,Return_train,Price_train,100)

optimizer = Adam(learning_rate=0.05)
agent = DQN.Agent(environment, optimizer)



##############################################################################
"""
Training model
"""
keras.backend.clear_session()
batch_size = 8
num_of_episodes = 10
timesteps_per_episode = len(Return_train)
agent.q_network.summary()
print("#######################################################################")
print("Constructing portfolio with",ticker1,'and',ticker2)
#clear expirience storage
agent.store_clear()
training_size=151

date=0 # starting training date
state = environment.Reset(0,0)     # Reset the environment
new_w=agent.q_network.get_weights()

for timestep in range(0, training_size*5):
    
    # Initialize variables
    reward = 0
    
    # Run Action
    action = agent.act(state)
    
    # Take action    
    next_state, reward, terminated,total= environment.step(state,action,timestep+date+3) 
    agent.store(state, action, reward, next_state, terminated)
    
    state = next_state
    date=date+1
    
    if len(agent.valid_experience) > batch_size:
        new_w=agent.retrain(batch_size)
    
    if timestep%25==0:
        agent.alighn_target_model(new_w)
        #print("updating weights")    
    
    if (timestep+1)%150==0:
        print("one episode finished,reset weights")
        state = environment.Reset(date)  


nn_weights=agent.q_network.get_weights()
np.save('weights.npy',nn_weights)
#nn_weights=np.load('./weights.npy',allow_pickle = True)

##############################################################################
"""
Model Comparison
"""

network=agent.q_network
#network.set_weights(nn_weights)
test_env=env.env(act,7,ticker1,ticker2,Return_test,Price_test,100)
state0=test_env.Reset(0)
comparison=case.test(test_env,state0,network)
Do_Nothing=comparison.do_nothing(30)
Q_Learning=comparison.Q(30)





















