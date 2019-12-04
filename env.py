#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:32:37 2019

@author: niyu
"""

"""
define environment 
"""

import numpy as np
import pandas as pd


class env:
    def __init__(self,act,state_size,ticker1,ticker2,Return,Price,Value):
        """
        act: list of all possible actions
        state_size: size of input state
        ticker1,ticker2: name of both stocks
        Return: df of all stock returns
        Price: df of all stock prices
        Value: inital total value of money
        """
        self.action_size=len(act)
        self.action=act
        self.state_size=state_size
        self.r1,self.r2=np.array(Return[ticker1]),np.array(Return[ticker2])
        self.p1,self.p2=np.array(Price[ticker1]),np.array(Price[ticker2])
        self.Value=Value
        self.total_reward=0
        self.n=int((state_size-3)/2) #how many days of history we are using

    
    def Reset(self,T=0,random=1):
        """
        Reset initial state
        n: number of days of history
        T: reset weights for time T, typically the starting date of training
        """
        if random==1:
            w0=np.random.uniform(0,1)
        else:
            w0=0.5
        self.w=[w0,1-w0]
        TT=self.n+1+T # so you should just start at T=0
        self.total_reward=0
        
        his1=self.r1[TT-self.n-1:TT-1]
        his2=self.r2[TT-self.n-1:TT-1]
        
        initial=np.concatenate((self.w,self.Value,his1,his2), axis=None)
        
        return initial
        
    def random_act(self):
        """
        randomly choose an action
        """
        return np.random.choice(self.action)

    def step(self,now_state,action,T):
        """
        return next_state, reward, terminated information if we take action from T-1 to T
        """
        
        #sell low beta, buy in corresponding high beta
        T=T
        w1=now_state[0:2]
        total1=now_state[2] #previous state total value
        
                
        #update weights
        new_w=w1[0]+action
        w2=[new_w,1-new_w]
        
        num1=w2[0]*total1/self.p1[T-1]
        num2=w2[1]*total1/self.p2[T-1]
        
        print(self.p1[T-1])
        
        value1=num1*self.p1[T]
        value2=num2*self.p2[T]
        total2=value1+value2 # current total value
        

        his1=self.r1[T-self.n:T]
        his2=self.r2[T-self.n:T]
        
        #update state 
        next_state=np.concatenate((w2,total2,his1,his2), axis=None)
        
        #reward function
        reward=total2-total1
        self.total_reward=self.total_reward+reward
        #print("take action:",action," and got reward:",reward)
        
        #judge termination state:
        if T==len(self.p1)-3:
            terminated=True
        else:
            terminated=False
        
        return next_state,reward,terminated,total2
        

