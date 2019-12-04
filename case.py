#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:53:49 2019

@author: niyu
"""

"""
Test case fot deep Q learning
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import env
import DQN


class test:
    def __init__(self,env,state0,network):
        """
        Do comparison among q-network and other portfolio
        state0: initial state
        network: trained q-network model
        """
        self.env=env
        self.state0=state0
        self.net=network
        self.envp1=np.array(env.p1)
        self.envp2=np.array(env.p2)
        self.envr1=np.array(env.r1)
        self.envr2=np.array(env.r2)
    
    def do_nothing(self,T):
        """
        doing nothing portfolio, starting form T
        """
        w1=self.state0[0:2]
        total1=self.state0[2] 
        num1=w1[0]*total1/self.envp1[T-1]
        num2=w1[1]*total1/self.envp2[T-1]
        
        port=num1*np.array(self.env.p1)+num2*np.array(self.env.p2)
        plt.plot(port)
        plt.title("Do-Nothing Portfolio Value")
        plt.xlabel("timestep")
        plt.show()
        
        return port
    
    def Q(self,T):
        """
        Q learning portfolio performance
        T: starting date (should be the first testing date)
        """
        current_state=self.state0
        value=[0]*(len(self.envr1)-T)
        value[0]=self.env.Value
        
        for t in range(T,len(self.envr1)-1000):
            q_values=self.net.predict(current_state.reshape(1,self.env.state_size))
            action=self.env.action[np.argmax(q_values)]
            next_state,reward,terminated,total_value=self.env.step(current_state,action,t)
            value[t-T+1]=total_value
            current_state=next_state
            
            if t%150==0:
                print(t," days finished")
            
        plt.plot(value)
        plt.title("Q-Learning portfolio value")
        plt.xlabel("timestep")
        plt.show()
        
        return value