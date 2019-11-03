
import os
directory=os.getcwd()
os.chdir(directory)
import xml.etree.ElementTree as ET
import pandas as pd
import Map
import numpy as np

class path():

    def __init__(self, action_space, learning_rate, reward_decay, e_greedy,My_Map):
        """
        Initialize game
        action_space: all possible actions(list)
        learning_rate: how fast the algorithm learns(float)
        reward_decay: how fast the reward decays(float)
        e_greedy: probability to choose best action instead of ramdom actions(float)
        My_Map: map object(map_construction)
        """
        
        self.actions = action_space 
        self.lr = learning_rate
        self.gamma = reward_decay
        self.My_Map=My_Map
        self.epsilon = e_greedy
        self.map_info,self.obj_info=My_Map.construction()
        self.R=self.My_Map.R_reset()
        self.Q=pd.DataFrame(columns=self.actions)

    def check_state(self, state):
        """
        Check whether this room has been explored before;
        if not, add a new record to Q matrix
        No return
        """
        
        if state not in self.Q.index:
        # append new state to q table
            self.Q = self.Q.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.Q.columns,
                    name=state,
                )
            )

    def routes(self,state):
        """
        Check whether the action is allowed
        return : the perimitted routes(list)
        """
        
        all_action=self.R.transpose()[state]
        permitted_action=list(all_action[all_action!=-1].index)
        route=permitted_action

        return route

    def act(self,state):
        """
        Choose action for state according to Q and restrictions
        return: one possible action(str)
        """
        
        #check whether the state is in the Q table
        self.check_state(state)
        all_action=self.R.transpose()[state]
        permitted_action=list(all_action[all_action!=-1].index)

        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.Q.loc[state, :].loc[permitted_action]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action=np.random.choice(permitted_action)

        return action
    
    
    def learning(self, state,epoch,goal_list):
        """
        Learning algorithm to update Q and R, generating paths
        return: final Q table(df)
        output: examples of the possible paths
        """
        
        f=open('output.txt','w')
        
        #iterate for 100 times
        for j in range(100):
            self.R=self.My_Map.R_reset()
            state='scullery'
            collection=[]
            last_state='None'
            
            # for each iteration, set a maximum steps to prevent infinity loop
            for i in range(epoch):
                
                #with restriction:whether the action is revisiting the previous room
                while True:
                    action=self.act(state)
                    next_state=self.map_info.loc[state,action]
                    if next_state!=last_state or len(self.routes(state))==1:
                        break
    
                next_action=self.act(next_state)
                string='You go '+action+' from '+state+' to '+next_state
                print(string,file=f)
                instant_R=self.R.loc[state,action]
    
                # Finding the item to decide the reward of room
                if type(instant_R)==str:
                    item=self.obj_info[next_state]
                    string='You collect the item: '+item
                    print(string,file=f)
                    collection.append(item)
                    self.R=self.R.replace(next_state,0)
                    r=100
                else:
                    r=0
    
                #update Qtable
                q_predict=self.Q.loc[state,action]
    
                if set(collection)==set(goal_list):
                    #Terminate the Game
                    q_target=r
                    print('Game End',file=f)
                    string="You collected all the items: "+str(collection)
                    print(string, file=f)
                    self.Q.loc[state, action] += self.lr * (q_target - q_predict)
                    break
    
                else:
                    # Continue the game
                    q_target = r + self.gamma * self.Q.loc[next_state, next_action]
                    self.Q.loc[state, action] += self.lr * (q_target - q_predict)
    
                    last_state=state
                    state=next_state
            print('#############################################################',file=f)
        
        f.close()
        return self.Q
