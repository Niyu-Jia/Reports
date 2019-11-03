

import os
directory=os.getcwd()
os.chdir(directory)
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

class map_construction():
    def __init__(self,file):
        """
        Initialize map
        """
        self.file=file

    def construction(self):
        """
        Construct map with xml file
        return: tuple of (map information(df) object information(dict))
        """
        tree = ET.parse(self.file)
        root = tree.getroot()
        room=[]
        surround=[]
        direction=['north','south','east','west']
        obj_info=dict()

        for child in root:
            # get room id and its surrounding room
            room.append(child.get('id'))
            surround.append([child.get(i) if type(child.get(i))!=type(None) else -1 for i in direction ]) # -1 means no access

            # get the object if it exits in the room
            for item in child:
                obj=item.get('name')
                obj_info[child.get('id')]=obj

        self.obj_info=obj_info
        self.map_info=pd.DataFrame(data=surround,columns=direction,index=room)
        self.room=room

        return (self.map_info,self.obj_info)

    def R_reset(self):
        """
        Initialize reward matrix
        return: R matrix(df)
        """

        R=self.map_info
        room_set=set(self.room)
        obj_set=set(self.obj_info.keys())
        non_reward=room_set-obj_set

        # Set the reward to 0 if the item is collected
        for i in list(non_reward):
            R=R.replace(i,0)

        return R

class Target(map_construction):
    def __init__(self,target,file):
        """
        Initialize target information
        """
        super().__init__(file)
        self.target=target

    def collection_target(self):
        """
        Read in start room and goal collection
        return: tuple of (starting room(str), your target object(list))
        """
        
        txt=open(self.target)
        txt_list=[]
        f=open(self.target)
        for line in f.readlines():
            txt_list.append(line.strip())

        start=txt_list[0]
        obj_target=txt_list[1:]
        #print(start)
        #print(obj_target)

        return (start,obj_target)
