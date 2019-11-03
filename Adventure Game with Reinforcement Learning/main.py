import pandas as pd
import numpy as np
import os
import Path
import Map

directory=os.getcwd()
os.chdir(directory)


# Construct the map information
My_Map=Map.map_construction('map.xml')

# Get the target collection list and start room
target=Map.Target('scenario.txt','map.xml')
start,goal_list=target.collection_target()

# Construct possible path
My_Path=Path.path(['north','south','east','west'],0.01,0.95,0.01,My_Map)

# Output a possible path with maximum k steps
k=150
Q=My_Path.learning(start,k,goal_list)
