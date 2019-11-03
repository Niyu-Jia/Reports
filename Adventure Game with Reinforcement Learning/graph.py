
from graphviz import Digraph
import Map
import os
directory=os.getcwd()
os.chdir(directory)

My_Map=Map.map_construction('map.xml')
map_info,obj_info=My_Map.construction()


graph=Digraph(comment="Map Graph",format='png')
#graph.attr(rankdir='LR')

name_list=list(map_info.index)

for k in range(len(name_list)):
    if name_list[k] in obj_info.keys():
        graph.node(name_list[k],name_list[k],color='red')
    else:
        graph.node(name_list[k],name_list[k])
        
        
for k in range(len(name_list)):
    info=map_info.iloc[k]
    info=info[info!=-1]
    
    for m in range(len(info)):
        direction=info.index[m]
        graph.edge(info.name,info[m],label=direction)

graph.render('./map graph.gv', view=True)  
print(graph.source)
