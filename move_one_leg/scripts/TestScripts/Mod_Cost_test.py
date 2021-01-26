

import sys
sys.path.append('src/move_one_leg/scripts')

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#from Parameter import *
#from mutations import *
from NeuralNet.Graph import Graph
from MAP_Elite.connectioncost import *
import NeuralNet.activations as activations
from parameters import *
def main():

    #build Graphs
    DG = nx.DiGraph()
            #setlayers
    DG.add_nodes_from([1, 2, 3, 4], ActFunc = activations.noFunc, active = True, pos = (0,0),bias=0)
    DG.add_nodes_from([5, 6, 7, 8], ActFunc = activations.tanh, active = True, pos = (0,0),bias=1)
    DG.add_nodes_from([9, 10, 11,12], ActFunc = activations.tanh, active = True, pos = (0,0),bias=1)
    #set edges
    DG.add_edges_from([(1, 5), (1, 6), (1, 7), (1, 8), (2, 5), (2, 6), (2, 7), (2, 8),(3, 5), (3, 6), (3, 7), (3, 8),(4, 5), (4, 6), (4, 7), (4, 8)], weight = -15, active = True)
    DG.add_edges_from([(5, 9), (5, 10), (5, 11), (5, 12),(6, 9), (6, 10), (6, 11), (6, 12),(7, 9), (7, 10), (7, 11), (7, 12),(8, 9), (8, 10), (8, 11), (8, 12)], weight = -15, active = True)    
    pos= [None] *12  

    nn = Graph()
    nn.DG = DG  
    nn.WG = DG.copy()

    WG = DG.copy()

    c=0 # set positions of neurons
    for i in range(NUMLAYER):
        for j in range(NUMNEURONS):
            pos[c]= np.array([j,i])
            c+=1
    nn.setNodeAttribute(pos,"pos")


    #set active nodes, especially remove not needed In- and Outputneurons
    activeNodes = nx.get_node_attributes(nn.WG, "active")
    setactivieNodes = [True, True, True, False, True, True, True, True, True, True, True, False]
    activeNodes.update(zip(activeNodes,setactivieNodes))
    nx.set_node_attributes(nn.WG, activeNodes, "active")
    nx.set_node_attributes(nn.DG, activeNodes, "active")
    nn.removeNodes()
    nn.removeParentNodes()
    

    # print and get modularity
    nn.printGraph()  
    modularity = nn.getCommunitiesModularity()
    ccost = getCost(nn.WG)
    print("Mod: ", modularity," ccost: ", ccost)

    #set active edges
    activeEdges = nx.get_edge_attributes(nn.DG, "active")
    print(activeEdges)
    setactivieEdges = [True, False, False, False, False, False, False, False, False, False, True, False,
                        False, False, False, False, False, True, False, False, False, True, False, False]
    setactivieEdges = [True, True, True, True, True, True, True, True, True, True, True, True,
                        False, False, False, False, False, False, False, False, False, False, False, False]
    activeEdges.update(zip(activeEdges,setactivieEdges))
    nx.set_edge_attributes(nn.WG, activeEdges, "active")


    nn.removeEdge()
    modularity = nn.getCommunitiesModularity()
    ccost = getCost(nn.WG)
    print("Mod: ", modularity," ccost: ", ccost)
    nn.printGraph()    

    '''
    global nn, data, result, ME


    #build Graphs
    DG = nx.DiGraph()

    #make Fully connectet 3x4
    #setlayers
    DG.add_nodes_from([1, 2, 3, 4], ActFunc = activations.noFunc, active = True, pos = (0,0),bias=0)
    DG.add_nodes_from([5, 6, 7, 8], ActFunc = activations.modSigmoid, active = True, pos = (0,0),bias=1)
    DG.add_nodes_from([9, 10, 11,12], ActFunc = activations.modSign, active = True, pos = (0,0),bias=1)
    #set edges
    DG.add_edges_from([(1, 5), (1, 6), (1, 7), (1, 8), (2, 5), (2, 6), (2, 7), (2, 8),(3, 5), (3, 6), (3, 7), (3, 8),(4, 5), (4, 6), (4, 7), (4, 8)], weight = 10, active = False)
    DG.add_edges_from([(5, 9), (5, 10), (5, 11), (5, 12),(6, 9), (6, 10), (6, 11), (6, 12),(7, 9), (7, 10), (7, 11), (7, 12),(8, 9), (8, 10), (8, 11), (8, 12)], weight = 10, active = False)    
    pos= [None] *12   

    nn = Graph()
    nn.DG = DG  
    nn.WG = DG.copy()
    
    c=0
    for i in range(3):
        for j in range(4):
            pos[c]= np.array([j,i])
            c+=1
    nn.setNodeAttribute(pos,"pos")

    
    nn.WG = DG.copy()

    #Remove all unneeded neurons from the in- and outputlayer
    activeNodes = nx.get_node_attributes(nn.WG, "active")
    setactivieNodes = [True, False, False, True, True, True, True, True,False, True, False, False]
    activeNodes.update(zip(activeNodes,setactivieNodes))
    nx.set_node_attributes(nn.WG, activeNodes, "active")
    nx.set_node_attributes(nn.DG, activeNodes, "active")
    nn.removeNodes()
    nn.removeParentNodes()

    # set all edges as positive, because they are initialized as negative
    activeEdges = nx.get_edge_attributes(nn.WG, "active")
    acts = [True,True,True,True,True,True,True,True,True,True,True,True,]
    activeEdges.update(zip(activeEdges,acts))
    nx.set_edge_attributes(nn.WG, activeEdges, "active")
    print(nx.get_edge_attributes(nn.WG,"active"))
    nn.removeEdge()

    # print and get modularity
    nn.printGraph()  
    modularity = nn.getCommunitiesModularity()
    ccost,nn.WG = getCost(nn.WG)
    print("Mod: ", modularity," ccost: ", ccost)
    nn.printGraph()    


    '''


if __name__=="__main__":
    main()