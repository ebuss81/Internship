
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


def gausMut1():
    mu = 0
    sig = 3
    print(np.random.normal(mu,sig,2))
    #return np.random.nomal(mu,sig,1)

def main():
    nn = Graph()
    nn.initializeGraph()

    #build Graphs
    DG = nx.DiGraph()
            #setlayers
    DG.add_nodes_from([1, 2, 3, 4], ActFunc = activations.noFunc, active = True, pos = (0,0),bias=0)
    DG.add_nodes_from([5, 6, 7, 8], ActFunc = activations.tanh, active = True, pos = (0,0),bias=0)
    DG.add_nodes_from([9, 10, 11,12], ActFunc = activations.tanh, active = True, pos = (0,0),bias=0.5)
    #set edges
    DG.add_edges_from([(1, 5), (1, 6), (1, 7), (1, 8), (2, 5), (2, 6), (2, 7), (2, 8),(3, 5), (3, 6), (3, 7), (3, 8),(4, 5), (4, 6), (4, 7), (4, 8)], weight = 1, active = True)
    DG.add_edges_from([(5, 9), (5, 10), (5, 11), (5, 12),(6, 9), (6, 10), (6, 11), (6, 12),(7, 9), (7, 10), (7, 11), (7, 12),(8, 9), (8, 10), (8, 11), (8, 12)], weight = -0.9, active = False)    
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
    print("hallo")

    #set active nodes, especially remove not needed In- and Outputneurons
    activeNodes = nx.get_node_attributes(nn.WG, "active")
    setactivieNodes = [True, True, True, False, True, True, True, True, True, True, True, False]
    activeNodes.update(zip(activeNodes,setactivieNodes))
    nx.set_node_attributes(nn.WG, activeNodes, "active")
    nx.set_node_attributes(nn.DG, activeNodes, "active")
    nn.removeNodes()
    nn.removeParentNodes()
    nn.removeEdge()

    for i in range(14):

        '''
        print(nx.get_edge_attributes(nn.WG, "weight"))
        gausMut(nn.WG, nn.DG)
        print(nx.get_edge_attributes(nn.WG, "weight"))
        '''
        '''
        print(nx.get_node_attributes(nn.WG, "bias"))
        nn.WG,nn.DG = gausMut_bias(nn.WG, nn.DG)
        print(nx.get_node_attributes(nn.WG, "bias"))
        '''


        #print(nx.get_node_attributes(nn.WG, "bias"))
        #nn.WG,nn.DG = changeBias(nn.WG,BR, nn.DG)
        #print(nx.get_node_attributes(nn.WG, "bias"))

        #nn.removeEdge()
        #nn.DG,edge = addEdge(nn.WG,nn.DG)
        #if edge != 0:
        #    nn.addEdgeFromParent(edge)
        
        #nn.WG, nn.DG = deleteEdge(nn.WG, nn.DG) 
        #nn.removeEdge()

        #print(nx.get_edge_attributes(nn.WG, "weight"))
        #nn.WG, nn.DG = changeWeights(nn.WG,WR, nn.DG)   
        #print(nx.get_edge_attributes(nn.WG, "weight"))
        modularity = nn.getCommunitiesModularity()
        ccost = getCost(nn.WG)
        print("Mod: ", modularity," ccost: ", ccost)
        nn.printGraph()



if __name__=="__main__":
    main()