import numpy as np
import random
import networkx as nx
from parameters import *
def deleteEdge (WG,DG): # gets list of Node activity
    try:
        edgeActivity = nx.get_edge_attributes(WG, "active") 
    except:
        print("noEdges anymore")
        return WG.copy   
    edgeActivity = nx.get_edge_attributes(WG, "active")    
    Activity = list(edgeActivity.values())
    #stop if only one edge
    if (sum(Activity) == 1) :       
        return WG.copy(),0   
    
    #if (all(np.logical_not(Activity))) :       
    #    return WG.copy(),0
    # set probability
    if random.random() < 0.1:
        activeEdges = np.where(Activity)
        idx = np.random.choice(activeEdges[0])
        Activity[idx] = False
    
    edgeActivity.update(zip(edgeActivity,Activity)) 
    nx.set_edge_attributes(WG,edgeActivity,"active")
    nx.set_edge_attributes(DG,edgeActivity,"active")
    return WG.copy(),DG.copy()

def addEdge(WG,DG): 
    idx = None
    try:
        edgeActivity = nx.get_edge_attributes(DG, "active")   
    except:
        print("noEdges anymore")
        return WG,DG

    edgelist = list(edgeActivity)   
    Activity = list(edgeActivity.values())    
    # stop if all True
    if (all(Activity)):         
        return DG.copy(),0
    # set probability
    if random.random() < 0.1: 
        activeEdges = np.where(np.logical_not(Activity))
        idx = np.random.choice(activeEdges[0])
        Activity[idx] = True
        edgeActivity.update(zip(edgeActivity,Activity)) 
        nx.set_edge_attributes(DG,edgeActivity,"active")
    if idx is None:
        return DG.copy(),0 
    else:
        return DG.copy(),edgelist[idx]


#TODO only DG could be enough
def changeWeights(WG, weightsRange, DG):   #here directiory not list
    try:
        Activity = nx.get_edge_attributes(WG, "active")   
    except:
        print("noEdges anymore")
        return WG,DG
    for edge, active in Activity.items():       #go through all active edges
        if active == True:
            # set probability
            if random.random() < 2./WG.number_of_edges(): #0.1
                curr_weight = WG.get_edge_data(*edge)["weight"]
                weightsRangeIdx = np.where(weightsRange == curr_weight)
                try:
                    if random.random() < 0.5:
                        WG[edge[0]][edge[1]] ["weight"] = weightsRange[weightsRangeIdx[0]+1][0]
                        DG[edge[0]][edge[1]] ["weight"] = weightsRange[weightsRangeIdx[0]+1][0]
                        #print("++++")

                    else:
                        WG[edge[0]][edge[1]] ["weight"] = weightsRange[weightsRangeIdx[0]-1][0]
                        DG[edge[0]][edge[1]] ["weight"] = weightsRange[weightsRangeIdx[0]-1][0]
                        #print("----")

                except:
                    pass
    return WG,DG

def changeBias(WG, biasRange,DG):
    numInNeurons = 3 # in parent graph
    try:
        BiasValue = nx.get_node_attributes(DG, "bias")   
    except:
        print("noEdges anymore")
        return WG,DG
    
    BiasValue = nx.get_node_attributes(DG, "bias")
    biases = list(BiasValue.values())  
    for idx,bias in enumerate(biases[numInNeurons-1:]):     #input neurons have no bias
        if random.random() <= 0.1:
            curr_bias = DG.nodes[idx+numInNeurons+1]["bias"]
            biasRangeIdx = np.where(biasRange == curr_bias)
            try:
                if random.random() <0.5:
                    WG.nodes[idx+numInNeurons+1]["bias"] = biasRange[biasRangeIdx[0]+1][0]   
                    DG.nodes[idx+numInNeurons+1]["bias"] = biasRange[biasRangeIdx[0]+1][0]      
                    #print("++++")
                else:
                    WG.nodes[idx+numInNeurons+1]["bias"] = biasRange[biasRangeIdx[0]-1][0]   
                    DG.nodes[idx+numInNeurons+1]["bias"] = biasRange[biasRangeIdx[0]-1][0]   
                    #print("----")
            except:
                pass
    return WG,DG

def gausMut_weigth(WG,DG):
    mu = 0
    sig = 3
    try:
        Activity = nx.get_edge_attributes(WG, "active")   
    except:
        print("noEdges anymore")
        return WG,DG
    for edge, active in Activity.items():       #go through all active edges
        if active == True:
            # set probability
            if random.random() < 2./WG.number_of_edges():
                x_hat = np.random.normal(mu,sig,1)
                curr_weight = WG.get_edge_data(*edge)["weight"] 
                new_weight = curr_weight + x_hat
                if new_weight[0] < -MAX_R:                   
                    WG[edge[0]][edge[1]] ["weight"] = -MAX_R
                    DG[edge[0]][edge[1]] ["weight"] = -MAX_R
                elif new_weight[0] > MAX_R:
                    WG[edge[0]][edge[1]] ["weight"] = MAX_R
                    DG[edge[0]][edge[1]] ["weight"] = MAX_R
                else:
                    WG[edge[0]][edge[1]] ["weight"] = new_weight[0]
                    DG[edge[0]][edge[1]] ["weight"] = new_weight[0]
                #print("++++")

    return WG,DG

def gausMut_bias(WG,DG):
    mu = 0
    sig = 3
    numInNeurons = 4 # in parent graph
    try:
        BiasValue = nx.get_node_attributes(DG, "bias")   
    except:
        print("noEdges anymore")
        return WG,DG
    
    BiasValue = nx.get_node_attributes(DG, "bias")
    biases = list(BiasValue.values())  
    for idx,bias in enumerate(biases[numInNeurons-1:]):     #input neurons have no bias
        if random.random() <= 0.1:
            curr_bias = DG.nodes[idx+numInNeurons+1]["bias"]
            x_hat = np.random.normal(mu,sig,1)
            new_bias = curr_bias + x_hat
            if new_bias[0] < -MAX_R:                   
                WG.nodes[idx+numInNeurons+1]["bias"] = -MAX_R  
                DG.nodes[idx+numInNeurons+1]["bias"] = -MAX_R
            elif new_bias[0] > MAX_R:
                WG.nodes[idx+numInNeurons+1]["bias"] = MAX_R  
                DG.nodes[idx+numInNeurons+1]["bias"] = MAX_R
            else:
                WG.nodes[idx+numInNeurons+1]["bias"] = new_bias[0]  
                DG.nodes[idx+numInNeurons+1]["bias"] = new_bias[0]
    return WG,DG