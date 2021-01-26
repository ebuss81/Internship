import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from NeuralNet.Graph import Graph
from MAP_Elite.MAPElite import MapElite
from MAP_Elite.connectioncost import *
import NeuralNet.activations as activations
from utils import*
from parameters import *


def main():
    dataPaths = ["/home/ed/Dokumente/Codes/Classification_Network_Deap/data2.txt"]
    titles = ["Neural Network Classification"]
    data, result  = loadData(dataPaths[0])

    #build NeuralNet
    NN = Graph()
    NN.initializeGraph()
    

    #build MAP_Elite
    #size = np.array([10,10])
    size = np.array([50,50])
    
    ME = MapElite(size)
    ME.data, ME.result = data, result
    
    #Run MAP Elite
    #ME.initialzePopulation(100,NN)
    #ME.run(20000,NN)

    #Analyze
    ME.readFile(1)
    NN.WG = ME.getIndi(33,0)
    #NN.printGraph()
    ME.analyze1()
    '''
    NN.WG = ME.getBest(0)
    #nn.printGraph()
    ME.plotGrid()
    modularity = NN.getCommunitiesModularity()
    ccost = getCost(NN.WG)
    print("Fit :",ME.evaluate(NN.predict(data)), " Mod: ", modularity," ccost: ", ccost)
    print(nx.get_edge_attributes(NN.WG,"weight"))
    print(nx.get_node_attributes(NN.WG,"bias"))
    plotNN(NN, NN.WG,data)
    plotResults(data, result, NN.predict(data))
    plt.show()
    NN.printGraph()
    '''
    '''
    ME.readFile(1)
    NN.WG = ME.getIndi(5,5)
    print(evaluate2(NN.predict(data)))
    NN.printGraph()
    ME.plotGrid()
    '''
    
if __name__ == "__main__":
    main()
