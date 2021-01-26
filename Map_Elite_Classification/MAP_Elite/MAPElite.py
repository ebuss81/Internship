import sys
sys.path.append('/home/ed/Dokumente/Codes/Map_Elite_Rework')


import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from NeuralNet.Graph import Graph
import NeuralNet.activations as activations
import networkx as nx
import pickle
from datetime import datetime

from MAP_Elite.connectioncost import *

class MapElite:
    def __init__(self, gridsize = np.array([10,10]) ):
        self.size = gridsize
        self.Grid = np.zeros(self.size)
        self.archive  = [[None for i in range(self.size[0])] for j in range(self.size[1])] 
        #self.nn = Graph()
        self.gss = 0.02    #gridstepsize, depending on resolution. for 10x10 in [0,1] gss =0.1 for 50x50 in [0,1] gss = 0.02
        self.data = []
        self.result = []
        self.bf = [] #best fitness
        self.af = [] #average fitness

    def plotGrid(self):
        min_val, max_val = 0, 1

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(self.Grid,cmap=plt.cm.Blues,extent=[min_val, max_val, max_val, min_val], origin="upper")

        clb=fig.colorbar(cax)
        cax.set_clim(0, 1)
        clb.set_label('Fitness', rotation=90)
        '''
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                c = self.Grid.transpose()[i][j]
                ax.text((i+0.5)/self.size[0], (j+0.5)/self.size[1], str(c), va='center', ha='center')
        '''
        
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(max_val, min_val)
        ax.set_xticks(np.arange(10.)/10)
        ax.set_yticks(np.arange(10.)/10)
        plt.xlabel("Connection cost")
        plt.ylabel("Modularity")

        plt.show()

        
    def insertIndividuum(self, f1,f2,fitness):
        fitness = np.round(fitness[0],3)
        f1 = int(f1/self.gss)
        f2 = int(f2/self.gss)
        if self.Grid[f1,f2] < fitness:
            self.Grid[f1,f2] = fitness
            print("fit insert")


    def archiv (self, f1,f2,WG, fitness):
        fitness = np.round(fitness[0],3)
        f1 = int(f1/self.gss)
        f2 = int(f2/self.gss)
        #print("f1",f1,"f2",f2)
        if self.Grid[f1,f2] < fitness:
            self.archive[f1][f2] = WG
            print("graph insered")

    def getArchiv(self):
        return self.archive

    def getIndi(self,f1,f2):
        try:
            indi = self.archive[f1][f2]  
            return indi.copy()
        except:
            print("no Graph in Slot")
            return None

    def getAllIndi(self):
        allIndis = np.array(np.where(self.Grid >0))
        allIndis = allIndis.transpose()
        return allIndis

    def largest_indices(ary, n):
        #"""Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    def getBest(self,x):
        maxidx = np.argmax(self.Grid)
        print(maxidx)
        #print((-self.Grid.flatten()).argsort()[:4])
        maxidx = (-self.Grid.flatten()).argsort()[:10]
        print(maxidx)
        row = int(maxidx[x]/self.size[0])
        col = maxidx[x] % self.size [1]
        return self.archive[row][col].copy()


    def saveInFile(self,num):
        if num == 0:
            pickle_out = open("/home/ed/Dokumente/Codes/Map_Elite_Rework/results/fit0.pickle","wb")
            pickle.dump(self.Grid,pickle_out,protocol= 0)
            pickle_out.close()

            pickle_out1 = open("/home/ed/Dokumente/Codes/Map_Elite_Rework/results/archiv0.pickle","wb")
            pickle.dump(self.archive,pickle_out1,protocol= 0)
            pickle_out1.close()
        if num == 1:
            pickle_out = open("/home/ed/Dokumente/Codes/Map_Elite_Rework/results/fit1.pickle","wb")
            pickle.dump(self.Grid,pickle_out,protocol= 0)
            pickle_out.close()

            pickle_out1 = open("/home/ed/Dokumente/Codes/Map_Elite_Rework/results/archiv1.pickle","wb")
            pickle.dump(self.archive,pickle_out1,protocol= 0)
            pickle_out1.close()

    def readFile(self,num):
        if num == 0:
            pickle_in = open("/home/ed/Dokumente/Codes/Map_Elite_Rework/results/fit0.pickle","rb")
            self.Grid = pickle.load(pickle_in)
            
            pickle_in1 = open("/home/ed/Dokumente/Codes/Map_Elite_Rework/results/archiv0.pickle","rb")
            self.archive = pickle.load(pickle_in1)       
        if num == 1:
            pickle_in = open("/home/ed/Dokumente/Codes/Map_Elite_Rework/results/fit1.pickle","rb")
            self.Grid = pickle.load(pickle_in)
            
            pickle_in1 = open("/home/ed/Dokumente/Codes/Map_Elite_Rework/results/archiv1.pickle","rb")
            self.archive = pickle.load(pickle_in1)   

    def getRandomIndi(self):
        allIndis = np.array(np.where(self.Grid >0))
        allIndis = allIndis.transpose()
        randomIdx = allIndis[np.random.choice(allIndis.shape[0], 1, replace=False)]
        return self.archive [randomIdx[0][0]][randomIdx[0][1]].copy()

    def evaluate(self,prediction):
        return (self.result == prediction.transpose()).sum() / len(self.data[0]),

    def initialzePopulation(self,numIter,nn):
        for i in range(numIter):
            print(i)
            nn.randomIndi()
            fit = self.evaluate(nn.predict(self.data))
            modularity = nn.getCommunitiesModularity()
            ccost = getCost(nn.WG)
            #print("mod: ", modularity, " cc: ",ccost)
            try:
                self.archiv(modularity,ccost, nn.WG,fit)
                self.insertIndividuum(modularity,ccost,fit)
            except:
                print("Somethin went wron in initialzePopulation")  
            self.recordFitness()          
        self.plotGrid()
        self.plotFitnes()
        self.saveInFile(0)
        self.saveFitness()

    def recordFitness(self):
        bestfit = np.max(self.Grid)
        self.bf.append(bestfit)
        allFitnesses = np.where(self.Grid >0)
        self.af.append(np.mean(self.Grid[allFitnesses]))


    def saveFitness(self):
        file_name = "Map_Elite_Rework/storage/Fitness" + str(datetime.now()) + ".txt"
        file = open (file_name,"w+")
        for k in range(len(self.bf)):
            file.write("%d %f %f  \n"%(k,self.bf[k], self.af[k]))
        file.close()

    def plotFitnes(self):
        x= np.arange(len(self.bf))
        fig, ax = plt.subplots()
        ax.plot(x, self.bf,label="best")
        ax.plot(x, self.af,label="average")
        ax.legend()
        ax.set(xlabel='time (s)', ylabel='Fitness',
            title='Fitnes')
        ax.grid()

        #fig.savefig("test.png")
        plt.show()


    def run(self,numIter,nn):
        self.readFile(0)
        for i in range(numIter):
            print("Iteration:",i)
            #if(i%1000 == 0):
            #    ME.plotGrid()
            nn.WG = self.getRandomIndi()

            activeEdges = nx.get_edge_attributes(nn.WG,"active")
            nx.set_edge_attributes(nn.DG, activeEdges, "active")

            WightEdges = nx.get_edge_attributes(nn.WG,"weight")
            nx.set_edge_attributes(nn.DG, WightEdges, "weight")

            biasNodes = nx.get_node_attributes(nn.WG,"bias")
            nx.set_node_attributes(nn.DG, biasNodes, "bias")


            nn.WG,nn.DG = changeBias(nn.WG,BR, nn.DG)
            nn.DG,edge = addEdge(nn.WG,nn.DG)
            if edge != 0:
                nn.addEdgeFromParent(edge)
            nn.WG, nn.DG = deleteEdge(nn.WG, nn.DG) 
            nn.removeEdge()

            nn.WG, nn.DG = changeWeights(nn.WG,WR, nn.DG)   
            modularity = 0
            try:
                modularity = nn.getCommunitiesModularity() 
            except:
                print("no mudularity, becouse no edges or nodes")  

            fit = self.evaluate(nn.predict(self.data))
            ccost = getCost(nn.WG)
            #print("Mod: ", modularity," ccost: ", ccost)
            try:
                self.archiv(modularity,ccost, nn.WG,fit)
                self.insertIndividuum(modularity,ccost,fit)
            except:
                print("not iserted, out of range, correct normalization?")
                
            #reset activity of DG to false, to overwrite it later with TRUE after mutatinos
            activeEdges = nx.get_edge_attributes(nn.DG,"active")
            setactivieEdges = [False, False, False, False, False, False, False, False, False,False,False,False]
            activeEdges.update(zip(activeEdges,setactivieEdges))
            nx.set_edge_attributes(nn.DG, activeEdges, "active")
            self.recordFitness()
        self.plotGrid()
        self.saveInFile(1)
        self.plotFitnes()
        self.saveFitness()

    def analyze1(self):
        indis = self.getAllIndi()
        print(indis)
        occupied = len(indis)/len(self.Grid)**2
        print("Occupied", occupied)
        bestfit = np.max(self.Grid)
        allFitnesses = np.where(self.Grid >0)
        averagefit = np.mean(self.Grid[allFitnesses])
        print("Fitneess : best: ", bestfit, " average: ", averagefit)
        highperf= sum(self.Grid >= 0.994)
        print(len(highperf))
        self.plotGrid()

