import sys
sys.path.append('src/move_one_leg/scripts')

import rospy
import numpy as np
import random
import matplotlib.pyplot as plt
from NeuralNet.Graph import Graph
import pickle
import NeuralNet.activations as activations
import networkx as nx
from utils import*
from MAP_Elite.connectioncost import *
import time
from datetime import datetime
class MapElite:
    def __init__(self, gridsize = np.array([10,10]) ):
        self.size = gridsize
        self.Grid = np.zeros(self.size)
        self.GridRC = self.Grid # used to recalc the grid 
        self.archive  = [[None for i in range(self.size[0])] for j in range(self.size[1])] 
        self.archiveRC = [[None for i in range(self.size[0])] for j in range(self.size[1])] 
        #self.nn = Graph()
        self.gss = 0.1    #gridstepsize, depending on resolution. for 10x10 in [0,1] gss =0.1 for 50x50 in [0,1] gss = 0.02
        self.data = []
        self.result = []
        self.bestfitnesses = []
        self.averagefitnesses = []
        self.weights = [[None for i in range(self.size[0])] for j in range(self.size[1])] 
        self.startduration = 3
        self.duration = np.ones(self.size) * self.startduration


        self.samefitnesses = []

    def plotGrid(self):
        min_val, max_val = 0, 1

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(self.Grid,cmap=plt.cm.Blues,extent=[min_val, max_val, max_val, min_val], origin="upper")

        clb = fig.colorbar(cax)
        cax.set_clim(-0, 13)
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

    def plotGridRC(self):
        min_val, max_val = 0, 1

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(self.GridRC,cmap=plt.cm.Blues,extent=[min_val, max_val, max_val, min_val], origin="upper")

        fig.colorbar(cax)
        cax.set_clim(-0, 130)

        
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                c = self.GridRC.transpose()[i][j]
                ax.text((i+0.5)/self.size[0], (j+0.5)/self.size[1], str(c), va='center', ha='center')
        
        
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(max_val, min_val)
        ax.set_xticks(np.arange(10.)/10)
        ax.set_yticks(np.arange(10.)/10)
        plt.xlabel("Connection cost")
        plt.ylabel("Modularity")

        plt.show()


        
    def insertIndividuum(self, f1,f2,fitness,RC,init,duration): #depends on featurevalue
        #RC used to recalculate MAP
        fitness = np.round(fitness[0],3)
        f1 = int(f1/self.gss)
        f2 = int(f2/self.gss)
        #print("f1",f1,"f2",f2)
        if not RC:
            try:
                if self.Grid[f1,f2] < fitness:
                    self.Grid[f1,f2] = fitness
                    if init:
                        self.duration[f1,f2] = self.startduration+1
                        #print(self.duration)
                    else:
                        if self.duration[f1,f2] <= 60:
                            if duration > 10:
                                self.duration[f1,f2] = duration +1 #duration + np.round(duration/10,0)
                            else:
                                self.duration[f1,f2] = duration+1
                    print("fit insert,duration insert")
            except:
                print("out of grid range")
        else:
            try:
                if self.GridRC[f1,f2] < fitness:
                    self.GridRC[f1,f2] = fitness
                    print("fit insert")
            except:
                print("out of grid range")



    def archiv(self, f1,f2,WG, fitness,RC):
        #RC used to recalculate MAP
        fitness = np.round(fitness[0],3)
        f1 = int(f1/self.gss)
        f2 = int(f2/self.gss)
        #print(self.archiveRC)
        #print("f1",f1,"f2",f2)
        if not RC:
            try:
                if self.Grid[f1,f2] < fitness:
                    self.archive[f1][f2] = WG
                    self.weights[f1][f2] = nx.get_edge_attributes(WG,"weight")
                    print("graph insered")
            except:
                print("out of grid range")
        else:
            print("f1",f1,"f2",f2)
            try:
                if self.GridRC[f1,f2] < fitness:
                    self.archiveRC[f1][f2] = WG
                    self.weights[f1][f2] = nx.get_edge_attributes(WG,"weight")
                    print("graph insered")
            except:
                print("out of grid range")

    def getArchiv(self):
        return self.archive

    def getIndi(self,f1,f2):
        try:
            indi = self.archive[f1][f2]  
            return indi.copy()
        except:
            print("no Graph in Slot")
            return None
    


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
        print(self.Grid[row,col])
        return self.Grid[row,col], self.archive[row][col].copy(), self.duration[row,col]
    
    def getBestweights(self,x):
        maxidx = np.argmax(self.Grid)
        maxidx = (-self.Grid.flatten()).argsort()[:10]
        row = int(maxidx[x]/self.size[0])
        col = maxidx[x] % self.size [1]
        print(self.weights[row][col])

    def saveInFile(self,num):
        if num == 0:
            pickle_out = open("src/move_one_leg/scripts/results/fit0.pickle","wb")
            pickle.dump(self.Grid,pickle_out)
            pickle_out.close()

            pickle_out1 = open("src/move_one_leg/scripts/results/archiv0.pickle","wb")
            pickle.dump(self.archive,pickle_out1)
            pickle_out1.close()
            
            pickle_out2 = open("src/move_one_leg/scripts/results/duration0.pickle","wb")
            pickle.dump(self.duration,pickle_out2)
            pickle_out2.close()
           
        if num == 1:
            pickle_out = open("src/move_one_leg/scripts/results/fit1.pickle","wb")    
            pickle.dump(self.Grid,pickle_out)
            pickle_out.close()

            pickle_out1 = open("src/move_one_leg/scripts/results/archiv1.pickle","wb")
            pickle.dump(self.archive,pickle_out1)
            pickle_out1.close()

            pickle_out2 = open("src/move_one_leg/scripts/results/duration1.pickle","wb")
            pickle.dump(self.duration,pickle_out2)
            pickle_out2.close()

    def readFile(self,num):
        if num == 0:
            pickle_in = open("src/move_one_leg/scripts/results/fit0.pickle","rb")
            self.Grid = pickle.load(pickle_in)
            
            pickle_in1 = open("src/move_one_leg/scripts/results/archiv0.pickle","rb")
            self.archive = pickle.load(pickle_in1)      
            
            pickle_in2 = open("src/move_one_leg/scripts/results/duration0.pickle","rb")
            self.duration = pickle.load(pickle_in2)

        if num == 1:
            pickle_in = open("src/move_one_leg/scripts/results/fit1.pickle","rb")
            self.Grid = pickle.load(pickle_in)
            
            pickle_in1 = open("src/move_one_leg/scripts/results/archiv1.pickle","rb")
            self.archive = pickle.load(pickle_in1)   
            
            pickle_in2 = open("src/move_one_leg/scripts/results/duration1.pickle","rb")
            self.duration = pickle.load(pickle_in2)  

    def getRandomIndi(self):
        allIndis = np.array(np.where(self.Grid >0))
        allIndis = allIndis.transpose()
        randomIdx = allIndis[np.random.choice(allIndis.shape[0], 1, replace=False)]
        return self.archive [randomIdx[0][0]][randomIdx[0][1]].copy() , self.duration[randomIdx[0][0]][randomIdx[0][1]]

    def getAllIndi(self):
        allIndis = np.array(np.where(self.Grid >0))
        allIndis = allIndis.transpose()
        return allIndis
        
    def recordFitness(self):
        bestfit = np.max(self.Grid)
        self.bestfitnesses.append(bestfit)
        allFitnesses = np.where(self.Grid >0)
        self.averagefitnesses.append(np.mean(self.Grid[allFitnesses]))

    def recordsameFitness(self,fitness):
        self.samefitnesses.append(fitness)



    def plotFitnes(self):
        x= np.arange(len(self.bestfitnesses))
        fig, ax = plt.subplots()
        ax.plot(x, self.bestfitnesses,label="best")
        ax.plot(x, self.averagefitnesses,label="average")
        ax.legend()
        ax.set(xlabel='time (s)', ylabel='Fitness',
            title='Fitnes')
        ax.grid()

        #fig.savefig("test.png")
        plt.show()
    
    def saveFitness(self):
        file_name = "src/move_one_leg/scripts/storage/Fitness/" + str(datetime.now()) + ".txt"
        file = open (file_name,"w+")
        for k in range(len(self.bestfitnesses)):
            file.write("%d %f %f  \n"%(k,self.bestfitnesses[k], self.averagefitnesses[k]))
        file.close()
        



    def initialzePopulation(self,numIter,NN,atstart,joints,data,duration,rate,SIM):
        for i in range(numIter):
            print("Iteration:",i)
            NN.randomIndi()
            sim = True
            currtime = time.time()#rospy.get_time()
            start_time = time.time()
            while sim:
               
                joints.moveNN(i,NN)
                i += 1
                if currtime + duration < time.time():#rospy.get_time():
                    fit = evaluate1(joints.posrecord,data)
                    fit = [fit,0]
                    #ME.insertIndividuum1(indi[0],indi[1],fit)
                    modularity = NN.getCommunitiesModularity()
                    ccost = getCost(NN.WG)
                    print("mod: ", modularity, " cc: ",ccost, " fit: ",fit[0] )
                    #try:
                    self.archiv(modularity,ccost, NN.WG,fit,False)
                    self.insertIndividuum(modularity,ccost,fit,False)

                    joints.record = []
                    joints.posrecord = []
                    self.recordFitness()
                    sim = False
                    joints.pub1.publish(joints.startPos1)
                    joints.pub3.publish(joints.startPos3)
                    joints.pub5.publish(joints.startPos5)
                    SIM.reset_joints()
                rate.sleep()
        print "My program took", time.time() - start_time, "to run"

        #print(nx.get_node_attributes(NN.WG,"bias"))
        #print(nx.get_edge_attributes(NN.WG,"weight"))
        self.getBestweights(0)
        self.plotFitnes()
        self.saveInFile(0)
        self.plotGrid()
        rospy.signal_shutdown("Records done")


    def run(self,numIter,NN,atstart,joints,data,duration,rate,SIM):
        self.readFile(0)
        for i in range(numIter):
            print("Iteration:",i)
            NN.WG = self.getRandomIndi()

            activeEdges = nx.get_edge_attributes(NN.WG,"active")
            nx.set_edge_attributes(NN.DG, activeEdges, "active")

            WightEdges = nx.get_edge_attributes(NN.WG,"weight")
            nx.set_edge_attributes(NN.DG, WightEdges, "weight")

            biasNodes = nx.get_node_attributes(NN.WG,"bias")
            nx.set_node_attributes(NN.DG, biasNodes, "bias")


            NN.WG,NN.DG = gausMut_bias(NN.WG, NN.DG)
            NN.DG,edge = addEdge(NN.WG,NN.DG)
            if edge != 0:
                NN.addEdgeFromParent(edge)
            NN.WG, NN.DG = deleteEdge(NN.WG, NN.DG) 
            NN.removeEdge()

            NN.WG, NN.DG = gausMut_weigth(NN.WG, NN.DG)   
            
            sim = True
            currtime = rospy.get_time()
            while sim:
                joints.moveNN(i,NN)
                i += 1
                if currtime + duration < rospy.get_time():
                    fit = evaluate1(joints.posrecord,data)
                    fit = [fit,0]
                    modularity = NN.getCommunitiesModularity() 
                    ccost = getCost(NN.WG)
                    print("mod: ", modularity, " cc: ",ccost, "fit", fit)

                    self.archiv(modularity,ccost, NN.WG,fit,False)
                    self.insertIndividuum(modularity,ccost,fit,False)

                    joints.record = []
                    joints.posrecord = []
                    self.recordFitness()
                    sim = False
                    joints.pub1.publish(joints.startPos1)
                    joints.pub3.publish(joints.startPos3)
                    joints.pub5.publish(joints.startPos5)
                    SIM.reset_joints()
                rate.sleep()

                    
            #reset activity of DG to false, to overwrite it later with TRUE after mutatinos
            activeEdges = nx.get_edge_attributes(NN.DG,"active")
            setactivieEdges = [False, False, False, False, False, False, False, False, False,False,False,False]
            activeEdges.update(zip(activeEdges,setactivieEdges))
            nx.set_edge_attributes(NN.DG, activeEdges, "active")
        self.saveInFile(1)
        self.plotFitnes()
        self.plotGrid()
        rospy.signal_shutdown("Records done")



    def analyze(self,NN,atstart,joints,data,rate,SIM):
        self.readFile(1)
        #self.getBestweights(0)
        #print(ME.getAllIndi())
        #print(ME.getArchiv())
        self.plotGrid()
        fit, NN.WG, duration = self.getBest(0)
        modularity = NN.getCommunitiesModularity()
        ccost = getCost(NN.WG)
        print("Fit :",fit, " Mod: ", modularity," ccost: ", ccost)
        print(nx.get_node_attributes(NN.WG,"bias"))
        print(nx.get_edge_attributes(NN.WG,"weight"))
        #NN.printGraph()
        sim = True
        currtime = time.time() #rospy.get_time()
        i=0
        start_time = time.time()
        while sim:            
            joints.moveNN(i,NN)
            #joints.move(i)
            
            #print(np.round((seconds-currtime) % 21,0))
            #if np.round((seconds-currtime) % duration+1,0) == duration:
            if currtime + duration < time.time():#rospy.get_time():
                fit = evaluate1(joints.posrecord,data)
                modularity = NN.getCommunitiesModularity() 
                ccost = getCost(NN.WG)
                print("mod: ", modularity, " cc: ",ccost, " fit: ",fit )
                

                joints.record = []
                joints.posrecord = []
                joints.pub1.publish(joints.startPos1)
                joints.pub3.publish(joints.startPos3)
                joints.pub5.publish(joints.startPos5)
                SIM.reset_joints()

                sim = False
            rate.sleep()
            i=i+1
        print "My program took", time.time() - start_time, "to run"
        rospy.signal_shutdown("Records done") 

    def analyze1(self):
        self.readFile(1)
        indis = self.getAllIndi()
        print(len(indis))
        print(len(self.Grid)**2)
        occupied = float(len(indis))/len(self.Grid)**2
        print("Occupied", occupied)
        bestfit = np.max(self.Grid)
        allfitnesses = np.where(self.Grid>0)
        averagefit = np.mean(self.Grid[allfitnesses])
        print("Fitnesses: best: ",bestfit," average: ", averagefit)
        self.plotGrid()
        rospy.signal_shutdown("Records done")

    def analyze_mut_runs(self,numIter,NN,atstart,joints,data,duration,rate,SIM):
        '''
        same NN, multiple runs to record fitness
        '''
        self.readFile(1)
        self.getBestweights(0)
        #print(ME.getAllIndi())
        #print(ME.getArchiv())
        #self.plotGrid()
        fit, NN.WG = self.getBest(0)
        modularity = NN.getCommunitiesModularity()
        ccost = getCost(NN.WG)
        print("Fit :",fit, " Mod: ", modularity," ccost: ", ccost)
        print(nx.get_node_attributes(NN.WG,"bias"))
        print(nx.get_edge_attributes(NN.WG,"weight"))
        for j in range(numIter):
            #NN.printGraph()
            sim = True
            currtime = time.time() #rospy.get_time()
            i=0
            start_time = time.time()
            while sim:            
                joints.moveNN(i,NN)
                #joints.move(i)
                
                #print(np.round((seconds-currtime) % 21,0))
                #if np.round((seconds-currtime) % duration+1,0) == duration:
                if currtime + duration < time.time():#rospy.get_time():
                    fit = evaluate1(joints.posrecord,data)
                    modularity = NN.getCommunitiesModularity() 
                    ccost = getCost(NN.WG)
                    print("mod: ", modularity, " cc: ",ccost, " fit: ",fit )
                    
                    self.recordsameFitness(fit)
                    joints.record = []
                    joints.posrecord = []
                    joints.pub1.publish(joints.startPos1)
                    joints.pub3.publish(joints.startPos3)
                    joints.pub5.publish(joints.startPos5)
                    SIM.reset_joints()

                    sim = False
                rate.sleep()
                i=i+1
        x = np.arange(len(self.samefitnesses))
        fig, ax = plt.subplots()
        ax.plot(x,self.samefitnesses)
        ax.set(xlabel = "i",ylabel = "fitness", title = "fitness over independent runs, same NN")
        print "my prog needed", time.time() - start_time
        rospy.signal_shutdown("Records done")

    def reCalcGrid(self,repeats, NN,joints,data,duration,rate,SIM):
        self.readFile(1)
        individuals = self.getAllIndi()
        for indi in individuals:
            fits = []
            print(indi)
            NN.WG = self.getIndi(indi[0],indi[1])
            modularity = NN.getCommunitiesModularity()
            ccost = getCost(NN.WG)
            print(" Mod: ", modularity," ccost: ", ccost)
            start_time = time.time()
            for k in range(repeats):
                sim = True
                currtime = time.time() #rospy.get_time()
                i=0
                while sim:            
                    joints.moveNN(i,NN)
                    
                    if currtime + duration < time.time():#rospy.get_time():
                        fit = evaluate1(joints.posrecord,data)
                        fits.append(fit)
            
                        joints.record = []
                        joints.posrecord = []
                        joints.pub1.publish(joints.startPos1)
                        joints.pub3.publish(joints.startPos3)
                        joints.pub5.publish(joints.startPos5)
                        SIM.reset_joints()
                        sim = False
                    rate.sleep()
            print "My program took", time.time() - start_time, "to run"
            print(fits)
            fit = [np.round(np.mean(fits),2),0]
            modularity = NN.getCommunitiesModularity() 
            ccost = getCost(NN.WG)
            print("mod: ", modularity, " cc: ",ccost, " fit: ",fit )
            self.archiv(modularity,ccost, NN.WG,fit,True)
            self.insertIndividuum(modularity,ccost,fit,True)
            #self.archiv(modularity,ccost, NN.WG,fit,False)
            #self.insertIndividuum(modularity,ccost,fit,False)
            print(self.archiveRC)
        self.plotGrid()
        self.archive = self.archiveRC
        self.Grid = self.GridRC 
        self.plotGrid()
        self.plotGridRC()
        self.saveInFile(1)
        rospy.signal_shutdown("Records done")

    def initialzePopulation_with_M(self,numIter,repeats,NN,atstart,joints,data,rate,SIM):
        for i in range(numIter):
            print("Iteration:",i)
            NN.randomIndi()
            fits = []
            start_time = time.time()
            for k in range(repeats):
                #print(k)
                sim = True
                currtime = time.time()#rospy.get_time()
                while sim:
                
                    joints.moveNN(i,NN)
                    i += 1
                    if currtime + self.startduration < time.time():#rospy.get_time():
                        fit = evaluate1(joints.posrecord,data)
                        #print(fit)
                        fits.append(fit)
                        joints.record = []
                        joints.posrecord = []
                        #self.recordFitness()
                        sim = False
                        joints.pub1.publish(joints.startPos1)
                        joints.pub3.publish(joints.startPos3)
                        joints.pub5.publish(joints.startPos5)
                        SIM.reset_joints()
                    rate.sleep()
            print "My program took", time.time() - start_time, "to run"
            print(fits)
            fit = [np.round(np.mean(fits),2),0]
            modularity = NN.getCommunitiesModularity()
            ccost = getCost(NN.WG)
            print("mod: ", modularity, " cc: ",ccost, " fit: ",fit[0] )
            self.archiv(modularity,ccost, NN.WG,fit,False)
            self.insertIndividuum(modularity,ccost,fit,False,True,self.startduration)
            self.recordFitness()

        #print(nx.get_node_attributes(NN.WG,"bias"))
        #print(nx.get_edge_attributes(NN.WG,"weight"))
        #self.getBestweights(0)
        print(self.duration)
        self.plotFitnes()
        self.saveInFile(0)
        self.plotGrid()
        self.saveFitness()
        rospy.signal_shutdown("Records done")
    
    def run_with_mean(self,numIter,repeats,NN,atstart,joints,data,rate,SIM):
        self.readFile(1)
        #print(self.archive)
        #self.plotGrid()
        for i in range(numIter):
            print("Iteration:",i)
            NN.WG, duration = self.getRandomIndi()
            print(duration)

            activeEdges = nx.get_edge_attributes(NN.WG,"active")
            nx.set_edge_attributes(NN.DG, activeEdges, "active")

            WightEdges = nx.get_edge_attributes(NN.WG,"weight")
            nx.set_edge_attributes(NN.DG, WightEdges, "weight")

            biasNodes = nx.get_node_attributes(NN.WG,"bias")
            nx.set_node_attributes(NN.DG, biasNodes, "bias")


            NN.WG,NN.DG = gausMut_bias(NN.WG, NN.DG)
            NN.DG,edge = addEdge(NN.WG,NN.DG)
            if edge != 0:
                NN.addEdgeFromParent(edge)
            NN.WG, NN.DG = deleteEdge(NN.WG, NN.DG) 
            NN.removeEdge()

            NN.WG, NN.DG = gausMut_weigth(NN.WG, NN.DG)   
            fits = []
            for k in range(repeats):
                sim = True
                currtime = rospy.get_time()
                while sim:
                    joints.moveNN(i,NN)
                    i += 1
                    if currtime + duration < rospy.get_time():
                        fit = evaluate1(joints.posrecord,data)
                        fits.append(fit)

                        joints.record = []
                        joints.posrecord = []

                        sim = False
                        joints.pub1.publish(joints.startPos1)
                        joints.pub3.publish(joints.startPos3)
                        joints.pub5.publish(joints.startPos5)
                        SIM.reset_joints()
                    rate.sleep()
            print(fits)
            fit = [np.round(np.mean(fits),2),0]
            modularity = NN.getCommunitiesModularity()
            ccost = getCost(NN.WG)
            print("mod: ", modularity, " cc: ",ccost, " fit: ",fit[0] )
            self.archiv(modularity,ccost, NN.WG,fit,False)
            self.insertIndividuum(modularity,ccost,fit,False,False,duration)
            self.recordFitness()

                    
            #reset activity of DG to false, to overwrite it later with TRUE after mutatinos
            activeEdges = nx.get_edge_attributes(NN.DG,"active")
            setactivieEdges = [False, False, False, False, False, False, False, False, False,False,False,False]
            activeEdges.update(zip(activeEdges,setactivieEdges))
            nx.set_edge_attributes(NN.DG, activeEdges, "active")

            print(self.duration)
        self.saveInFile(1)
        self.plotFitnes()
        self.plotGrid()
        self.saveFitness()
        rospy.signal_shutdown("Records done")
        


        

