#!/usr/bin/env python
# license removed for brevity

# doees newman realy works???

import sys
sys.path.append('/home/ed/catkin_ws/src/move_one_leg/scripts')

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from control_msgs.msg import JointControllerState
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Clock



import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time


from NeuralNet.Graph import Graph
from MAP_Elite.MAPElite import MapElite
from MAP_Elite.connectioncost import *
import NeuralNet.activations as activations
from utils import*
from parameters import *
from Simulation import Simulation

from joints import Joints



def main():
    global joints, NN, rate, data, ME
    rospy.init_node('Leg_talker', anonymous=True)
    rate = rospy.Rate(100) # 10hz


    dataPaths = ["src/move_one_leg/scripts/data.txt"]
    data  = loadData(dataPaths[0])


    SIM = Simulation()
    joints = Joints()

    #build NeuralNet
    NN = Graph()
    NN.initializeGraph()
    
    #build MAP_Elite
    size = np.array([10,10])
    #size = np.array([50,50])

    ME = MapElite(size)
    duration =25

    rospy.on_shutdown(joints.stop)
    sim = True
    while not rospy.is_shutdown():

        SIM.reset_joints()
        atstart = True
        #ME.initialzePopulation(1,NN,atstart,joints,data,duration,rate,SIM)
        #ME.run(5,NN,atstart,joints,data,duration,rate,SIM)
        #ME.analyze(NN,atstart,joints,data,rate,SIM)
        #ME.analyze_mut_runs(10,NN,atstart,joints,data,rate,SIM)
        #ME.reCalcGrid(3,NN,joints,data,duration,rate,SIM)
        #ME.initialzePopulation_with_M(500,3,NN,atstart,joints,data,rate,SIM)
        #ME.run_with_mean(200,3,NN,atstart,joints,data,rate,SIM)
        ME.analyze1()
        rate.sleep()


    
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
