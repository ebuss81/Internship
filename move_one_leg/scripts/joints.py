import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
import math
import time
import matplotlib.pyplot as plt
class Joints:
    def __init__(self):
        self.name = []
        self.position = [1,1,1] ## just default
        self.velocity = []
        self.maxAngle1 = [-1.5,1.5]
        self.maxAngle3 = [-1.0,0.5]
        self.maxAngle5 = [-1,1]
        self.record = []
        self.posrecord = []
        self.posrecordCommand =[]
        self.should =[]

        self.pub1 = rospy.Publisher('/moveoneleg/j_c1_rf_position_controller/command', Float64, queue_size=10)
        self.pub3 = rospy.Publisher('/moveoneleg/j_thigh_rf_position_controller/command', Float64, queue_size=10)
        self.pub5 = rospy.Publisher('/moveoneleg/j_tibia_rf_position_controller/command', Float64, queue_size=10)

        '''
        self.ss1 = rospy.ServiceProxy('/servo1/set_speed', SetSpeed)
        self.ss3 = rospy.ServiceProxy('/servo3/set_speed', SetSpeed)
        self.ss5 = rospy.ServiceProxy('/servo5/set_speed', SetSpeed)
        '''
        self.sub_joints = rospy.Subscriber('/moveoneleg/joint_states', JointState, self.currentValues)
        
        self.startPos1 = 0.0
        self.startPos3 = 0.0
        self.startPos5 = 0.0

        self.pub1_stored = 0.0
        self.pub3_stored = 0.0
        self.pub5_stored = 0.0

        self.timestemp = 0
        self.reset = False

        
    def startposition (self):
        #currPosition= np.asarray(self.position)
        #currPosition = np.round(currPosition,3)
        #print(currPosition)
        #self.pub1.publish(self.startPos1)
        #self.pub3.publish(self.startPos3)
        #self.pub5.publish(self.startPos5)
        #print(self.position)
        if all(np.abs(i) <0.1 for i in self.position):
            return True
        else:
            return False
    
    '''
    def setVelocitiy(self,speed):
        rospy.wait_for_service('/servo1/set_speed')
        rospy.wait_for_service('/servo3/set_speed')
        rospy.wait_for_service('/servo5/set_speed')

        resp1 = self.ss1(speed)
        resp3 = self.ss3(speed)
        resp5 = self.ss5(speed)
    '''

    def currentValues(self,data):
        #print(data.position)
        self.name = data.name
        self.position = data.position
        self.velocity = data.velocity
        #print(self.position)
    '''
    def recording (self,i):
        #print(np.round(rospy.get_time()%1,2))
        #print("TIme",self.timestemp)
        #print(self.position)
        if self.timestemp < np.round(rospy.get_time()%1,2):
            self.posrecord.append([self.position[0],self.position[1],self.position[2]])
            self.timestemp = np.round(rospy.get_time()%1,2)
            if np.round(rospy.get_time()%1,2) > 0.8:
                self.reset = True

        if np.round(rospy.get_time()%1,1) == 0 and self.reset:
                self.timestemp=0
                self.reset = False
        #if i % 2 == 1:
        #self.record.append([2*math.sin(i*0.1),self.position])
        #self.posrecord.append([self.position[0],self.position[1],self.position[2]])
            
            #print(self.record)
    '''
    def recording (self,i,pos):
        #print(np.round(rospy.get_time()%1,2))
        #print(np.round(time.time()%1,2))
        #print(len(self.posrecord))
        #print("TIme",self.timestemp)
        #print(self.position)
        if self.timestemp < np.round(time.time()%1,2):
            self.posrecord.append([self.position[0],self.position[1],self.position[2]])
            self.posrecordCommand.append([self.position[0],self.position[1],self.position[2]])
            self.should.append(pos)
            self.timestemp = np.round(time.time()%1,2)
            if np.round(time.time()%1,2) > 0.8:
                self.reset = True

        if np.round(time.time()%1,1) == 0 and self.reset:
                self.timestemp=0
                self.reset = False
        #if i % 2 == 1:
        #self.record.append([2*math.sin(i*0.1),self.position])
        #self.posrecord.append([self.position[0],self.position[1],self.position[2]])
            
            #print(self.record)


    def saveRecords(self):
        f = open("src/move_one_leg/scripts/storage/test.txt","w+")
        for k,j in enumerate(self.record):
            f.write("%d %f %f %f %f \n"%(k,j[0],j[1][0],j[1][1],j[1][2]))
        f.close()

    def gotostart(self):
        self.pub1.publish(self.startPos1)
        self.pub3.publish(self.startPos3)
        self.pub5.publish(self.startPos5)

    def move(self,i):
        #if not self.position:
        #    pass
        #else:
        position1 = 0 #2*math.sin(i*0.01)
        position3 = 0#2*math.sin(i*0.1)
        position5 = math.sin(i*0.01)
        position5 = np.round(position5,2)
        print("i: ",i, " goto ",position5,"is: ", self.position[2])
        #Servo1
        if position1 > self.maxAngle1[0] and position1 < self.maxAngle1[1]:
            self.pub1.publish(position1)   # publish it  
            self.pub1_stored = position1
        else:
            self.pub1.publish(self.pub1_stored)
        
        #Servo2
        if position3 > self.maxAngle3[0] and position3 < self.maxAngle3[1]:
            self.pub3.publish(position3)
            self.pub3_stored = position3
        else:
            self.pub3.publish(self.pub3_stored)
    
        #Servo3
        if position5  > self.maxAngle5[0] and position5  < self.maxAngle5[1]:
            self.pub5.publish(position5)
            self.pub5_stored = position5
        else:
                self.pub5.publish(self.pub5_stored) 
            #print("Soll : ", position5," Ist : ",  self.position[0])
        self.recording(i)

    def moveNN(self,i,NN):
        currPosition= np.asarray(self.position)
        #print(currPosition)
        positions = NN.predict(currPosition)
        positions = np.multiply(positions,[1.5,1,0.9])
        self.pub1.publish(positions[0])
        self.pub3.publish(positions[1])
        self.pub5.publish(positions[2])
        self.recording(i,positions)

    def moveNN_1(self,here,NN,i):
        currPosition= here
        #print(currPosition)
        positions = NN.predict(currPosition)
        positions = np.multiply(positions,[1.5,1,0.9])
        #self.pub1.publish(positions[0])
        #self.pub3.publish(positions[1])
        #self.pub5.publish(positions[2])
        self.recording(i,positions)



    def stop(self):
        self.pub1.publish(self.startPos1)
        self.pub3.publish(self.startPos3)
        self.pub5.publish(self.startPos5)
        self.saveRecords()
        '''
        x= np.arange(len(self.posrecordCommand))
        fig, ax = plt.subplots()
        ax.plot(x, [item[0] for item in self.posrecordCommand],label= "coxa")
        ax.plot(x, [item[1] for item in self.posrecordCommand],label= "femur")
        ax.plot(x, [item[2] for item in self.posrecordCommand],label= "tibia")

        ax.plot(x, [item[0] for item in self.should],label= "coxa")
        ax.plot(x, [item[1] for item in self.should],label= "femur")
        ax.plot(x, [item[2] for item in self.should],label= "tibia")

        #ax.plot(x,self.should)
        #ax.plot(x, np.round(math.sin(x*0.1),2),label="average")
        ax.legend()
        ax.set(xlabel='time (s)', ylabel='Fitness',
            title='Fitnes')
        ax.grid()

        #fig.savefig("test.png")
        plt.show()
        '''

