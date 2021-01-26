#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import Float64
import math
from sensor_msgs.msg import JointState
from control_msgs.msg import JointControllerState


from joints import Joints
"""
find topics with rostopic list (--> copy in pub)
find datatype with rostopic info ${topic name} (--> insert in pub and import from std_msgs or somethink else)
"""
def talker1():
    i = 0
    atstart = False
    rospy.init_node('Leg_talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    
    joints = Joints()
    rospy.on_shutdown(joints.stop)

    rate = rospy.Rate(10) # 10hz

    #joints.setVelocitiy(1)
    
    while not rospy.is_shutdown():
        
        if atstart == False:
            print("go to start")
            atstart = joints.startposition()
        
        if atstart == True:
            print("moving")
            joints.move(i)
        #print(joints.position[0])
        #sjoints.recording(i)
        if len(joints.record) >= 100:
            print(joints.record)
            rospy.signal_shutdown("Records done") 
             
        rate.sleep()
        i += 1


if __name__ == '__main__':
    try:
        talker1()

    except rospy.ROSInterruptException:
        pass


"""
OTHER HINDS

go in folder with .py (terminal

ls: lists things inside
ls -l: list things and look if executable

"""