#!/usr/bin/env python
# license removed for brevity

## from https://www.youtube.com/watch?v=fV68_tq1ipk
import time
import rospy
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest, SetModelConfiguration, SetModelConfigurationRequest
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from std_srvs.srv import Empty


class Simulation():
    def __init__(self):
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_sim  = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world  = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        rospy.wait_for_service("/gazebo/set_physics_properties")
        self.set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        rospy.wait_for_service("/gazebo/set_model_configuration")
        self.set_joints = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)

        self.init_values()

    def init_values(self):
        '''
        dont works??????????
        rospy.wait_for_service("gazebo/reset_world")
        try :
            self.reset_world()
            print ("reset")
        except:
            print("reset sim failed")
        '''

        #currently default values after calling service
        self.time_step = Float64(0.001)
        self.max_update_rate = Float64(1000.0)
        self.gravity = Vector3()
        self.gravity.x = 0.0
        self.gravity.y = 0.0
        self.gravity.z = -9.8
        self.ode_config = ODEPhysics()
        self.ode_config.auto_disable_bodies = False
        self.ode_config.sor_pgs_precon_iters = 0
        self.ode_config.sor_pgs_iters = 50
        self.ode_config.sor_pgs_w = 1.3
        self.ode_config.sor_pgs_rms_error_tol = 0.0
        self.ode_config.contact_surface_layer = 0.001
        self.ode_config.contact_max_correcting_vel = 100.0
        self.ode_config.cfm = 0.0
        self.ode_config.erp = 0.2
        self.ode_config.max_contacts = 20

        self.update_sim()

    def update_sim(self):
        rospy.wait_for_service("gazebo/pause_physics")
        try :
            self.pause()
            print ("reset")
        except:
            print("pause sim failed")
        set_physics_request = SetPhysicsPropertiesRequest()
        set_physics_request.time_step = self.time_step.data
        set_physics_request.max_update_rate= self.max_update_rate.data
        set_physics_request.gravity = self.gravity
        set_physics_request.ode_config = self.ode_config

        print str(set_physics_request.time_step)
        result = self.set_physics(set_physics_request)

        rospy.wait_for_service("gazebo/unpause_physics")
        try :
            self.unpause()
            print ("reset")
        except:
            print("unpause sim failed")

    def reset_joints(self):
        #print("Hallo")
        '''
        rospy.wait_for_service("gazebo/pause_physics")
        try :
            self.pause()
            print ("reset")
        except:
            print("pause sim failed")
        '''
        set_joints_request = SetModelConfigurationRequest()
        set_joints_request.model_name = "moveoneleg"
        set_joints_request.urdf_param_name = "robot_description"
        set_joints_request.joint_names = ['j_c1_rf', 'j_thigh_rf', 'j_tibia_rf']
        set_joints_request.joint_positions = [0.0, 0.0, 0.0]

        result1 = self.set_joints(set_joints_request)

        rospy.wait_for_service("gazebo/unpause_physics")

        '''
        try :
            self.unpause()
            print ("reset")
        except:
            print("unpause sim failed")
        '''

