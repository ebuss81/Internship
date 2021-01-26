
import numpy as np


import matplotlib.pyplot as plt


fileData = np.genfromtxt("/home/ed/catkin_ws/src/one_leg/scripts/storage/test.txt", dtype=None)
data = np.array([fileData["f0"], fileData["f1"], fileData["f2"], fileData["f3"], fileData["f4"]])
plt.plot(data[0,:],data[1,:],'--', color='grey', label = "Trajectory")
plt.plot(data[0,:],data[2,:], label = "Coxa")
plt.plot(data[0,:],data[3,:], label = "Femur")
plt.plot(data[0,:],data[4,:], label = "Tibia")
plt.xlabel('time')
plt.ylabel('joint angle [rad]')
plt.legend()
plt.title("vel :1, 200 timestep, every second recorded")
plt.show()