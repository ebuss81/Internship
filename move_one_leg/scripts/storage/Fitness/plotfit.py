
import numpy as np


import matplotlib.pyplot as plt

fileData7 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-14 16:45:18.189379.txt", dtype=None)
fileData6 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-15 12:44:06.396126.txt", dtype=None)

fileData5 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-15 15:18:22.846526.txt", dtype=None)
fileData4 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-16 10:50:04.953556.txt", dtype=None)
fileData3 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-16 18:41:38.985816.txt", dtype=None)
fileData0 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-17 10:35:34.334368.txt", dtype=None)
fileData1 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-18 09:00:17.064431.txt", dtype=None)
fileData2 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-19 08:55:11.599055.txt", dtype=None)
fileData8 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-20 11:12:56.489374.txt", dtype=None)
fileData9 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-21 09:07:11.035151.txt", dtype=None)
fileData10 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-22 10:56:09.793626.txt", dtype=None)
fileData11 = np.genfromtxt("src/move_one_leg/scripts/storage/Fitness/2021-01-23 16:32:44.697393.txt", dtype=None)
data0 = np.array([fileData0["f0"], fileData0["f1"], fileData0["f2"]])
data1 = np.array([fileData1["f0"], fileData1["f1"], fileData1["f2"]])
data2 = np.array([fileData2["f0"], fileData2["f1"], fileData2["f2"]])
data3 = np.array([fileData3["f0"], fileData3["f1"], fileData3["f2"]])
data4 = np.array([fileData4["f0"], fileData4["f1"], fileData4["f2"]])
data5 = np.array([fileData5["f0"], fileData5["f1"], fileData5["f2"]])
data6 = np.array([fileData6["f0"], fileData6["f1"], fileData6["f2"]])
data7 = np.array([fileData7["f0"], fileData7["f1"], fileData7["f2"]])
data8 = np.array([fileData8["f0"], fileData8["f1"], fileData8["f2"]])
data9 = np.array([fileData9["f0"], fileData9["f1"], fileData9["f2"]])
data10 = np.array([fileData10["f0"], fileData10["f1"], fileData10["f2"]])
data11 = np.array([fileData11["f0"], fileData11["f1"], fileData11["f2"]])
#data = data0.append(data1)
#data = data.append(data2)

data = np.concatenate((data7,data6,data5,data4,data3,data0,data1,data2,data8,data9,data10,data11),axis=1)
print(data.shape)
print(len(data[0]))
x = np.arange(len(data[0]))

plt.plot(x,data[1,:], color='black', label = "best")
plt.plot(x,data[2,:],'--', color='grey', label = "average")

plt.xlabel('Number of iteration')
plt.ylabel('Fitness')
plt.legend()
#plt.title("vel :1, 200 timestep, every second recorded")
plt.show()

