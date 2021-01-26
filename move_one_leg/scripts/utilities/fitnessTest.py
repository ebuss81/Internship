import numpy as np



fileData = np.genfromtxt("/home/ed/catkin_ws/src/move_one_leg/scripts/data.txt", dtype=None)
data = np.array([fileData["f1"], fileData["f2"], fileData["f3"]])
data =  np.transpose(data)
#data = np.array([[0,0,0],[0.1,0,0.1],[0,0,0.1]])
motion = np.array([[0,0,0],[0.1,0,0.1],[0,0,0]])

fitness = 0
for pos in motion:
    comparison = (data == pos).all(axis=1)
    if sum(comparison) == 1:
        fitness +=1
        deleteRow = np.where(comparison == True)
        data = np.delete(data,deleteRow[0][0],0)
    else:
        pass
print("Fitness: ",fitness)


