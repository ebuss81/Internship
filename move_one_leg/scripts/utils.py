import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import time


def loadData(filePath):
    fileData = np.genfromtxt(filePath, dtype=None)
    data = np.array([fileData["f1"], fileData["f2"], fileData["f3"]])
    return np.transpose(data)
  
def evaluate(motion,data): #TODO Renamed evaluate2->evaluate
    start_time = time.time() 
    motion = np.array(motion)
    motion = np.round(motion,1)
    saveMotion(motion)
    print(len(motion))
    datacopy = data
    fitness = 0
    for pos in motion:
        comparison = (datacopy == pos).all(axis=1)
        if sum(comparison) == 1:
            #print("is in")
            fitness +=1
            deleteRowInData = np.where(comparison == True)
            datacopy = np.delete(datacopy,deleteRowInData[0][0],0)
        else:
            pass
            #print("not in")
    print("Fitness: ",float(fitness)/100)
    print("Time needed:", time.time()-start_time)
    return float(fitness)/100

def evaluate1(motion,data): #TODO Renamed evaluate2->evaluate
    #start_time = time.time() 
    motion = np.array(motion)
    motion = np.round(motion,1)
    saveMotion(motion)
    #print(len(motion))
    datacopy = data
    fitness = 0
    while len(motion) > 0:
        pos=motion[0]
        comparison = (datacopy == pos).all(axis=1)
        comparison1 = (motion == pos).all(axis=1)
        #print(comparison1)
        deleteRowInMotion = np.where(comparison1 == True)
        #print(deleteRowInMotion)
        motion = np.delete(motion,deleteRowInMotion[:][0],0)
        #if sum(comparison1)> 0:
            #fitness -= sum(comparison1)*0.1
            #print(-sum(comparison1)*0.1)

        if sum(comparison) >0:
            #print("is in")
            fitness +=1
            deleteRowInData = np.where(comparison == True)
            datacopy = np.delete(datacopy,deleteRowInData[0][0],0)
        else:
            pass
            #print("not in")
    #print("Time needed:", time.time()-start_time)
    #print("Fitness: ",float(fitness)/100)
    if fitness < 0:
        fitness = 1 #to secure that grid gets filled if no entry
    return float(fitness)/100



def plotNN(nn, best, data):
    # Plot Neural Network output as colored plane
    xvalues = np.arange(data[0].min() - 0.1, data[0].max() + 0.1, 0.005)
    yvalues = np.arange(data[1].min() - 0.1, data[1].max() + 0.1, 0.005)
    icoords, jcoords = np.meshgrid(xvalues, yvalues)
    testdata = np.array([icoords.flatten(), jcoords.flatten()])
    Z = nn.predict(testdata).reshape(icoords.shape)
    plt.pcolormesh(icoords, jcoords, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))


def plotResults(datas, results, predictions):
    for i in range(len(predictions)):
        data = datas[:, i]
        result = results[0][i]
        prediction = predictions[i]

        marker = "o" if result == 0 else "s"
        color = "r" if result == 0 else "b"
        faceColor = color if result != prediction else "w"
        plt.scatter(*data, marker=marker, facecolor=faceColor, edgecolors=color)

    plt.xlabel("x")
    plt.ylabel("y")

def saveMotion(motion):
    f = open("src/move_one_leg/scripts/storage/motion1.txt","w+")
    for k,j in enumerate(motion):
        #print(k,j)
        f.write("%d %f %f %f \n" %(k,j[0],j[1],j[2]))
    f.close()
