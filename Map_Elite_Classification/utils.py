import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def loadData(filePath):
    fileData = np.genfromtxt(filePath, dtype=None)
    data = np.array([fileData["f2"], fileData["f3"]])
    result = np.array([fileData["f1"]])

    return data, result
  
def evaluate2(prediction):
    return (result == prediction.transpose()).sum() / len(data[0]),

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
