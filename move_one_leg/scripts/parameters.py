import numpy as np

## used in graph.py and main.py

NPL = np.array([3,4,3])     #   NeuronsperLayer, to iterate in prediction function
NBL =  7                    #   NumBiasedLayers, i think first biased neuron, check it!
SNPL = np.array([1,5,10])   #   StartNeuronsPerLayer, only to get activiationsfunction of layers
MAX_R = 15
WR = np.arange(-MAX_R,MAX_R+0.1,2) #  weights range
BR = np.arange(-MAX_R,MAX_R+0.1,2) #  bias range
#WR = np.arange(-1,1.1,0.1) #  weights range
#BR = np.arange(-1,1.1,0.1) #  bias range
WR = np.round(WR,1)
BR = np.round(BR,1)
# next parameter for parent graph!
NUMLAYER = 3
NUMNEURONS = 4

# to normalize Connection Cost
# maxcc (connection cost) was tested with a fully connected layer with max weights in Mod_Cost_test.py
MAXCC = 60#33.2