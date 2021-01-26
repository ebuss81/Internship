import numpy as np

import pickle
import matplotlib.pyplot as plt
        
pickle_in2 = open("src/move_one_leg/scripts/results/duration1.pickle","rb")
duration = pickle.load(pickle_in2)  
gridsize = np.array([10,10])
size = gridsize

Grid = np.zeros(size)

min_val, max_val = 0, 1

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(duration,cmap=plt.cm.Blues,extent=[min_val, max_val, max_val, min_val], origin="upper")

test = fig.colorbar(cax)
cax.set_clim(-0, 60)
test = test.set_label("time [s]")

for i in range(size[0]):
    for j in range(size[1]):
        c = duration.transpose()[i][j]
        ax.text((i+0.5)/size[0], (j+0.5)/size[1], str(c), va='center', ha='center')


ax.set_xlim(min_val, max_val)
ax.set_ylim(max_val, min_val)
ax.set_xticks(np.arange(10.)/10)
ax.set_yticks(np.arange(10.)/10)
plt.xlabel("ConnectionCost")
plt.ylabel("Modularity")

plt.show()