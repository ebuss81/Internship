import numpy as np
import matplotlib.pyplot as plt

coxa = np.arange(-1.5,1.6,0.1)
femur = np.arange(-1,1.1,0.1)
tibia = np.arange(-0.9,1,0.1)
coxa = np.round(coxa,1)
femur = np.round(femur,1)
tibia = np.round(tibia,1)
space = np.zeros([len(coxa)*len(femur)*len(tibia),3])


i=0
for cox in coxa:
    for fem in femur:
        for tib in tibia:

            space[i,:] = [cox,fem,tib]
            i +=1


f = open("/home/ed/catkin_ws/src/move_one_leg/scripts/data.txt","w+")
for k,j in enumerate(space):
        print(k,j)
        f.write("%d %f %f %f \n" %(k,j[0],j[1],j[2]))
f.close()


'''
lencoxa = 10#6.1
lenfemur= 5#6.0
lentibia = 5 # 6.6
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.scatter(np.sin(coxa)*lencoxa, 0, np.cos(coxa)*lencoxa, cmap='viridis', linewidth=0.5);
#for coxX,coxZ in zip(np.sin(coxa)*lencoxa,np.cos(coxa)*lencoxa):
#    ax.scatter(coxX, np.sin(femur)*lenfemur, np.cos(femur)*lenfemur+coxZ, color='red', linewidth=0.5);
#    for femy,femz in zip(np.sin(femur)*lenfemur,np.cos(femur)*lenfemur+coxZ):
#        ax.scatter(coxX, np.sin(tibia)*lentibia + femy, np.cos(tibia)*lentibia+femz, color='green', linewidth=0.1);

ax.scatter(lencoxa * np.cos(space[:,0]),0 ,lencoxa * np.sin(space[:,0]), cmap='viridis', linewidth=0.5);

x = lencoxa * np.cos(space[:,0] ) #+ lenfemur * np.cos(space[:,0] )
y = 0                            #+ lenfemur * np.cos(space[:,0] + space[:,1])
z = lencoxa * np.sin(space[:,0]) #+ lenfemur * np.sin(space[:,0] + space[:,1])
#x =  0
#y =  lenfemur * np.cos(0 + space[:,1]+ np.pi/2)
#z =   lenfemur * np.sin(space[:,1]+ np.pi/2)
ax.scatter(x,y ,z, cmap='viridis', linewidth=0.5);

ax.set(xlabel='x', ylabel='y', zlabel='z') 
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
ax.set_zlim([0,20])

plt.show()
'''