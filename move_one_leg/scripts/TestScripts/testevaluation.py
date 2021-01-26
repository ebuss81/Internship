
import sys
sys.path.append('/home/ed/catkin_ws/src/move_one_leg/scripts')


from utils import*


def evaluate1(motion,data): #TODO Renamed evaluate2->evaluate
    motion = np.array(motion)
    motion = np.round(motion,1)
    saveMotion(motion)
    print(len(motion))
    datacopy = data
    fitness = 0
    while len(motion) > 0:
        print(motion[0])
        print(len(motion))
        pos=motion[0]
        comparison = (datacopy == pos).all(axis=1)
        comparison1 = (motion == pos).all(axis=1)
        #print(comparison1)
        deleteRowInMotion = np.where(comparison1 == True)
        #print(deleteRowInMotion)
        motion = np.delete(motion,deleteRowInMotion[:][0],0)
        if sum(comparison1)> 0:
            fitness -= sum(comparison1)*0.3
        if sum(comparison) >0:
            #print("is in")
            fitness +=1
            deleteRowInData = np.where(comparison == True)
            datacopy = np.delete(datacopy,deleteRowInData[0][0],0)
        else:
            pass

            #print("not in")
    print("Fitness: ",float(fitness)/100)
    return float(fitness)/100



def main():
    dataPaths = ["/home/ed/catkin_ws/src/move_one_leg/scripts/data.txt","/home/ed/catkin_ws/src/move_one_leg/scripts/TestScripts/motion.txt"]
    titles = ["Neural Network Classification"]
    data  = loadData(dataPaths[0])
    motion = loadData(dataPaths[1])
    #print(motion)

    print(evaluate1(motion,data))



if __name__ == "__main__":
        main()

