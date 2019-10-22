import numpy as np


def readInstance(instanceName):
    instance = []
    for line in open(instanceName):
        row = line.split()
        tmp = [int(row[0])] + [float(x) for x in row[1:]]
        instance.append(tmp)
    return instance

def distance2(a, b):
    dist = np.sqrt((a[1] - b[1])**2 + (a[2] - b[2])**2)
    return dist

def make_Adj(instance, R):
    # Cette fonction fait la matrice d'adjacence telle que Adj[i,j]=1 ssi
    # dist(i,j)<=R pour i et j deux cibles
    n = len(instance)
    Adj = np.zeros((n,n), dtype=np.int)
    for i in range(n):
        for j in range(n):
            if distance2(instance[i], instance[j]) <= R:
                Adj[i, j] = 1
    return Adj

def make_Neigh(Adj):
    '''
    Create the list of all the neighbors for each vertex
    '''
    n = Adj.shape[0]
    Neigh = []
    for i in range(n):
        v = np.where(Adj[i, :] == 1)[0]
        v = list(v[v != i])
        Neigh.append([len(v), v])
    return Neigh

def parseData(instanceName, Rcapt, Rcom):
    instance = readInstance(instanceName)
    Acapt = make_Adj(instance, Rcapt)
    Acom = make_Adj(instance, Rcom)
    NeighCapt = make_Neigh(Acapt)
    NeighCom = make_Neigh(Acom)
    return Acapt, Acom, NeighCapt, NeighCom

if __name__ == '__main__':
    instanceName = 'Instances/captANOR225_9_20.dat'
    Rcapt = 1
    Rcom = 2
    Acapt, Acom, NeighCapt, NeighCom = parseData(instanceName, Rcapt, Rcom)
