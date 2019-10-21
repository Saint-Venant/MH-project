import numpy as np

import readData as rd


def readInstance(instanceName):
    instance = []
    for line in open(instanceName):
        row = line.split()
        tmp = [int(row[0])] + [float(x) for x in row[1:]]
        instance.append(tmp)
    return instance

def make_Adj(instance, R):
    # Cette fonction fait la matrice d'adjacence telle que Adj[i,j]=1 ssi
    # dist(i,j)<=R pour i et j deux cibles
    n = len(instance)
    Adj = np.zeros((n,n), dtype=np.int)
    for i in range(n):
        for j in range(n):
            if rd.distance2(instance[i], instance[j]) <= R:
                Adj[i, j] = 1
    return Adj

def parseData(instanceName, Rcapt, Rcom):
    instance = readInstance(instanceName)
    Acapt = make_Adj(instance, Rcapt)
    Acom = make_Adj(instance, Rcom)
    return Acapt, Acom

if __name__ == '__main__':
    instanceName = 'Instances/captANOR225_9_20.dat'
    Rcapt = 1
    Rcom = 2
    Acapt, Acom = parseData(instanceName, Rcapt, Rcom)
