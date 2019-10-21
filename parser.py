import numpy as np

import readData as rd


instanceName = 'Instances/captANOR225_9_20.dat'


def readInstance(instanceName):
    instance = []
    for line in open(instanceName):
        row = line.split()
        tmp = [int(row[0])] + [float(x) for x in row[1:]]
        instance.append(tmp)
    return instance

instance = readInstance(instanceName)

def make_Acapt(instance, Rcapt):
    # Cette fonction fait la matrice Acapt telle que Acapt[i,j]=1 ssi
    # dist(i,j)<=Rcapt pour i et j deux cibles
    n = len(instance)
    Acapt = np.zeros((n,n), dtype=np.int)
    for i in range(n):
        for j in range(n):
            if rd.distance2(instance[i], instance[j]) <= Rcapt:
                Acapt[i, j] = 1
    return Acapt

Rcapt = 1
Acapt = make_Acapt(instance, Rcapt)
