'''
On représente une solution par un vecteur de taille n (nombre de sommets):
    - sol[i] = 1 si cible i reçoit un capteur
    - sol[i] = 0 sinon

In this file:
    - Acapt : matrix of adjacency in the capt graph
    - Acom : matrix of adjacency in the com graph
    - NeighCapt : list of neighbors of each vertex in capt graph
    - NeighCom : list of neighbors of each vertex in com graph
'''
import numpy as np
import time

import parserInstance
import constraints


def greedyDelete(solution, Acapt, Acom, NeighCom):
    '''
    For a given solution, delete 1 vertex if possible (remains feasible)
    '''
    solBis = np.copy(solution)
    indexSelected = np.where(solution == 1)[0]
    nSelected = indexSelected.shape[0]
    assert(nSelected > 0)

    np.random.shuffle(indexSelected)
    ind = 0
    feasible = False
    while not(feasible) and (ind < nSelected):
        i = indexSelected[ind]
        solBis[i] = 0
        feasible = constraints.checkConstraints(solBis, Acapt, Acom, NeighCom)
        if not(feasible):
            solBis[i] = 1
            ind += 1
    return solBis, feasible

def VNS(instanceName, Rcapt, Rcom):
    '''
    Implement VNS metaheuristic
    '''
    # parse data
    Acapt, Acom, NeighCapt, NeighCom = parserInstance.parseData(
        instanceName, Rcapt, Rcom)
    nNodes = Acapt.shape[0]

    # parameters
    

    
