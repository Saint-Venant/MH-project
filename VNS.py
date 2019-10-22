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
    indexSelected = np.where(solution == 1)[0][1:]
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

def greedyInsert1(solution, score, Acapt, Acom, NeighCom):
    '''
    For a given solution, test if this move if possible :
        Select an empty vertex
        Try to delete as many other vertices as possible

    score : current score of the solution
    '''
    solBis = np.copy(solution)
    indexEmpty = np.where(solution == 0)[0]
    nEmpty = indexEmpty.shape[0]
    assert(nEmpty > 0)

    np.random.shuffle(indexEmpty)
    ind = 0
    improved = False
    while not(improved) and (ind < nEmpty):
        i = indexEmpty[ind]
        solBis[i] = 1

        # candidates to be deleted
        candidates = [i] + NeighCom[i][1]
        nCandidates = len(candidates)
        assert(nCandidates >= 2)

        # find a pair of vertices to delete
        np.random.shuffle(candidates)
        ind1 = 0
        ind2 = 1
        feasible = False
        while not(feasible) and (ind2 < nCandidates):
            j1 = candidates[ind1]
            j2 = candidates[ind2]
            solBis[j1] = 0
            solBis[j2] = 0
            feasible = constraints.checkConstraints(
                solBis, Acapt, Acom, NeighCom)
            if not(feasible):
                solBis[j1] = 1
                solBis[j2] = 1
                if ind2 < nCandidates - 1:
                    ind2 += 1
                else:
                    ind1 += 1
                    ind2 = ind1 + 1

        # Try another pivot if the solution cannot be improved
        if not(feasible):
            ind += 1
            solBis[i] = 0
        else:
            improved = True

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

    # initialization
    solution = np.ones(nNodes, dtype=np.int)
    assert(constraints.checkConstraints(solution, Acapt, Acom, NeighCom))
    score = np.sum(solution)

    # iterations
    descent = True
    while descent:
        solution, descent = greedyDelete(solution, Acapt, Acom, NeighCom)
        score = np.sum(solution)

    return score


if __name__ == '__main__':
    Rcapt = 1
    Rcom = 2
    instanceName = 'Instances/captANOR225_9_20.dat'

    t1 = time.time()
    score = VNS(instanceName, Rcapt, Rcom)
    t2 = time.time()
    print('score : {}'.format(score))
    print('\ndt : {}\n'.format(t2-t1))

    vectScore = []
    for i in range(100):
        score = VNS(instanceName, Rcapt, Rcom)
        vectScore.append(score)
    print('score mean : {}'.format(np.mean(vectScore)))
    print('score min : {}'.format(np.min(vectScore)))
    
