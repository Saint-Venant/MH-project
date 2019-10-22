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

def greedyPivot1(solution, Acapt, Acom, NeighCom):
    '''
    For a given solution, test if this move if possible :
        Select an empty vertex
        Try to delete as many other vertices as possible
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
        v = NeighCom[i][1].copy()
        candidates = [j for j in v if (solBis[j] == 1) and (j > 0)]
        nCandidates = len(candidates)

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

    return solBis, improved

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

    # iterations neighborhood 1
    descent = True
    while descent:
        solution, descent = greedyDelete(solution, Acapt, Acom, NeighCom)
        score = np.sum(solution)
        assert(constraints.checkConstraints(solution, Acapt, Acom, NeighCom))
    score1 = score

    # iterations neighborhood 2
    descent = True
    while descent:
        solution, descent = greedyPivot1(solution, Acapt, Acom, NeighCom)
        score = np.sum(solution)
    assert(constraints.checkConstraints(solution, Acapt, Acom, NeighCom))
    score2 = score

    

    return score1, score2


if __name__ == '__main__':
    Rcapt = 1
    Rcom = 2
    instanceName = 'Instances/captANOR225_9_20.dat'

    t1 = time.time()
    score1, score2 = VNS(instanceName, Rcapt, Rcom)
    t2 = time.time()
    print('score1 : {}'.format(score1))
    print('score2 : {}'.format(score2))
    print('\ndt : {}\n'.format(t2-t1))

    vectScore1 = []
    vectScore2 = []
    t1 = time.time()
    for i in range(100):
        score1, score2 = VNS(instanceName, Rcapt, Rcom)
        vectScore1.append(score1)
        vectScore2.append(score2)
    t2 = time.time()
    print('score1 mean : {}'.format(np.mean(vectScore1)))
    print('score1 min : {}\n'.format(np.min(vectScore1)))
    print('score2 mean : {}'.format(np.mean(vectScore2)))
    print('score2 min : {}\n'.format(np.min(vectScore2)))
    print(t2 - t1)
    
