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


def greedyDelete(solution, Acapt, Acom, NeighCom, candidates=None):
    '''
    For a given solution, delete 1 vertex if possible (remains feasible)

    When candidates are specified, only consider vertices in candidates for
    deletion
    '''
    solBis = np.copy(solution)
    if candidates == None:
        indexSelected = np.where(solution == 1)[0][1:]
    else:
        indexSelected = np.array(candidates)
        #print(indexSelected)
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
        Try to delete 2 other vertices in its neighborhood
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

def greedyPivotS1(solution, Acapt, Acom, NeighCom):
    '''
    For a given solution, test if this move is possible:
        - select an ampty vertex i
        - insert this vertex + all its neighbors in NeighCom
        - try N=100 stochastic descent using deletion of vertices
    '''
    N = 5
    nNodes = solution.shape[0]
    
    solBis = np.copy(solution)
    indexEmpty = np.where(solution == 0)[0]
    nEmpty = indexEmpty.shape[0]
    assert(nEmpty > 0)
    score = np.sum(solBis)
    print('nEmpty : {}\n'.format(nEmpty))

    np.random.shuffle(indexEmpty)
    ind = 0
    improved = False
    while not(improved) and (ind < nEmpty):
        i = indexEmpty[ind]
        print(i)

        # insert i and its empty neighbors
        v = NeighCom[i][1].copy()
        inserted = [i] + [j for j in v if (solBis[j] == 0)]
        solBis[inserted] = 1
        print('nInserted : {}'.format(len(inserted)))

        # candidates to be deleted
        markedCandidates = np.zeros(nNodes, dtype=np.int)
        markedCandidates[0] = 1
        markedCandidates[i] = 1
        candidates = [i]
        for j in v:
            if markedCandidates[j] == 0:
                markedCandidates[j] = 1
                candidates.append(j)
            v_j = NeighCom[j][1]
            for k in v_j:
                if markedCandidates[k] == 0:
                    markedCandidates[k] = 1
                    candidates.append(k)
        print('nCandidates : {}'.format(len(candidates)))

        # try to improve to improve as much as possible the solution by
        # successive deletions among the pre-selected candidates
        it = 0
        while not(improved) and (it < N):
            print('  > it = {}'.format(it))
            solTer = np.copy(solBis)
            descent = True
            while descent:
                solTer, descent = greedyDelete(
                    solTer, Acapt, Acom, NeighCom, candidates=candidates)
                scoreTer = np.sum(solTer)
            if scoreTer < score:
                improved = True
            else:
                it += 1

        if not(improved):
            ind += 1
            solBis[inserted] = 0

    if improved:
        solBis = solTer

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

    # iterations over neighborhoods
    neighborhoods = [greedyDelete, greedyPivot1, greedyPivotS1]
    descent = True
    ind = 0
    while ind < len(neighborhoods):
        V = neighborhoods[ind]
        solution, descent = V(solution, Acapt, Acom, NeighCom)
        score = np.sum(solution)
        assert(constraints.checkConstraints(solution, Acapt, Acom, NeighCom))
        if descent:
            ind = 0
        else:
            ind += 1
    
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
    '''
    vectScore = []
    t1 = time.time()
    for i in range(100):
        score = VNS(instanceName, Rcapt, Rcom)
        vectScore.append(score)
    t2 = time.time()
    print('score mean : {}'.format(np.mean(vectScore)))
    print('score min : {}\n'.format(np.min(vectScore)))
    print(t2 - t1)
    '''
