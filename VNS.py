'''
On représente une solution par un vecteur de taille n (nombre de sommets):
    - sol[i] = 1 si cible i reçoit un capteur
    - sol[i] = 0 sinon

In this file:
    - Acapt : matrix of adjacency in the capt graph
    - Acom : matrix of adjacency in the com graph
    - NeighCapt : list of neighbors of each vertex in capt graph
    - NeighCom : list of neighbors of each vertex in com graph

Heuristics:
    * speedCapt : only consider candidates for deletion that have at least
                  one nother vertex selected in their Capt neighborhood
                  (heuristic only relevant for Rcapt < Rcom)
    * nearSearch : when searching for an improving solution, tend to favor
                   neighbors of the previous pivot(s)
'''
import numpy as np
import time

import parserInstance

import constraints


def heurNearSearch(candidatesInsert, pivots, NeighCapt, NeighCom, param):
    '''
    candidatesInsert : list of (empty) vertices, for which insertion will be
                       considered
    pivots : list of pivots (vertices inserted in the previous local search)
    param : 1 -> correspond to candidatesInsert for greedyPivot1
            2 -> correspond to candidatesInsert for greedyPivot2

    Return a list of elements in candidatesInsert, reordered in a way that
    vertices in the Com neighborhood of the pivots will be tested first
    '''
    assert(param in [1, 2])
    nNodes = len(NeighCapt)
    first = []
    second = []

    if (param == 1) or (param == 2):
        markedCandidates = np.ones(nNodes, dtype=np.int)
        markedCandidates[candidatesInsert] = 0
        for i in pivots:
            v = NeighCom[i][1]
            for j in v:
                if markedCandidates[j] == 0:
                    markedCandidates[j] = 1
                    first.append(j)
        second = list(np.where(markedCandidates == 0)[0])

    np.random.shuffle(first)
    np.random.shuffle(second)
    orderedCandidates = np.array(first+second)
    return orderedCandidates

def greedyDelete(solution, Acapt, Acom, NeighCapt, NeighCom, \
                 givenCandidates=None, \
                 speedCapt=False, nearSearch=[False, []]):
    '''
    For a given solution, delete 1 vertex if possible (remains feasible)

    When candidates are specified, only consider vertices in candidates for
    deletion
    '''
    solBis = np.copy(solution)
    if givenCandidates == None:
        indexSelected = np.where(solution == 1)[0][1:]
        if speedCapt:
            covers = np.sum(Acapt[np.ix_(indexSelected, indexSelected)], axis=0)
            candidates = indexSelected[covers > 1]
        else:
            candidates = indexSelected
    else:
        candidates = np.array(givenCandidates)
    nSelected = candidates.shape[0]

    np.random.shuffle(candidates)
    ind = 0
    feasible = False
    while not(feasible) and (ind < nSelected):
        i = candidates[ind]
        solBis[i] = 0
        feasible = constraints.checkConstraints(solBis, Acapt, Acom, NeighCom)
        if not(feasible):
            solBis[i] = 1
            ind += 1

    if (nearSearch[0]) and not(feasible):
        pivots = nearSearch[1]
    else:
        pivots = []
        
    return solBis, feasible, pivots

def greedyPivot1(solution, Acapt, Acom, NeighCapt, NeighCom, \
                 speedCapt=False, nearSearch=[False, []]):
    '''
    For a given solution, test if this move is possible :
        - Select an empty vertex
        - Try to delete 2 other vertices in its neighborhood
    '''
    solBis = np.copy(solution)
    indexEmpty = np.where(solution == 0)[0]
    nEmpty = indexEmpty.shape[0]
    assert(nEmpty > 0)

    # order of exploration
    if nearSearch[0]:
        pivots = nearSearch[1]
        assert(len(pivots) > 0)
        indexEmpty = heurNearSearch(indexEmpty, pivots, NeighCapt, NeighCom, 1)
    else:
        np.random.shuffle(indexEmpty)
    
    ind = 0
    improved = False
    while not(improved) and (ind < nEmpty):
        i = indexEmpty[ind]
        solBis[i] = 1

        # candidates to be deleted
        v = NeighCom[i][1]
        if speedCapt:
            indexSelected = np.where(solBis == 1)[0][1:]
            candidates = [j for j in v if (solBis[j] == 1) and (j > 0) and \
                         (np.sum(Acapt[np.ix_(indexSelected, [j])]) > 1)]
        else:
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

    if improved:
        pivots = [i]
    elif nearSearch[0]:
        pivots = nearSearch[1]
    else:
        pivots = []
        
    return solBis, improved, pivots

def greedyPivot2(solution, Acapt, Acom, NeighCapt, NeighCom, \
                 speedCapt=False, nearSearch=[False, []]):
    '''
    For a given solution, test if this move if possible :
        - Select 1 empty vertex + another in its com neighborhood
        - Try to delete 3 vertices in their neighborhood
    '''
    solBis = np.copy(solution)
    indexEmpty = np.where(solution == 0)[0]
    nEmpty = indexEmpty.shape[0]
    assert(nEmpty > 0)
    nNodes = solution.shape[0]

    # order of exploration
    if nearSearch[0]:
        pivots = nearSearch[1]
        assert(len(pivots) > 0)
        indexEmpty = heurNearSearch(indexEmpty, pivots, NeighCapt, NeighCom, 2)
    else:
        np.random.shuffle(indexEmpty)

    # mark explored pairs
    markedPairs = np.zeros((nNodes, nNodes), dtype=np.int)
    markedPairs[np.arange(nNodes), np.arange(nNodes)] = 1

    ind_i1 = 0
    improved = False
    while not(improved) and (ind_i1 < nEmpty):
        i1 = indexEmpty[ind_i1]
        v_i1 = NeighCom[i1][1].copy()
        np.random.shuffle(v_i1)
        assert(len(v_i1) > 0)

        ind_i2 = 0
        while not(improved) and (ind_i2 < len(v_i1)):
            i2 = v_i1[ind_i2]

            if (markedPairs[i1, i2] == 0) and (solBis[i2] == 0):
                v_i2 = NeighCom[i2][1].copy()
                np.random.shuffle(v_i2)
                solBis[i1] = 1
                solBis[i2] = 1

                # candidates to be deleted
                markedCandidates = np.zeros(nNodes, dtype=np.int)
                markedCandidates[0] = 1
                markedCandidates[i1] = 1
                markedCandidates[i2] = 1
                candidates = [i1, i2]
                if speedCapt:
                    indexSelected = np.where(solBis == 1)[0][1:]
                    for j in v_i1+v_i2:
                        if markedCandidates[j] == 0:
                            markedCandidates[j] = 1
                            if (solBis[j] == 1) and \
                               (np.sum(Acapt[np.ix_(indexSelected, [j])]) > 1):
                                candidates.append(j)
                else:
                    for j in v_i1+v_i2:
                        if markedCandidates[j] == 0:
                            markedCandidates[j] = 1
                            if solBis[j] == 1:
                                candidates.append(j)
                nCandidates = len(candidates)

                # find 3 vertices to delete
                np.random.shuffle(candidates)
                ind1 = 0
                ind2 = 1
                ind3 = 2
                while not(improved) and (ind3 < nCandidates):
                    j1 = candidates[ind1]
                    j2 = candidates[ind2]
                    j3 = candidates[ind3]
                    solBis[j1] = 0
                    solBis[j2] = 0
                    solBis[j3] = 0
                    improved = constraints.checkConstraints(
                        solBis, Acapt, Acom, NeighCom)
                    if not(improved):
                        solBis[j1] = 1
                        solBis[j2] = 1
                        solBis[j3] = 1
                        if ind3 < nCandidates - 1:
                            ind3 += 1
                        elif ind2 < nCandidates - 2:
                            ind2 += 1
                            ind3 = ind2 + 1
                        else:
                            ind1 += 1
                            ind2 = ind1 + 1
                            ind3 = ind2 + 1

                if not(improved):
                    solBis[i1] = 0
                    solBis[i2] = 0

                markedPairs[i1, i2] = 1
                markedPairs[i2, i1] = 1

            ind_i2 += 1

        # Try another pivot if the solution cannot be improved
        if not(improved):
            ind_i1 += 1

    if improved:
        pivots = [i1, i2]
    elif nearSearch[0]:
        pivots = nearSearch[1]
    else:
        pivots = []

    return solBis, improved, pivots

def greedyPivotS1(solution, Acapt, Acom, NeighCapt, NeighCom):
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
    score = np.sum(solBis) - 1
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
                scoreTer = np.sum(solTer) - 1
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
            

def VNS(instanceName, Rcapt, Rcom, dtMax=60*10):
    '''
    Implement VNS metaheuristic
    '''
    t1 = time.time()
    
    # parse data
    Acapt, Acom, NeighCapt, NeighCom = parserInstance.parseData(
        instanceName, Rcapt, Rcom)
    nNodes = Acapt.shape[0]

    # parameters

    # heuristics
    speedCapt = (Rcapt < Rcom)

    # initialization
    solution = np.ones(nNodes, dtype=np.int)
    assert(constraints.checkConstraints(solution, Acapt, Acom, NeighCom))
    score = np.sum(solution) - 1

    # iterations over neighborhoods
    neighborhoods = [greedyDelete, greedyPivot1, greedyPivot2]
    descent = True
    ind = 0
    nearSearch = [False, []]
    dt = time.time() - t1
    while (ind < len(neighborhoods)) and (dt < dtMax):
        V = neighborhoods[ind]
        solution, descent, pivots = V(
            solution, Acapt, Acom, NeighCapt, NeighCom, \
            speedCapt=speedCapt, nearSearch=nearSearch)
        scoreNew = np.sum(solution) - 1
        assert(constraints.checkConstraints(solution, Acapt, Acom, NeighCom))
        assert(((scoreNew < score) and descent) or \
               ((scoreNew == score) and not(descent)))
        score = scoreNew
        if descent:
            ind = 0
        else:
            ind += 1
        if len(pivots) == 0:
            nearSearch = [False, []]
        else:
            nearSearch = [True, pivots]
        dt = time.time() - t1
    
    return solution, score


if __name__ == '__main__':
    Rcapt = 1
    Rcom = 2
    instanceName = 'Instances/captANOR400_10_80.dat'

    t1 = time.time()
    solution, score = VNS(instanceName, Rcapt, Rcom)
    t2 = time.time()
    print('score : {}'.format(score))
    print('\ndt : {}\n'.format(t2-t1))
    
    vectScore = []
    t1 = time.time()
    for i in range(3):
        print(i)
        solution, score = VNS(instanceName, Rcapt, Rcom)
        vectScore.append(score)
    t2 = time.time()
    print('score mean : {}'.format(np.mean(vectScore)))
    print('score min : {}\n'.format(np.min(vectScore)))
    print(t2 - t1)
    
