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
    * nearSearch : 
'''
if __name__ == '__main__':
    import sys
    sys.path.append('..\\')

import numpy as np
import time

from meta_VNS import parserInstance
from meta_VNS import constraints
from meta_VNS import heuristics



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
    if (nearSearch[0]) and (len(nearSearch[1]) > 0):
        pivots = nearSearch[1]
        indexEmpty = heuristics.heurNearSearch(
            indexEmpty,
            pivots,
            NeighCapt,
            NeighCom,
            1
        )
    else:
        np.random.shuffle(indexEmpty)
    
    ind = 0
    improved = False
    while not(improved) and (ind < nEmpty):
        i = indexEmpty[ind]
        solBis[i] = 1

        # candidates to be deleted
        v = np.array(NeighCom[i][1])
        if speedCapt:
            candidates = heuristics.heurSpeedCapt(solBis, v, Acapt)
        else:
            cand1 = v[solBis[v] == 1]
            candidates = cand1[cand1 > 0]
        nCandidates = candidates.shape[0]

        # find a pair of vertices to delete
        np.random.shuffle(candidates)
        ind1 = 0
        ind2 = 1
        feasible = False
        while not(feasible) and (ind2 < nCandidates):
            j1 = candidates[ind1]
            j2 = candidates[ind2]
            assert(solBis[j1] == 1)
            assert(solBis[j2] == 1)
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
    if (nearSearch[0]) and (len(nearSearch[1]) > 0):
        pivots = nearSearch[1]
        indexEmpty = heuristics.heurNearSearch(
            indexEmpty,
            pivots,
            NeighCapt,
            NeighCom,
            2
        )
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
                neighborVertices = []
                for j in v_i1+v_i2:
                    if markedCandidates[j] == 0:
                        markedCandidates[j] = 1
                        neighborVertices.append(j)
                neighborVertices = np.array(neighborVertices)
                if speedCapt:
                    candidates = heuristics.heurSpeedCapt(
                        solBis, neighborVertices, Acapt)
                else:
                    candidates = neighborVertices[solBis[neighborVertices] == 1]
                nCandidates = candidates.shape[0]

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

def V(solution, Acapt, Acom, NeighCapt, NeighCom, speedCapt, nearSearch, \
      t_max, neighFunctions, indStart):
    '''
    Compute local search using the given list of neighborhoods
    neighFunctions

    t_max : maximum time at which the function should stop
    indStart : index in the neighFunctions list at which to start
    '''
    score = np.sum(solution) - 1
    descent = True
    nNeighFunctions = len(neighFunctions)
    ind = indStart
    while (ind < nNeighFunctions) and (time.time() < t_max):
        neighFunc = neighFunctions[ind]
        solution, descent, pivots = neighFunc(
            solution,
            Acapt,
            Acom,
            NeighCapt,
            NeighCom,
            speedCapt=speedCapt,
            nearSearch=nearSearch
        )
        assert(constraints.checkConstraints(solution, Acapt, Acom, NeighCom))
        scoreBis = np.sum(solution) - 1
        assert(
            ((scoreBis < score) and descent) or \
            ((scoreBis == score) and not(descent))
        )
        score = scoreBis
        if descent:
            ind = 0
        else:
            ind += 1
        if nearSearch[0] and (len(pivots) > 0):
            nearSearch[1] = pivots
            
    return solution, score

def V0(solutionInitial, Acapt, Acom, NeighCapt, NeighCom, speedCapt, nearSearch, \
       t_max):
    '''
    Compute local search using neighborhoods:
    - greedyDelete

    t_max : maximum time at which the function should stop
    '''
    neighFunctions = [greedyDelete]
    indStart = 0
    
    arraySolutions = []
    arrayScores = []
    while time.time() < t_max:
        nearSearch0 = [False, []]
        solution, score = V(
            solutionInitial,
            Acapt,
            Acom,
            NeighCapt,
            NeighCom,
            speedCapt,
            nearSearch0,
            time.time() + 60*10,
            neighFunctions,
            indStart
        )
        arraySolutions.append(solution)
        arrayScores.append(score)
        print('  score : {}'.format(score))

    # sort solutions by increasing score
    arraySolutions = np.array(arraySolutions)
    arrayScores = np.array(arrayScores)
    indexSort = np.argsort(arrayScores)
        
    return arraySolutions[indexSort], arrayScores[indexSort]

def V1(arraySolutions, arrayScores, Acapt, Acom, NeighCapt, NeighCom, \
       speedCapt, nearSearch, t_max):
    '''
    Compute local search using neighborhoods:
    - greedyDelete
    - greedyPivot1

    t_max : maximum time at which the function should stop
    '''
    neighFunctions = [greedyDelete, greedyPivot1]
    indStart = 1

    nbSolutions = arraySolutions.shape[0]
    i_solution = 0
    while (i_solution < nbSolutions) and (time.time() < t_max):
        nearSearch1 = [nearSearch[0], []]
        solution, score = V(
            arraySolutions[i_solution],
            Acapt,
            Acom,
            NeighCapt,
            NeighCom,
            speedCapt,
            nearSearch1,
            t_max,
            neighFunctions,
            indStart
        )
        print('  > {} ; score : {} (previous score = {})'.format(
            i_solution,
            score,
            arrayScores[i_solution]))
        arraySolutions[i_solution] = solution
        arrayScores[i_solution] = score
        i_solution += 1

    # sort solutions by increasing score
    arraySolutions = np.array(arraySolutions)
    arrayScores = np.array(arrayScores)
    indexSort = np.argsort(arrayScores)
        
    return arraySolutions[indexSort], arrayScores[indexSort]

def V2(arraySolutions, arrayScores, Acapt, Acom, NeighCapt, NeighCom, \
       speedCapt, nearSearch, t_max):
    '''
    Compute local search using neighborhoods:
    - greedyDelete
    - greedyPivot1
    - greedyPivot2

    t_max : maximum time at which the function should stop
    '''
    neighFunctions = [greedyDelete, greedyPivot1, greedyPivot2]
    indStart = 2

    nbSolutions = arraySolutions.shape[0]
    i_solution = 0
    while (i_solution < nbSolutions) and (time.time() < t_max):
        nearSearch2 = [nearSearch[0], []]
        solution, score = V(
            arraySolutions[i_solution],
            Acapt,
            Acom,
            NeighCapt,
            NeighCom,
            speedCapt,
            nearSearch2,
            t_max,
            neighFunctions,
            indStart
        )
        print('  > {} ; score : {} (previous score = {})'.format(
            i_solution,
            score,
            arrayScores[i_solution]))
        arraySolutions[i_solution] = solution
        arrayScores[i_solution] = score
        i_solution += 1

    # sort solutions by increasing score
    arraySolutions = np.array(arraySolutions)
    arrayScores = np.array(arrayScores)
    indexSort = np.argsort(arrayScores)
        
    return arraySolutions[indexSort], arrayScores[indexSort]
            

def VNS(instanceName, Rcapt, Rcom, dtMax=60*4):
    '''
    Implement VNS metaheuristic
    '''
    t1 = time.time()
    
    # parse data
    Acapt, Acom, NeighCapt, NeighCom = parserInstance.parseData(
        instanceName, Rcapt, Rcom)
    nNodes = Acapt.shape[0]

    # parameters
    t_max0 = t1 + dtMax/20
    t_max1 = t1 + 4*dtMax/20
    t_max2 = t1 + dtMax

    # heuristics
    speedCapt = (Rcapt < Rcom)
    nearSearch = [True, []]

    # initialization
    solutionInitial = np.ones(nNodes, dtype=np.int)
    assert(constraints.checkConstraints(solutionInitial, Acapt, Acom, NeighCom))
    score = np.sum(solutionInitial) - 1

    # iterations over neighborhoods
    arraySolutions, arrayScores = V0(
        solutionInitial, Acapt, Acom, NeighCapt, NeighCom,
        speedCapt, nearSearch, t_max0)
    print()
    arraySolutions, arrayScores = V1(
        arraySolutions, arrayScores, Acapt, Acom, NeighCapt, NeighCom,
        speedCapt, nearSearch, t_max1)
    print()
    arraySolutions, arrayScores = V2(
        arraySolutions, arrayScores, Acapt, Acom, NeighCapt, NeighCom,
        speedCapt, nearSearch, t_max2)
    print()
    
    return arraySolutions[0], arrayScores[0]


if __name__ == '__main__':    
    Rcapt = 1
    Rcom = 2
    instanceName = '../Instances/captANOR400_10_80.dat'

    t1 = time.time()
    solution, score = VNS(instanceName, Rcapt, Rcom)
    t2 = time.time()
    print('score : {}'.format(score))
    print('dt : {}\n'.format(t2-t1))
    #print('vectTime : {}\n'.format(vectTime))

    '''
    vectScore = []
    t1 = time.time()
    for i in range(3):
        print(i)
        solution, score = VNS(instanceName, Rcapt, Rcom)
        #print('  > vectTime : {}'.format(vectTime))
        vectScore.append(score)
        print()
    t2 = time.time()
    print('score mean : {}'.format(np.mean(vectScore)))
    print('score min : {}\n'.format(np.min(vectScore)))
    print(t2 - t1)
    '''
