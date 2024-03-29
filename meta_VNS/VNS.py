'''
On représente une solution par un vecteur de taille n (nombre de sommets):
    - sol[i] = 1 si cible i reçoit un capteur
    - sol[i] = 0 sinon

In this file:
    - Acapt : matrix of adjacency in the capt graph
    - Acom : matrix of adjacency in the com graph
    - NeighCapt : list of neighbors of each vertex in capt graph
    - NeighCom : list of neighbors of each vertex in com graph

Heuristics: (described in the file heuristics.py)
    * speedCapt
    * nearSearch 
'''
if __name__ == '__main__':
    import sys
    sys.path.append('..\\')

import numpy as np
import time
import multiprocessing as mp

from meta_VNS import parserInstance
from meta_VNS import constraints
from meta_VNS import heuristics
import displaySolution



def greedyDelete(solution, Acapt, Acom, NeighCapt, NeighCom, t_max=None, \
                 givenCandidates=None, \
                 speedCapt=False, nearSearch=[False, []]):
    '''
    For a given solution, delete 1 vertex if possible (remains feasible)

    When candidates are specified, only consider vertices in candidates for
    deletion

    t_max : useless here -> we want the whole neighborhood to be explored as it
    is small
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
        feasible = constraints.checkConstrQuick(
            solBis, i, Acapt, Acom, NeighCapt, NeighCom)
        if not(feasible):
            solBis[i] = 1
            ind += 1

    if (nearSearch[0]) and not(feasible):
        pivots = nearSearch[1]
    else:
        pivots = []
        
    return solBis, feasible, pivots

def deleteNb(solution, candidates, nbDeletions, Acapt, Acom, NeighCapt, \
             NeighCom):
    '''
    Try to delete nbDeletions from the given solution
    '''
    if (nbDeletions == 0):
        improved = True
        return solution, improved
    elif len(candidates) == 0:
        improved = False
        return solution, improved
    else:
        improved = False
        nCandidates = candidates.shape[0]
        ind = 0
        while not(improved) and (ind < nCandidates):
            # Try to delete i
            i = candidates[ind]
            assert(solution[i] == 1)
            solution[i] = 0
            
            feasible = constraints.checkConstrQuick(
                solution, i, Acapt, Acom, NeighCapt, NeighCom)

            if feasible:
                otherCandidates = candidates[ind+1:]
                solution, improved = deleteNb(
                    solution, otherCandidates, nbDeletions-1, Acapt, Acom, \
                    NeighCapt, NeighCom)
                
            if not(improved):
                solution[i] = 1
                ind += 1

        return solution, improved

def getCandidatesDeletion(solution, listInserted, Acapt, NeighCom):
    '''
    listInserted : list of inserted vertices

    Return a list of candidates to be deleted
    '''
    n = solution.shape[0]
    markedCandidates = np.zeros(n, dtype=np.int)
    # -- do not delete vertex 0
    markedCandidates[0] = 1
    # -- do not delete vertices just inserted
    markedCandidates[listInserted] = 1
    # -- can only consider vertices that are selected
    markedCandidates[solution == 0] = 1

    # get neighbors of degree 1 and 2 in the Com graphe
    neighborVertices = []
    for i in listInserted:
        v_i = NeighCom[i][1]
        for j in v_i:
            v_j = NeighCom[j][1]
            for k in v_j:
                if markedCandidates[k] == 0:
                    markedCandidates[k] = 1
                    neighborVertices.append(k)
    neighborVertices = np.array(neighborVertices)
    
    # -- can only consider vertices that are capted by more than themselves
    indexSelected = np.where(solution == 1)[0][1:]
    Scapt = np.sum(Acapt[np.ix_(indexSelected, neighborVertices)], axis=0)
    candidates = neighborVertices[Scapt > 1]
    np.random.shuffle(candidates)
    return candidates
    
def greedyPivot1(solution, Acapt, Acom, NeighCapt, NeighCom, t_max, size,
                 speedCapt=False, nearSearch=[False, []]):
    '''
    For a given solution, test if this move is possible :
        - Select an empty vertex
        - Try to delete 2 other vertices in its neighborhood

    size:
        * 'small' : try candidates for deletion in the neighborhood of the
                inserted vertex
        * 'large' : try to delete any selected vertex
    '''
    assert(size in ['small', 'large'])
    
    solBis = np.copy(solution)
    indexEmpty = np.where(solution == 0)[0]
    indexSelected = np.where(solution == 1)[0][1:]
    nEmpty = indexEmpty.shape[0]
    nNodes = solBis.shape[0]
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

    # count tries (when size = 'large')
    tries = 0
    maxTries = np.abs(np.sqrt(nNodes))

    ind = 0
    improved = False
    while not(improved) and (ind < nEmpty) and (tries < maxTries) and \
          (time.time() < t_max):
        if size == 'large':
            ind = np.random.randint(nEmpty)
        i = indexEmpty[ind]
        solBis[i] = 1

        # candidates to be deleted
        if size == 'small':
            candidates = getCandidatesDeletion(solBis, [i], Acapt, NeighCom)
        elif size == 'large':
            candidates = np.copy(indexSelected)
            np.random.shuffle(candidates)
        nCandidates = candidates.shape[0]

        # Try to delete 2 vertices
        solTer = np.copy(solBis)
        scoreTer = np.sum(solTer) - 1
        solBis, improved = deleteNb(
            solBis, candidates, 2, Acapt, Acom, NeighCapt, NeighCom)
        score = np.sum(solBis) - 1
        if improved:
            assert(score < scoreTer - 1)
        else:
            assert(np.all(solBis == solTer))

        # Try another pivot if the solution cannot be improved
        if not(improved):
            if size == 'small':
                ind += 1
            elif size == 'large':
                tries += 1
            solBis[i] = 0

    if improved:
        pivots = [i]
    elif nearSearch[0]:
        pivots = nearSearch[1]
    else:
        pivots = []
        
    return solBis, improved, pivots

def greedyPivot1_small(solution, Acapt, Acom, NeighCapt, NeighCom, t_max, \
                 speedCapt=False, nearSearch=[False, []]):
    return greedyPivot1(
        solution, Acapt, Acom, NeighCapt, NeighCom, t_max, 'small',
        speedCapt=speedCapt, nearSearch=nearSearch
    )

def greedyPivot1_large(solution, Acapt, Acom, NeighCapt, NeighCom, t_max, \
                 speedCapt=False, nearSearch=[False, []]):
    return greedyPivot1(
        solution, Acapt, Acom, NeighCapt, NeighCom, t_max, 'large',
        speedCapt=speedCapt, nearSearch=nearSearch
    )

def greedyPivot2(solution, Acapt, Acom, NeighCapt, NeighCom, t_max, size,
                 speedCapt=False, nearSearch=[False, []]):
    '''
    For a given solution, test if this move if possible :
        - Select 1 empty vertex + another in its com neighborhood
        - Try to delete 3 vertices in their neighborhood

    size:
        * 'small' : try candidates for deletion in the neighborhood of the
                inserted vertices
        * 'large' : try to delete any selected vertex
    '''
    assert(size in ['small', 'large'])
    
    solBis = np.copy(solution)
    indexEmpty = np.where(solution == 0)[0]
    indexSelected = np.where(solution == 1)[0][1:]
    nEmpty = indexEmpty.shape[0]
    assert(nEmpty > 0)
    if size == 'large':
        assert(nEmpty > 1)
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

    # count tries (when size = 'large')
    tries = 0
    maxTries = np.abs(np.sqrt(nNodes))

    # mark explored pairs
    markedPairs = np.zeros((nNodes, nNodes), dtype=np.int)
    markedPairs[np.arange(nNodes), np.arange(nNodes)] = 1

    ind_i1 = 0
    improved = False
    while not(improved) and (ind_i1 < nEmpty) and (tries < maxTries) and \
          (time.time() < t_max):
        if size == 'large':
            ind_i1 = np.random.randint(nEmpty)
        i1 = indexEmpty[ind_i1]
        v_i1 = NeighCom[i1][1].copy()
        np.random.shuffle(v_i1)
        assert(len(v_i1) > 0)
        
        ind_i2 = 0
        while not(improved) and (ind_i2 < len(v_i1)) and (time.time() < t_max):
            if size == 'small':
                i2 = v_i1[ind_i2]
            elif size == 'large':
                ind_i2 = np.random.randint(nEmpty)
                while ind_i2 == ind_i1:
                    ind_i2 = np.random.randint(nEmpty)
                i2 = indexEmpty[ind_i2]

            if (markedPairs[i1, i2] == 0) and (solBis[i2] == 0):
                v_i2 = NeighCom[i2][1].copy()
                np.random.shuffle(v_i2)
                solBis[i1] = 1
                solBis[i2] = 1

                # candidates to be deleted
                if size == 'small':
                    candidates = getCandidatesDeletion(
                        solBis, [i1, i2], Acapt, NeighCom)
                elif size == 'large':
                    candidates = np.copy(indexSelected)
                    np.random.shuffle(candidates)
                nCandidates = candidates.shape[0]

                # Try to find 3 vertices to delete
                solTer = np.copy(solBis)
                scoreTer = np.sum(solTer) - 1
                solBis, improved = deleteNb(
                    solBis, candidates, 3, Acapt, Acom, NeighCapt, NeighCom)
                score = np.sum(solBis) - 1
                if improved:
                    assert(score < scoreTer - 2)
                else:
                    assert(np.all(solBis == solTer))

                if not(improved):
                    solBis[i1] = 0
                    solBis[i2] = 0

                markedPairs[i1, i2] = 1
                markedPairs[i2, i1] = 1

            if size == 'small':
                ind_i2 += 1
            elif size == 'large':
                ind_2 = len(v_i1)
                tries += 1

        # Try another pivot if the solution cannot be improved
        if not(improved) and size == 'small':
            ind_i1 += 1

    if improved:
        pivots = [i1, i2]
    elif nearSearch[0]:
        pivots = nearSearch[1]
    else:
        pivots = []

    return solBis, improved, pivots

def greedyPivot2_small(solution, Acapt, Acom, NeighCapt, NeighCom, t_max,
                       speedCapt=False, nearSearch=[False, []]):
    return greedyPivot2(solution, Acapt, Acom, NeighCapt, NeighCom, t_max,
                        'small', speedCapt=speedCapt, nearSearch=nearSearch)

def greedyPivot2_large(solution, Acapt, Acom, NeighCapt, NeighCom, t_max,
                       speedCapt=False, nearSearch=[False, []]):
    return greedyPivot2(solution, Acapt, Acom, NeighCapt, NeighCom, t_max,
                        'large', speedCapt=speedCapt, nearSearch=nearSearch)

def greedyPivot3(solution, Acapt, Acom, NeighCapt, NeighCom, t_max, \
                 speedCapt=False, nearSearch=[False, []]):
    '''
    For a given solution, test if this move if possible :
        - Insert 3 vertices
        - Try to delete 4 vertices
    '''
    solBis = np.copy(solution)
    indexEmpty = np.where(solution == 0)[0]
    indexSelected = np.where(solution == 1)[0][1:]
    nEmpty = indexEmpty.shape[0]
    assert(nEmpty > 0)
    nNodes = solution.shape[0]
    print('hello')

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

    ind_i1 = 0
    ind_i2 = 1
    ind_i3 = 2
    improved = False
    while not(improved) and (ind_i3 < nEmpty) and (time.time() < t_max):
        # insert 3 vertices
        i1 = indexEmpty[ind_i1]
        i2 = indexEmpty[ind_i2]
        i3 = indexEmpty[ind_i3]
        assert(solBis[i1] == 0)
        assert(solBis[i2] == 0)
        assert(solBis[i3] == 0)
        solBis[i1] = 1
        solBis[i2] = 1
        solBis[i3] = 1

        # candidates to be deleted
        candidates = getCandidatesDeletion(
            solBis, [i1, i2, i3], Acapt, NeighCom)
        nCandidates = candidates.shape[0]

        # Try to find 4 vertices to deleted
        solTer = np.copy(solBis)
        scoreTer = np.sum(solTer) - 1
        solBis, improved = deleteNb(
            solBis, candidates, 4, Acapt, Acom, NeighCapt, NeighCom)
        score = np.sum(solBis) - 1
        if improved:
            assert(score < scoreTer - 3)
        else:
            assert(np.all(solBis == solTer))
        
        if not(improved):
            solBis[i1] = 0
            solBis[i2] = 0
            solBis[i3] = 0
            
            if ind_i3 < nEmpty - 1:
                ind_i3 += 1
            elif ind_i2 < nEmpty - 2:
                print('      -> ind_i2 {}'.format(ind_i2))
                ind_i2 += 1
                ind_i3 = ind_i2 + 1
            else:
                print('     -> ind_i1 {}'.format(ind_i1))
                ind_i1 += 1
                ind_i2 = ind_i1 + 1
                ind_i3 = ind_i2 + 1

    if improved:
        pivots = [i1, i2, i3]
    elif nearSearch[0]:
        pivots = nearSearch[1]
    else:
        pivots = []

    return solBis, improved, pivots

def localPathPivot1(solution, Acapt, Acom, NeighCapt, NeighCom, t_max, \
                    speedCapt=False, nearSearch=[False, []]):
    '''
    To explore a wider neighborhood, build a path (of max length 12) of
    neighbor solutions resulting in swaps
    '''
    maxLength = 12
    listInserted = []
    solBis = np.copy(solution)

    indexEmpty = np.where(solBis == 0)[0]
    nEmpty = indexEmpty.shape[0]
    np.random.shuffle(indexEmpty)

    improved = False
    indInsert = 0
    while not(improved) and (len(listInserted) < maxLength) and \
          (indInsert < nEmpty):
        i = indexEmpty[indInsert]
        solBis[i] = 1
        listInserted.append(i)

        # candidates to be deleted
        candidates = getCandidatesDeletion(
            solBis, listInserted, Acapt, NeighCom)
        nCandidates = candidates.shape[0]

        # find a candidate to delete
        indDelete = 0
        feasible = False
        while not(feasible) and (indDelete < nCandidates):
            j = candidates[indDelete]
            assert((solBis[j] == 1) and (j > 0))
            solBis[j] = 0
            
            feasible = constraints.checkConstraints(
                solBis, Acapt, Acom, NeighCom)
            if not(feasible):
                solBis[j] = 1
                indDelete += 1

        if feasible:
            # Try to perform a local descent on this new solution
            # -- greedyDelete
            solTer, improved, pivots = greedyDelete(
                solBis, Acapt, Acom, NeighCapt, NeighCom, \
                speedCapt=False, nearSearch=[False, []])
            if not(improved):
                # -- greedyPivot1
                solTer, improved, pivots = greedyPivot1(
                    solBis, Acapt, Acom, NeighCapt, NeighCom, \
                    speedCapt=False, nearSearch=[False, []])

            if improved:
                solBis = solTer
            else:
                indInsert += 1
        else:
            solBis[i] = 0
            listInserted.remove(i)
            indInsert += 1
    print('  * {}'.format(len(listInserted)))

    return solBis, improved, [] 

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
            t_max,
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
       t_max, outputQueue):
    '''
    Compute local search using neighborhoods:
    - greedyDelete

    t_max : maximum time at which the function should stop
    '''
    neighFunctions = [greedyDelete]
    indStart = 0
    count = 0
    
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
            t_max + 60*10,
            neighFunctions,
            indStart
        )
        outputQueue.put((solution, score))
        count += 1
    outputQueue.put('Done')
    print('V0 generated {} solutions'.format(count))

def V1(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
       speedCapt, nearSearch, t_max, outputQueue):
    '''
    Compute local search using neighborhoods:
    - greedyDelete
    - greedyPivot1_small

    t_max : maximum time at which the function should stop
    '''
    neighFunctions = [greedyDelete, greedyPivot1_small]
    indStart = 1

    nbSolutions = len(listSolutions)
    assert(len(listSolutions) == len(listScores))
    i_solution = 0
    while (i_solution < nbSolutions) and (time.time() < t_max):
        nearSearch1 = [nearSearch[0], []]
        solution, score = V(
            listSolutions[i_solution],
            Acapt, Acom,
            NeighCapt, NeighCom,
            speedCapt, nearSearch1,
            t_max,
            neighFunctions,
            indStart
        )
        outputQueue.put((solution, score))
        i_solution += 1
    nbExplored = i_solution

    # add also solutions for which the programm didn't have time to perform
    #   a local search
    while i_solution < nbSolutions:
        outputQueue.put((listSolutions[i_solution], listScores[i_solution]))
        i_solution += 1
    outputQueue.put('Done')
    print('V1 explored {}/{} solutions'.format(nbExplored, nbSolutions))

def V2(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
       speedCapt, nearSearch, t_max, outputQueue):
    '''
    Compute local search using neighborhoods:
    - greedyDelete
    - greedyPivot1_small
    - greedyPivot2_small

    t_max : maximum time at which the function should stop
    '''
    neighFunctions = [greedyDelete, greedyPivot1_small, greedyPivot2_small]
    indStart = 2

    nbSolutions = len(listSolutions)
    i_solution = 0
    while (i_solution < nbSolutions) and (time.time() < t_max):
        nearSearch2 = [nearSearch[0], []]
        solution, score = V(
            listSolutions[i_solution],
            Acapt, Acom,
            NeighCapt, NeighCom,
            speedCapt, nearSearch2,
            t_max,
            neighFunctions,
            indStart
        )
        outputQueue.put((solution, score))
        i_solution += 1
    nbExplored = i_solution

    # add also solutions for which the programm didn't have time to perform
    #   a local search
    while i_solution < nbSolutions:
        outputQueue.put((listSolutions[i_solution], listScores[i_solution]))
        i_solution += 1
    outputQueue.put('Done')
    print('V2 explored {}/{} solutions'.format(nbExplored, nbSolutions))

def V3(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
       speedCapt, nearSearch, t_max, outputQueue):
    '''
    Compute local search using neighborhoods:
    - greedyDelete
    - greedyPivot1_small
    - greedyPivot2_small
    - greedyPivot1_large

    t_max : maximum time at which the function should stop
    '''
    neighFunctions = [greedyDelete, greedyPivot1_small, greedyPivot2_small,
                      greedyPivot1_large]
    indStart = 3

    nbSolutions = len(listSolutions)
    i_solution = 0
    while (i_solution < nbSolutions) and (time.time() < t_max):
        nearSearch3 = [nearSearch[0], []]
        solution, score = V(
            listSolutions[i_solution],
            Acapt, Acom,
            NeighCapt, NeighCom,
            speedCapt, nearSearch3,
            t_max,
            neighFunctions,
            indStart
        )
        outputQueue.put((solution, score))
        i_solution += 1
    nbExplored = i_solution

    # add also solutions for which the programm didn't have time to perform
    #   a local search
    while i_solution < nbSolutions:
        outputQueue.put((listSolutions[i_solution], listScores[i_solution]))
        i_solution += 1
    outputQueue.put('Done')
    print('V3 explored {}/{} solutions'.format(nbExplored, nbSolutions))

def V4(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
       speedCapt, nearSearch, t_max, outputQueue):
    '''
    Compute local search using neighborhoods:
    - greedyDelete
    - greedyPivot1_small
    - greedyPivot2_small
    - greedyPivot1_large
    - greedyPivot2_large

    t_max : maximum time at which the function should stop
    '''
    neighFunctions = [greedyDelete, greedyPivot1_small, greedyPivot2_small,
                      greedyPivot1_large, greedyPivot2_large]
    indStart = 4

    nbSolutions = len(listSolutions)
    i_solution = 0
    while (i_solution < nbSolutions) and (time.time() < t_max):
        nearSearch4 = [nearSearch[0], []]
        solution, score = V(
            listSolutions[i_solution],
            Acapt, Acom,
            NeighCapt, NeighCom,
            speedCapt, nearSearch4,
            t_max,
            neighFunctions,
            indStart
        )
        outputQueue.put((solution, score))
        i_solution += 1
    nbExplored = i_solution

    # add also solutions for which the programm didn't have time to perform
    #   a local search
    while i_solution < nbSolutions:
        outputQueue.put((listSolutions[i_solution], listScores[i_solution]))
        i_solution += 1
    outputQueue.put('Done')
    print('V4 explored {}/{} solutions'.format(nbExplored, nbSolutions))

def V5(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
       speedCapt, nearSearch, t_max, outputQueue):
    '''
    Compute local search using neighborhoods:
    - greedyDelete
    - greedyPivot1_small
    - greedyPivot2_small
    - greedyPivot3

    t_max : maximum time at which the function should stop
    '''
    neighFunctions = [greedyDelete, greedyPivot1_small, greedyPivot2_small,
                      greedyPivot3]
    indStart = 3

    nbSolutions = len(listSolutions)
    i_solution = 0
    while (i_solution < nbSolutions) and (time.time() < t_max):
        nearSearch3 = [nearSearch[0], []]
        solution, score = V(
            listSolutions[i_solution],
            Acapt, Acom,
            NeighCapt, NeighCom,
            speedCapt, nearSearch3,
            t_max,
            neighFunctions,
            indStart
        )
        outputQueue.put((solution, score))
        i_solution += 1
    nbExplored = i_solution

    # add also solutions for which the programm didn't have time to perform
    #   a local search
    while i_solution < nbSolutions:
        outputQueue.put((listSolutions[i_solution], listScores[i_solution]))
        i_solution += 1
    outputQueue.put('Done')
    print('V5 explored {}/{} solutions'.format(nbExplored, nbSolutions))

def V6(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
       speedCapt, nearSearch, t_max, outputQueue):
    '''
    Compute local search using neighborhoods:
    - greedyDelete
    - greedyPivot1_small
    - greedyPivot2_small
    - localPathPivot1

    t_max : maximum time at which the function should stop
    '''
    neighFunctions = [greedyDelete, greedyPivot1_small, greedyPivot2_small,
                      localPathPivot1]
    indStart = 3

    nbSolutions = len(listSolutions)
    i_solution = 0
    while (i_solution < nbSolutions) and (time.time() < t_max):
        nearSearch4 = [nearSearch[0], []]
        solution, score = V(
            listSolutions[i_solution],
            Acapt, Acom,
            NeighCapt, NeighCom,
            speedCapt, nearSearch4,
            t_max,
            neighFunctions,
            indStart
        )
        outputQueue.put((solution, score))
        i_solution += 1
    nbExplored = i_solution

    # add also solutions for which the programm didn't have time to perform
    #   a local search
    while i_solution < nbSolutions:
        outputQueue.put((listSolutions[i_solution], listScores[i_solution]))
        i_solution += 1
    outputQueue.put('Done')
    print('V6 explored {}/{} solutions'.format(nbExplored, nbSolutions))

def collectResults(jobs, outputQueue):
    '''
    jobs : list of multiprocessing.Process that run in parallel

    Return the list of results, collected in the ouput multiprocessing.Queue
    + join the processes before returning any result
    '''
    results = []
    countDone = 0
    while countDone < len(jobs):
        time.sleep(0.01)
        x = outputQueue.get()
        if x == 'Done':
            countDone += 1
        else:
            results.append(x)
    for p in jobs:
        p.join()
    return results

def splitWork(listSolutions, listScores, nbProcesses):
    '''
    Return a list, of length nbProcesses), in which we have split the solutions
    Manage to put solutions with lower score first
    '''
    # sort solutions by increasing score
    arraySolutions = np.array(listSolutions)
    arrayScores = np.array(listScores)
    indexSort = np.argsort(arrayScores)
    arraySolutions = arraySolutions[indexSort]
    arrayScores = arrayScores[indexSort]
    
    # split the work
    listWork = [[[], []] for i in range(nbProcesses)]
    i_work = 0
    for i_solution in range(arraySolutions.shape[0]):
        listWork[i_work][0].append(arraySolutions[i_solution])
        listWork[i_work][1].append(arrayScores[i_solution])
        i_work = (i_work + 1)%nbProcesses

    return listWork

def runParallelV0(solutionInitial, Acapt, Acom, NeighCapt, NeighCom, \
                  speedCapt, nearSearch, t_max0, nbProcesses):
    '''
    Function to run in parallel several jobs
    and collect the results
    '''
    outputQueue = mp.Queue()
    jobs = []
    for i in range(nbProcesses):
        p = mp.Process(
            target=V0,
            args=(
                solutionInitial,
                Acapt, Acom,
                NeighCapt, NeighCom,
                speedCapt, nearSearch,
                t_max0,
                outputQueue
            )
        )
        jobs.append(p)
        p.start()
    results = collectResults(jobs, outputQueue)
    return results

def runParallelV1(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
                  speedCapt, nearSearch, t_max1, nbProcesses):
    '''
    Function to run in parallel several jobs
    and collect the results
    '''
    # split the solutions between the processes
    listWork = splitWork(listSolutions, listScores, nbProcesses)
    
    outputQueue = mp.Queue()
    jobs = []
    for i in range(nbProcesses):
        p = mp.Process(
            target=V1,
            args=(
                listWork[i][0], listWork[i][1],
                Acapt, Acom,
                NeighCapt, NeighCom,
                speedCapt, nearSearch,
                t_max1,
                outputQueue
            )
        )
        jobs.append(p)
        p.start()
    results = collectResults(jobs, outputQueue)
    return results

def runParallelV2(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
                  speedCapt, nearSearch, t_max2, nbProcesses):
    '''
    Function to run in parallel several jobs
    and collect the results
    '''
    # split the solutions between the processes
    listWork = splitWork(listSolutions, listScores, nbProcesses)
    
    outputQueue = mp.Queue()
    jobs = []
    for i in range(nbProcesses):
        p = mp.Process(
            target=V2,
            args=(
                listWork[i][0], listWork[i][1],
                Acapt, Acom,
                NeighCapt, NeighCom,
                speedCapt, nearSearch,
                t_max2,
                outputQueue
            )
        )
        jobs.append(p)
        p.start()
    results = collectResults(jobs, outputQueue)
    return results

def runParallelV3(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
                  speedCapt, nearSearch, t_max3, nbProcesses):
    '''
    Function to run in parallel several jobs
    and collect the results
    '''
    # split the solutions between the processes
    listWork = splitWork(listSolutions, listScores, nbProcesses)
    
    outputQueue = mp.Queue()
    jobs = []
    for i in range(nbProcesses):
        p = mp.Process(
            target=V3,
            args=(
                listWork[i][0], listWork[i][1],
                Acapt, Acom,
                NeighCapt, NeighCom,
                speedCapt, nearSearch,
                t_max3,
                outputQueue
            )
        )
        jobs.append(p)
        p.start()
    results = collectResults(jobs, outputQueue)
    return results

def runParallelV4(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
                  speedCapt, nearSearch, t_max4, nbProcesses):
    '''
    Function to run in parallel several jobs
    and collect the results
    '''
    # split the solutions between the processes
    listWork = splitWork(listSolutions, listScores, nbProcesses)
    
    outputQueue = mp.Queue()
    jobs = []
    for i in range(nbProcesses):
        p = mp.Process(
            target=V4,
            args=(
                listWork[i][0], listWork[i][1],
                Acapt, Acom,
                NeighCapt, NeighCom,
                speedCapt, nearSearch,
                t_max4,
                outputQueue
            )
        )
        jobs.append(p)
        p.start()
    results = collectResults(jobs, outputQueue)
    return results

def runParallelV5(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
                  speedCapt, nearSearch, t_max5, nbProcesses):
    '''
    Function to run in parallel several jobs
    and collect the results
    '''
    # split the solutions between the processes
    listWork = splitWork(listSolutions, listScores, nbProcesses)
    
    outputQueue = mp.Queue()
    jobs = []
    for i in range(nbProcesses):
        p = mp.Process(
            target=V5,
            args=(
                listWork[i][0], listWork[i][1],
                Acapt, Acom,
                NeighCapt, NeighCom,
                speedCapt, nearSearch,
                t_max5,
                outputQueue
            )
        )
        jobs.append(p)
        p.start()
    results = collectResults(jobs, outputQueue)
    return results

def runParallelV6(listSolutions, listScores, Acapt, Acom, NeighCapt, NeighCom, \
                  speedCapt, nearSearch, t_max6, nbProcesses):
    '''
    Function to run in parallel several jobs
    and collect the results
    '''
    # split the solutions between the processes
    listWork = splitWork(listSolutions, listScores, nbProcesses)
    
    outputQueue = mp.Queue()
    jobs = []
    for i in range(nbProcesses):
        p = mp.Process(
            target=V6,
            args=(
                listWork[i][0], listWork[i][1],
                Acapt, Acom,
                NeighCapt, NeighCom,
                speedCapt, nearSearch,
                t_max6,
                outputQueue
            )
        )
        jobs.append(p)
        p.start()
    results = collectResults(jobs, outputQueue)
    return results


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
    dt0 = 0.1*dtMax
    dt1 = 0.2*dtMax
    dt2 = 0.4*dtMax
    dt3 = 0.15*dtMax
    dt4 = 0#0.15*dtMax
    dt5 = 0
    dt6 = 0

    # multiprocessing (parallel programming)
    nbProcesses = 6

    # heuristics
    speedCapt = (Rcapt < Rcom)
    nearSearch = [True, []]

    # initialization
    solutionInitial = np.ones(nNodes, dtype=np.int)
    assert(constraints.checkConstraints(solutionInitial, Acapt, Acom, NeighCom))
    score = np.sum(solutionInitial) - 1

    # iterations over neighborhoods
    # -- V0
    z1 = time.time()
    results = runParallelV0(
        solutionInitial,
        Acapt, Acom,
        NeighCapt, NeighCom,
        speedCapt, nearSearch,
        time.time() + dt0,
        nbProcesses)
    z2 = time.time()
    listSolutions = [res[0] for res in results]
    listScores = [res[1] for res in results]
    print(' --- V0 ---')
    print('  > dt = {}'.format(z2 - z1))
    print('  > dt_max = {}'.format(dt0))
    print('  > best score : {}\n'.format(np.min(listScores)))

    # -- V1
    z1 = time.time()
    results = runParallelV1(
        listSolutions, listScores,
        Acapt, Acom,
        NeighCapt, NeighCom,
        speedCapt, nearSearch,
        time.time() + dt1,
        nbProcesses)
    z2 = time.time()
    listSolutions = [res[0] for res in results]
    listScores = [res[1] for res in results]
    print(' --- V1 ---')
    print('  > dt = {}'.format(z2 - z1))
    print('  > dt_max = {}'.format(dt1))
    print('  > best score : {}\n'.format(np.min(listScores)))

    # -- V2
    z1 = time.time()
    results = runParallelV2(
        listSolutions, listScores,
        Acapt, Acom,
        NeighCapt, NeighCom,
        speedCapt, nearSearch,
        time.time() + dt2,
        nbProcesses)
    z2 = time.time()
    listSolutions = [res[0] for res in results]
    listScores = [res[1] for res in results]
    print(' --- V2 ---')
    print('  > dt = {}'.format(z2 - z1))
    print('  > dt_max = {}'.format(dt2))
    print('  > best score : {}\n'.format(np.min(listScores)))

    # -- V3
    z1 = time.time()
    results = runParallelV3(
        listSolutions, listScores,
        Acapt, Acom,
        NeighCapt, NeighCom,
        speedCapt, nearSearch,
        time.time() + dt3,
        nbProcesses)
    z2 = time.time()
    listSolutions = [res[0] for res in results]
    listScores = [res[1] for res in results]
    print(' --- V3 ---')
    print('  > dt = {}'.format(z2 - z1))
    print('  > dt_max = {}'.format(dt3))
    print('  > best score : {}\n'.format(np.min(listScores)))

    # -- V4
    z1 = time.time()
    results = runParallelV4(
        listSolutions, listScores,
        Acapt, Acom,
        NeighCapt, NeighCom,
        speedCapt, nearSearch,
        time.time() + dt4,
        nbProcesses)
    z2 = time.time()
    listSolutions = [res[0] for res in results]
    listScores = [res[1] for res in results]
    print(' --- V4 ---')
    print('  > dt = {}'.format(z2 - z1))
    print('  > dt_max = {}'.format(dt4))
    print('  > best score : {}\n'.format(np.min(listScores)))

    # -- V5
    '''
    z1 = time.time()
    results = runParallelV5(
        listSolutions, listScores,
        Acapt, Acom,
        NeighCapt, NeighCom,
        speedCapt, nearSearch,
        time.time() + dt5,
        nbProcesses)
    z2 = time.time()
    listSolutions = [res[0] for res in results]
    listScores = [res[1] for res in results]
    print(' --- V5 ---')
    print('  > dt = {}'.format(z2 - z1))
    print('  > dt_max = {}'.format(dt5))
    print('  > best score : {}\n'.format(np.min(listScores)))

    # -- V6
    z1 = time.time()
    results = runParallelV6(
        listSolutions, listScores,
        Acapt, Acom,
        NeighCapt, NeighCom,
        speedCapt, nearSearch,
        time.time() + dt6,
        nbProcesses)
    z2 = time.time()
    listSolutions = [res[0] for res in results]
    listScores = [res[1] for res in results]
    print(' --- V6 ---')
    print('  > dt = {}'.format(z2 - z1))
    print('  > dt_max = {}'.format(dt6))
    print('  > best score : {}\n'.format(np.min(listScores)))
    '''

    # get best solution
    i_best = np.argmin(listScores)
    
    return listSolutions[i_best], listScores[i_best], listSolutions


if __name__ == '__main__':    
    Rcapt = 1
    Rcom = 1
    instanceName = '../Instances/captGRID900_30_30.dat'

    t1 = time.time()
    solution, score, listSolutions = VNS(instanceName, Rcapt, Rcom, dtMax=12*60)
    t2 = time.time()
    print('score : {}'.format(score))
    print('dt : {}\n'.format(t2-t1))

    # plot solution
    displaySolution.display(instanceName, Rcapt, Rcom, solution, score)
