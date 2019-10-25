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

def heurSpeedCapt(solution, neighborVertices, Acapt):
    '''
    neighborVertices : list of neighbor vertices

    Return candidates for deletion that are capted by at least one other vertex
    that themselves
    '''
    indexSelected = np.where(solution == 1)[0][1:]
    cand1 = neighborVertices[solution[neighborVertices] == 1]
    cand2 = cand1[cand1 > 0]
    cand3 = cand2[np.sum(Acapt[np.ix_(indexSelected, cand2)], axis=0) > 1]
    return cand3
