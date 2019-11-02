'''
Hold functions to evaluate constraints

On représente une solution par un vecteur de taille n (nombre de cibles):
    - sol[i] = 1 si cible 1 reçoit un capteur
    - sol[i] = 0 sinon

In this file:
    - Acapt : matrix of adjacency in the capt graph
    - Acom : matrix of adjacency in the com graph
    - NeighCapt : list of neighbors of each vertex in capt graph
    - NeighCom : list of neighbors of each vertex in com graph
'''
import numpy as np


def contrainteCapt(solution, Acapt):
    '''
    Return a list v of n elements where:
        - n = number of vertices
        - v_i = * 0 if capt constraint is respected for vertex i
                * 1 otherwise
    '''
    assert(solution[0] == 1)

    indexSelected = np.where(solution == 1)[0]
    Scapt = np.sum(Acapt[indexSelected, :], axis=0)
    violationCapt = np.where(Scapt > 0, 0, 1)
    return violationCapt

def contrainteCom(solution, Acom, NeighCom):
    '''
    Return a list v of n elements where:
        - n = number of vertices
        - v_i = * 0 if com constraint is respected for vertex i
                * 1 otherwise
    '''
    n = len(solution)
    assert(solution[0] == 1)

    violationCom = np.copy(solution)
    violationCom[0] = 0
    file = [0]

    while len(file) > 0:
        i = file.pop(0)
        v = NeighCom[i][1]
        for j in v:
            if (violationCom[j] == 1) and (Acom[i, j] == 1):
                violationCom[j] = 0
                file.append(j)

    return violationCom

def checkConstraints(solution, Acapt, Acom, NeighCom):
    '''
    For a given solution, checks whether or not the constraints are respected
    '''
    '''
    violationCom = contrainteCom(solution, Acom, NeighCom)
    sCom = np.sum(violationCom)
    ok = (sCom == 0)
    if ok:
        violationCapt = contrainteCapt(solution, Acapt)
        sCapt = np.sum(violationCapt)
        ok = (sCapt == 0)
    '''
    violationCapt = contrainteCapt(solution, Acapt)
    sCapt = np.sum(violationCapt)
    ok = (sCapt == 0)
    if ok:
        violationCom = contrainteCom(solution, Acom, NeighCom)
        sCom = np.sum(violationCom)
        ok = (sCom == 0)
    return ok
