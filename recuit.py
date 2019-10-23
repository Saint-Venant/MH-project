'''
Recuit simulé basique permettant d'obtenir rapidement une première solution
de bonne qualité

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
import time

import parserInstance
import constraints



def computeEnergy(solution, Acapt, Acom, NeighCom, coefCapt, coefCom, coefSize):
    '''
    Given a certain solution (maybe not admissible), compute the energy function
    for the recuit

    E = coefCapt*(number of capt constraints unrespected) +
        coefCom*(number of com constraints unrespected) +
        coefSize*(number of vertices in the solution)

    Return:
        - energy
        - ok : * True if solution is feasbible
               * False otherwise
    '''
    violationCapt = constraints.contrainteCapt(solution, Acapt)
    violationCom = constraints.contrainteCom(solution, Acom, NeighCom)
    sCapt = np.sum(violationCapt)
    sCom = np.sum(violationCom)
    sSol = np.sum(solution) - 1
    energy = coefCapt*sCapt + coefCom*sCom + coefSize*sSol
    ok = (sCapt == 0) and (sCom == 0)
    return energy, ok

def V1(solution):
    '''
    Neighbor fuction 1 : Hamming-distance = 1
    '''
    n = len(solution)
    solBis = np.copy(solution)
    i = np.random.randint(1, n)
    solBis[i] = (solBis[i] + 1)%2
    return solBis

def recuit(instanceName, Rcapt, Rcom, maxIter=10**4, verbose=False):
    '''
    Run a recuit simulé

    maxIter : stop criteria
    coefCapt, coefCom, coefSize : coefficients for the energy function

    T0 : initial temperature
    T : temperature at each iteration
    T_(it+1) = alpha * T_(it)

    score = * number of vertices in the solution if feasible (except 0)
            * inf if solution if not feasible

    Return:
        - bestsolution : best solution encountered for the score
        - vectScore : score of the solution at each iteration
    '''
    # parse data
    Acapt, Acom, NeighCapt, NeighCom = parserInstance.parseData(
        instanceName, Rcapt, Rcom)
    nNodes = Acapt.shape[0]

    # parameters
    coefCapt = 100
    coefCom = 100
    coefSize = 1
    T0 = 1
    alpha = 0.99982
    
    # initialisation
    bestSolution = np.ones(nNodes, dtype=np.int)
    bestScore = np.sum(bestSolution) - 1
    bestEnergy, ok = computeEnergy(
        bestSolution, Acapt, Acom, NeighCom, coefCapt, coefCom, coefSize)
    assert(ok)
    
    solution = np.copy(bestSolution)
    score = bestScore
    energy = bestEnergy

    T = T0
    vectScore = []
    msg = 'It = {} ; energy : {} ; bestEnergy : {} ; bestScore : {}'

    if verbose:
        print(msg.format('Init', energy, bestEnergy, bestScore))
        print('    T = {}'.format(T))

    # iterations
    it = 0
    while it < maxIter:
        # evaluate a neighbor
        solBis = V1(solution)
        energyBis, okBis = computeEnergy(
            solBis, Acapt, Acom, NeighCom, coefCapt, coefCom, coefSize)

        # evaluate whether or not to update the solution
        move = False
        if energyBis < energy:
            move = True
        else:
            delta = energyBis - energy
            proba = np.exp(-delta/T)
            if np.random.random() < proba:
                move = True

        if move:
            solution = solBis
            energy = energyBis
            ok = okBis

        # update track of best solution
        if energy < bestEnergy:
            bestEnergy = energy
        if ok:
            score = np.sum(solution) - 1
        else:
            score = np.inf
        if score < bestScore:
            bestScore = score
            bestSolution = solution

        it += 1
        T = alpha*T
        vectScore.append(score)
        
        if verbose and (it%100 == 0):
            print(msg.format(it, energy, bestEnergy, bestScore))
            print('    T = {}'.format(T))

    return bestSolution, vectScore
        

if __name__ == '__main__':
    Rcapt = 1
    Rcom = 2
    instanceName = 'Instances/captANOR225_9_20.dat'

    t1 = time.time()
    bestSolution, vectScore = recuit(instanceName, Rcapt, Rcom, verbose=True)
    t2 = time.time()
    print('\ndt : {}'.format(t2-t1))
