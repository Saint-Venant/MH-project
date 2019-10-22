'''
Recuit simulé basique permettant d'obtenir rapidement une première solution
de bonne qualité

On représente une solution par un vecteur de taille n (nombre de cibles):
    - sol[i] = 1 si cible 1 reçoit un capteur
    - sol[i] = 0 sinon
'''
import numpy as np
import time

import parserInstance


def contrainteCapt(solution, Acapt):
    assert(solution[0] == 1)

    indexSelected = np.where(solution == 1)[0]
    Scapt = np.sum(Acapt[indexSelected, :], axis=0)
    violationCapt = np.where(Scapt > 0, 0, 1)
    return violationCapt

def contrainteCom(solution, Acom, NeighCom):
    n = len(solution)
    assert(solution[0] == 1)

    violationCom = np.ones(n, dtype=np.int)
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

def computeEnergy(solution, Acapt, Acom, NeighCom, coefCapt, coefCom, coefSize):
    violationCapt = contrainteCapt(solution, Acapt)
    violationCom = contrainteCom(solution, Acom, NeighCom)
    energy = coefCapt*np.sum(violationCapt) + \
             coefCom*np.sum(violationCom) + \
             coefSize*np.sum(solution)
    return energy

def V1(solution):
    n = len(solution)
    solBis = np.copy(solution)
    i = np.random.randint(1, n)
    solBis[i] = (solBis[i] + 1)%2
    return solBis

def recuit(instanceName, Rcapt, Rcom, maxIter=10**4, verbose=False):
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
    bestScore = np.sum(bestSolution)
    bestEnergy = computeEnergy(
        bestSolution, Acapt, Acom, NeighCom, coefCapt, coefCom, coefSize)
    
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
        solBis = V1(solution)
        energyBis = computeEnergy(
            solBis, Acapt, Acom, NeighCom, coefCapt, coefCom, coefSize)
        if energyBis < energy:
            solution = solBis
            energy = energyBis
        else:
            delta = energyBis - energy
            proba = np.exp(-delta/T)
            if np.random.random() < proba:
                solution = solBis
                energy = energyBis

        if energy < bestEnergy:
            bestEnergy = energy
            bestSolution = solution
        score = np.sum(solution)
        if score < bestScore:
            bestScore = score

        it += 1
        T = alpha*T
        vectScore.append(np.sum(solution))
        
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
