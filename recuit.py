'''
Recuit simulé basique permettant d'obtenir rapidement une première solution
de bonne qualité

On représente une solution par un vecteur de taille n (nombre de cibles):
    - sol[i] = 1 si cible 1 reçoit un capteur
    - sol[i] = 0 sinon
'''
import numpy as np

import parserInstance


def contrainteCapt(solution, Acapt):
    n = len(solution)
    assert(solution[0] == 1)
    
    violationCapt = np.ones(n, dtype=np.int)
    indexCapteurs = np.where(solution == 1)[0]
    for i in indexCapteurs:
        for j in range(n):
            if (violationCapt[j] == 1) and (Acapt[i, j] == 1):
                violationCapt[j] = 0

    return violationCapt

def contrainteCom(solution, Acom):
    n = len(solution)
    assert(solution[0] == 1)

    violationCom = np.ones(n, dtype=np.int)
    violationCom[0] = 0
    file = [0]

    while len(file) > 0:
        i = file[0]
        file = file[1:]
        for j in range(n):
            if (violationCom[j] == 1) and (Acom[i, j] == 1):
                violationCom[j] = 0
                file.append(j)

    return violationCom

def computeEnergy(solution, Acapt, Acom, coefCapt, coefCom, coefSize):
    violationCapt = contrainteCapt(solution, Acapt)
    violationCom = contrainteCom(solution, Acom)
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

def recuit(instanceName, Rcapt, Rcom):
    # parse data
    Acapt, Acom = parserInstance.parseData(instanceName, Rcapt, Rcom)
    nNodes = Acapt.shape[0]

    # parameters
    coefCapt = 5
    coefCom = 10
    coefSize = 1
    maxIter = 10**3
    T0 = 1
    alpha = 0.9999
    
    # initialisation
    bestSolution = np.ones(nNodes, dtype=np.int)
    bestScore = np.sum(bestSolution)
    bestEnergy = computeEnergy(bestSolution, Acapt, Acom, coefCapt, coefCom, \
                               coefSize)
    
    solution = np.copy(bestSolution)
    score = bestScore
    energy = bestEnergy

    T = T0
    vectScore = []

    # iterations
    it = 0
    while it < maxIter:
        solBis = V1(solution)
        energyBis = computeEnergy(solBis, Acapt, Acom, coefCapt, coefCom, \
                                  coefSize)
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

        it += 1
        T = alpha*T
        vectScore.append(np.sum(solution))

    return bestSolution, vectScore
        

if __name__ == '__main__':
    Rcapt = 1
    Rcom = 2
    instanceName = 'Instances/captANOR225_9_20.dat'

    bestSolution, vectScore = recuit(instanceName, Rcapt, Rcom)
