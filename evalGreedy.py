'''
Evaluate the average performance of our greedy algorithm that deletes random
vertices
'''
import numpy as np
import time

from meta_VNS import parserInstance
from meta_VNS.VNS import greedyDelete


def evalPerf(instanceName, Rcapt, Rcom, nbTries):
    '''
    For a given instanceName and values of Rcapt and Rcom, compute nbTries
    greedy descents
    '''
    # parse data
    Acapt, Acom, NeighCapt, NeighCom = parserInstance.parseData(
        instanceName, Rcapt, Rcom)
    nNodes = Acapt.shape[0]
    
    listScores = []
    listDt = []
    for i in range(nbTries):
        descent = True
        solution = np.ones(nNodes, dtype=np.int)
        t1 = time.time()
        while descent:
            solution, descent, pivots = greedyDelete(
                solution, Acapt, Acom, NeighCapt, NeighCom
            )
        t2 = time.time()
        score = np.sum(solution) - 1
        listScores.append(score)
        listDt.append(t2 - t1)

    return listScores, listDt


if __name__ == '__main__':
    # instance names
    names = [
        'captGRID100_10_10.dat',
        'captGRID400_20_20.dat',
        'captGRID1600_40_40.dat'
    ]
    instanceNames = ['Instances/{}'.format(name) for name in names]

    # (Rcapt, Rcom)
    vectR = [(1,1), (1,2)]

    results = []
    nbTries = 100
    for instanceName in instanceNames:
        print(' --- {} ---'.format(instanceName))
        for (Rcapt, Rcom) in vectR:
            listScores, listDt = evalPerf(instanceName, Rcapt, Rcom, nbTries)
            results.append([
                instanceName,
                Rcapt, Rcom,
                nbTries,
                listScores, listDt
            ])
            print('  > Rcapt = {} ; Rcom = {}'.format(Rcapt, Rcom))
            scoreMean = np.mean(listScores)
            scoreSigma = np.sqrt(np.var(listScores))
            dtMean = np.mean(listDt)
            dtSigma = np.sqrt(np.var(listDt))
            print('      * score = {}  ;  sigma = {}'.format(scoreMean, \
                                                             scoreSigma))
            print('      * dt = {}  ;  sigma = {}'.format(dtMean, dtSigma))
        print()
