'''
Evaluate the average performance of our greedy algorithm that deletes random
vertices
'''
import numpy as np
import time

from meta_VNS import parserInstance
from meta_VNS.VNS import greedyDelete


def eval(instanceName, Rcapt, Rcom, nbTries):
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
