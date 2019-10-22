import numpy as np
import time

import parserInstance



def greedyDelete(solution, Acapt, Acom):
    solBis = np.copy(solution)
    indexSelected = np.where(solution == 1)[0]
    for i in indexSelected:
        

def VNS(instanceName, Rcapt, Rcom):
    # parse data
    Acapt, Acom, NeighCapt, NeighCom = parserInstance.parseData(
        instanceName, Rcapt, Rcom)
    nNodes = Acapt.shape[0]

    # parameters
    sizePop = 10

    
