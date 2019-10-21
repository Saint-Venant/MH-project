import numpy as np
import time

import parserInstance



def VNS(instanceName, Rcapt, Rcom):
    # parse data
    Acapt, Acom = parserInstance.parseData(instanceName, Rcapt, Rcom)
    nNodes = Acapt.shape[0]
