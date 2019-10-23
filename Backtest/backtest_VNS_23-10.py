import sys
sys.path.append('..//')

import numpy as np
import pickle as pkl
import time

import VNS



names = [
    'captANOR225_9_20.dat',
    'captANOR400_10_80.dat',
    'captANOR625_15_100.dat',
    'captANOR900_15_20.dat',
    'captANOR1500_15_100.dat',
    'captANOR1500_21_500.dat']
instanceNames = ['../Instances/{}'.format(x) for x in names]


vectR = [(1,1), (1,2), (2,2), (2,3)]

vectResult = []

for instanceName in instanceNames:
    for (Rcapt, Rcom) in vectR:
        t1 = time.time()
        solution, score = VNS.VNS(instanceName, Rcapt, Rcom, dtMax=60*5)
        t2 = time.time()
        result = [instanceName, Rcapt, Rcom, solution, score, t2-t1]
        vectResult.append(result)

        print('instanceName : {}'.format(instanceName))
        print('Rcapt : {} ; Rcom : {}'.format(Rcapt, Rcom))
        print('  > score = {}'.format(score))
        print('  > dt = {}'.format(t2 - t1))
        print()



with open('backtest_VNS_23-10.pkl', 'wb') as f:
    pkl.dump(vectResult, f)
