import sys
sys.path.append('..//')

import numpy as np
import pickle as pkl
import time

from meta_VNS import VNS



names = [
    'captANOR225_9_20.dat',
    'captANOR400_10_80.dat',
    'captANOR625_15_100.dat',
    'captANOR900_15_20.dat',
    'captANOR1500_15_100.dat',
    'captANOR1500_21_500.dat',
    'captGRID100_10_10.dat',
    'captGRID225_15_15.dat',
    'captGRID400_20_20.dat',
    'captGRID625_25_25.dat',
    'captGRID900_30_30.dat',
    'captGRID1600_40_40.dat',
    'captTRUNC87_10_10.dat',
    'captTRUNC90_10_10.dat',
    'captTRUNC199_15_15.dat',
    'captTRUNC200_15_15.dat',
    'captTRUNC332_20_20.dat',
    'captTRUNC351_20_20.dat',
    'captTRUNC436_25_25.dat',
    'captTRUNC557_25_25.dat',
    'captTRUNC669_30_30.dat',
    'captTRUNC800_30_30.dat',
    'captTRUNC1223_40_40.dat',
    'captTRUNC1425_40_40.dat']
instanceNames = ['../Instances/{}'.format(x) for x in names]

sizes = [225, 400, 625, 900, 1500, 100, 225, 400, 625, 900, 1600, 87, 90, \
         199, 200, 332, 351, 436, 557, 669, 800, 1223, 1425]
times = [60*n/100 for n in sizes]


vectR = [(1,1), (1,2), (2,2), (2,3)]

if __name__ == '__main__':

    vectResult = []

    for (instanceName, dtMax) in zip(instanceNames, times):
        for (Rcapt, Rcom) in vectR:
            t1 = time.time()
            solution, score, listSolutions = VNS.VNS(
                instanceName, Rcapt, Rcom, dtMax=dtMax)
            t2 = time.time()
            result = [instanceName, Rcapt, Rcom, solution, score, t2-t1, \
                      listSolutions]
            vectResult.append(result)

            print('instanceName : {}'.format(instanceName))
            print('Rcapt : {} ; Rcom : {}'.format(Rcapt, Rcom))
            print('  > score = {}'.format(score))
            print('  > dt = {}'.format(t2 - t1))
            print()

    with open('backtest_VNS_05-11.pkl', 'wb') as f:
        pkl.dump(vectResult, f)
