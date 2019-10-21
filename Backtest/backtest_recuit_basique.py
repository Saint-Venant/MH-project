import sys
sys.path.append('..//')

import recuit



names = [
    'captANOR225_9_20.dat',
    'captANOR400_10_80.dat',
    'captANOR625_15_100.dat',
    'captANOR900_12_20.dat',
    'captANOR1500_15_100.dat',
    'captANOR1500_21_500.dat']
instanceNames = ['../Instances/{}'.format(x) for x in names]


vectR = [(1,1), (1,2), (2,2), (2,3)]
