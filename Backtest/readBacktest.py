import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

backtestFiles = [
    'Backtest_VNS_23-10.pkl',
    'Backtest_VNS_24-10.pkl'
]

# read results
dictResults = dict()
for backtestName in backtestFiles:
    with open(backtestName, 'rb') as f:
        result = pkl.load(f)
    dictResults[backtestName] = result

# give index to caracteristics
dictBacktest = dict()
dictInstance = dict()
dictRR = dict()
for backtestName in backtestFiles:
    dictBacktest[backtestName] = len(dictBacktest)
    results = dictResults[backtestName]
    for res in results:
        instanceName = res[0]
        if instanceName not in dictInstance:
            dictInstance[instanceName] = len(dictInstance)
        Rcapt, Rcom = res[1:3]
        if (Rcapt, Rcom) not in dictRR:
            dictRR[(Rcapt, Rcom)] = len(dictRR)

# store info in a 3D array
nbBacktest = len(dictBacktest)
nbInstance = len(dictInstance)
nbRR = len(dictRR)
array = []
for i in range(nbBacktest):
    x = []
    for j in range(nbInstance):
        y = nbRR * [0]
        x.append(y)
    array.append(x)

# sort values
for backtestName in backtestFiles:
    i = dictBacktest[backtestName]
    results = dictResults[backtestName]
    for res in results:
        instanceName, Rcapt, Rcom = res[:3]
        j = dictInstance[instanceName]
        k = dictRR[(Rcapt, Rcom)]
        score = res[4]
        array[i][j][k] = score

# plot scores
width = 0.35
for instanceName in dictInstance.keys():
    j = dictInstance[instanceName]
    fig, ax = plt.subplots(num=instanceName)
    for backtestName in dictBacktest:
        i = dictBacktest[backtestName]
        vectScores = [0]*nbRR
        labels = [0]*nbRR
        for RR in dictRR.keys():
            k = dictRR[RR]
            vectScores[k] = array[i][j][k]
            labels[k] = RR
        x = np.arange(nbRR)
        rect = ax.bar(x - (nbBacktest - 1)*width/2 + i*width, vectScores, \
                      width, label=backtestName)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.title(instanceName)
    plt.ylabel('Score')
    plt.legend()
        
plt.show()
