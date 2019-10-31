'''
Display a solution on a map
'''
import numpy as np
import matplotlib.pyplot as plt

from meta_VNS import parserInstance


def display(instanceName, Rcapt, Rcom, solution, score):
    '''
    Display a graphical representation of the solution
    '''
    instance = parserInstance.readInstance(instanceName)
    Acapt, Acom, NeighCapt, NeighCom = parserInstance.parseData(
        instanceName, Rcapt, Rcom)
    fig, ax = plt.subplots()

    # plot all vertex
    x = np.array([s[1] for s in instance])
    y = np.array([s[2] for s in instance])
    plt.scatter(x, y)

    # plot selected vertices
    indexSelected = np.where(solution == 1)[0]
    plt.scatter(x[indexSelected], y[indexSelected], color='r')
    plt.scatter(x[0], y[0], color='magenta')

    # plot Com connections
    N = indexSelected.shape[0]
    for ind_i in range(N):
        i = indexSelected[ind_i]
        v = NeighCom[i][1]
        for j in v:
            plt.plot([x[i], x[j]], [y[i], y[j]], c='g')
        if i > 0:
            ax.add_patch(plt.Circle((x[i], y[i]), Rcapt, color='orange', \
                                    alpha=0.1))

    title = '{}\nRcapt = {} ; Rcom = {}\nScore = {}'
    plt.title(title.format(instanceName, Rcapt, Rcom, score))
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinates')

    plt.show()


if __name__ == '__main__':
    
    fir, ax = plt.subplots()
    N = 100
    x = 20 * np.random.random(N)
    y = 9 * np.random.random(N)
    plt.scatter(x, y)

    solution = np.random.randint(0, 2, N)
    indexSelected = np.where(solution == 1)[0]
    plt.scatter(x[indexSelected], y[indexSelected], color='r')

    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if (solution[i] == 1) and (solution[j] == 1) and (d < 2):
                plt.plot([x[i], x[j]], [y[i], y[j]], c='g')

    for i in indexSelected:
        ax.add_patch(plt.Circle((x[i], y[i]), 2, color='orange', alpha=0.1))

    plt.show()
