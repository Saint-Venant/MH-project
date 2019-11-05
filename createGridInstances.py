'''
Create grid instances with format:
- 10 x 10
- 15 x 15
- 20 x 20
- 25 x 25
- 30 x 30
- 40 x 40
'''



def createGrid(n):
    fileName = 'Instances\captGRID{}_{}_{}.dat'.format(n**2, n, n)
    row = ' {}  {} {}\n'
    with open(fileName, 'w') as f:
        for i in range(n):
            for j in range(n):
                f.write(row.format(i*n + j, i, j))

if __name__ == '__main__':
    listDim = [10, 15, 20, 25, 30, 40]

    for n in listDim:
        createGrid(n)
