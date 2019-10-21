'''
Recuit simulé basique permettant d'obtenir rapidement une première solution
de bonne qualité

On représente une solution par un vecteur de taille n (nombre de cibles):
    - sol[i] = 1 si cible 1 reçoit un capteur
    - sol[i] = 0 sinon
'''



def contrainteCapt(solution, Acapt):
    n = len(solution)
    assert(solution[0] == 1)
    
    violationCapt = np.ones(n, dtype=np.int)
    indexCapteurs = np.where(solution == 1)[0]
    for i in indexCapteurs:
        for j in range(n):
            if (violationCapt[j] == 1) and (Acapt[i, j] == 1):
                violationCapt[j] = 0

    return violationCapt

def contrainteCom(solution, Acom):
    n = len(solution)
    assert(solution[0] == 1)

    violationCom = np.ones(n, dtype=np.int)
    violationCom[0] = 0
    file = [0]

    while len(file) > 0:
        i = file[0]
        file = file[1:]
        for j in range(n):
            if (violationCom[j] == 1) and (Acom[i, j] == 1):
                violationCom[j] = 0
                file.append(j)

    return violationCom

