import numpy as np
import time
import multiprocessing as mp
import random as rd

from meta_VNS import parserInstance
from meta_VNS import constraints
from meta_VNS import PierreConstraints
from meta_VNS import heuristics

#On choisit l'instance avec laquelle on veut travailler
Rcapt = 1
Rcom = 2
instanceName = '../Instances/captTRUNC87_10_10.dat'

Acapt, Acom, NeighCapt, NeighCom=parserInstance.parseData(instanceName, Rcapt, Rcom)

def VNS(instanceName, Rcapt, Rcom):
    '''
    Implement VNS metaheuristic
    '''
    # parse data
    Acapt, Acom, NeighCapt, NeighCom = parserInstance.parseData(
        instanceName, Rcapt, Rcom)
    nNodes = Acapt.shape[0]

    # parameters

    # initialization
    solution = np.ones(nNodes, dtype=np.int)
    assert(constraints.checkConstraints(solution, Acapt, Acom, NeighCom))
    score = np.sum(solution)

    # iterations over neighborhoods
    neighborhoods = [greedyDelete, greedyPivot1, greedyPivot2]
    descent = True
    ind = 0
    while ind < len(neighborhoods):
        V = neighborhoods[ind]
        solution, descent = V(solution, Acapt, Acom, NeighCom)
        score = np.sum(solution)
        assert(constraints.checkConstraints(solution, Acapt, Acom, NeighCom))
        if descent:
            ind = 0
        else:
            ind += 1
    
    return solution, score

def SolutionToAnt(liste,NombreDInstance):
#    Cette fonction transforme une liste de sommets en une liste de booléen où l[i]=1 ssi i est dans liste
    res=NombreDInstance*[0]
    for elemnt in liste:
        res[elemnt]=1
    return res    

def AntToSolution(liste):
#    Donne une liste de booléen qui recense là où il y a des 1
    tmp=[]
    while np.sum(liste)>0:
        tmp.append(liste.index(1))
        liste[liste.index(1)]=0
    return tmp    

def CreerUnChemin(li,Accessible):
#    On part d'une liste li et on rajoute un sommet
    #    Accessible=Acom[DernierSommet] ?
#    Accessible est la liste des sommets tels que =0 ssi inaccessible, sinon =phéromone/distance
#    Dans cette fonction on ne rajoute pas la clause "on ne peut pas repasser par un sommet par lequel on est déjà passé" car il peut être nécessaire de passer plusieurs fois au même endroit
    DernierSommet=li[-1]
    print("li",li)
    if len(li)>1500 :
        return "ABORT EVERYTHING"
#    ON regarde dans un premier tmeps tous les sommets vers lesquels on peut aller
    if np.sum(Accessible)!=1 or np.sum(Accessible)!=0 :print("error proba",np.sum(Accessible))
    n=len(Accessible)
    p=rd.random()
#    Probabilité avec laquelle on va choisir le sommet sur lequel aller
    tmp=[Accessible[0]]
    li.append(OnVaOu(Accessible))
    return li
#    ON en choisit un en proba en fonction de rho

k=[0,0,0,0]

def OnVaOu(l):
    if not abs(sum(l)-1)<10**(-5) or abs(sum(l)-1)-1<10**(-5) : print("sum(l)",sum(l))
    assert(abs(sum(l)-1)<10**(-5) or abs(sum(l)-1)-1<10**(-5))
    p=rd.random()
    tp=[l[0]]
    if len(l)!=1:
        for i in range(1,len(l)):
            tp.append(tp[-1]+l[i])
    for i in range(len(l)):
#        il ne faut pas que tp s'incrémente tout de suite contrairement à k
        if p<tp[i]:
            return i
            break
        i+=1

def Visibilite1(k,Sommets):
#    C'est cette fonction qu'on va modifier lorsqu'on voudra changer la probabilite de'allr d'un somet à un autre
    if not (type(k)==type(1)): print("probleme de type")
#    Renvoie la liste relative au sommet k telle que : l[i]=0 ssi inateignable, l[i]=1/distance(i,k)
    global Acom
    LaOuOnPeutAller=Acom[k]
    n=len(LaOuOnPeutAller)
    for i in range(n):
        if LaOuOnPeutAller[i]!=0:
            if not i==k:
                LaOuOnPeutAller[i]=1/parserInstance.distance2(parserInstance.readInstance(instanceName)[k],parserInstance.readInstance(instanceName)[i])
            else:
                LaOuOnPeutAller[i]=0
    return LaOuOnPeutAller            

def Visibilite2(k,Sommets,NombreDInstance,p=0):
    print("k",k,"Sommets",Sommets)
#    C'est cette fonction qu'on va modifier lorsqu'on voudra changer la probabilite de'allr d'un somet à un autre
    if not (type(k)==type(1)): print("probleme de type")
#    Renvoie la liste relative au sommet k telle que : l[i]=0 ssi inateignable, l[i]=1 sinon et p ssi on est déjà passé là
    global Acom
    res=NombreDInstance*[0]
    LaOuOnPeutAller=NeighCom[k][1]
    for som in LaOuOnPeutAller:
           if not som in Sommets:
                res[som]=1
           else:
                res[som]=p
    return res            
    
def Rho(x,a=0.95,b=-0.05):
#    Lafocntoin qui évapore les phéromones
#    x est la liste des phéromones
#    Cette fonction est un fonction affine ax+b décroissante qu'il faut modifier si besoin
    def foo(z):
        return a*z+b
    tmp=[foo(i) for i in x]
    return tmp

def MultList(l1,l2):
    tmp=[]
    n=len(l1)        
    if n!=len(l2):
        print("probleme de longeur")
    for i in range(n):
        tmp.append(l1[i]*l2[i])  
    return tmp    

def score(solution):
	return np.sum(solution)

Pheromone=len(parserInstance.readInstance(instanceName))*[1]
    
sol=[]
def ColonyAnt(instanceName,Rcapt,Rcom):
	'''
	impmlement the ant colony heuristic
	'''
	NombreDInstance=len(parserInstance.readInstance(instanceName))
	# parse data	
	Acapt, Acom,NeighCapt, NeighCom = parserInstance.parseData(instanceName, Rcapt, Rcom)
	#Initialization
	global Pheromone   
	NombreDInstance=len(parserInstance.readInstance(instanceName))
#    Cette soluition est trouvée au préalable par une autre heuristique
#	print(solution)
#	assert(constraints.checkConstraints(solution, Acapt, Acom, NeighCom))
	assert(NombreDInstance==len(Acapt))
	global Pheromone
	
	#Iterations
	Stop=15#NombreDInstance
#    Stop permet d'arrêter un fourmi si elle n'a rien rapporté au bout de Stop tour
	NombreDeTour=1000
	# NombreDeTour est le nombre d'itération max qu'on s'est fixé
	LaSolutionConverge=0
	# LaSolutionConverge est incrémenté dès que la solution n'est pas changée
	CriteredArret=( NombreDeTour<0 and LaSolutionConverge>10)
	# Formule booleenne qui indique lorsqu on arrete les iterations pour renvoyer la derniere solution obtenue
	# CriteredArret = tant que on a un nombre de tour >0 et que la sol converge >10
	AntColony=[]
#            AntColony vaudra une liste de longueur n et telle que AntColony[i]= le ieme chemin proposé par la ieme fourmi
	Ant=NombreDInstance*[0]
	while not CriteredArret:
		# Construire la solution
		Ant=NombreDInstance*[0]	
		global sol
		sol=[]
#		On a besoin de sol car sol est une liste de sommet, il faut savoir quel sommet on a mis en dernier, et Ant est une liste de booléen, pour se formaliser sur cce qu'on s'était dit initialement
		OnEstBloque=False
		print("dfghj,jgbf")
#        Booléen qui devient vrai dès qu'il faut changer de fourmi
		for i in range(NombreDInstance):
#               Pour chaque cible i
			print("Ant",Ant,i)
			Ant=np.zeros(NombreDInstance) #liste de 0, on réinitialise Ant
			sol.append(i)
			print("sol",sol)
#                Les fourmis partent d'un point source à n'importe quel sommet en temps
			if constraints.checkConstraints(SolutionToAnt(sol,NombreDInstance), Acapt, Acom, NeighCom):
				print("ZAAAAAC")
				return "ZAAAAC"
			while not constraints.checkConstraints(SolutionToAnt(sol,NombreDInstance), Acapt, Acom, NeighCom):
				if OnEstBloque : 
					break
				for iter in range(Stop):
					print("Visibilite2",Visibilite2(sol[-1],sol,NombreDInstance,p=0))
					AccessibiliteDesCibles=MultList(Visibilite2(sol[-1],sol,NombreDInstance,p=0),Pheromone)
					j=np.sum(AccessibiliteDesCibles)
					if j==0:
						print("on a déjà les cibles")
						OnEstBloque=True
						break
					AccessibiliteDesCibles=[g * (1/j) for g in AccessibiliteDesCibles]
					print("AccessibiliteDesCibles",AccessibiliteDesCibles)
#                   C'est ici qu'on choisit la probabilité avec laquelle on se déplacae sur chaque somme
					print("iter",iter)
					sol=CreerUnChemin(sol,AccessibiliteDesCibles)
#                    On ajoute un nouveau sommet au chemin
			for sommet in sol:
				Ant[sommet]=1
#            On convertit une liste de sommet en une lise de booleen de longueur n
                    
	AntColony.append(Ant)
	ListeScore=[]
	for ant in AntColony:
		ListeScore.append(score(ant))
	mintmp=0
	print("AntColony",AntColony)
	print("baplou",ListeScore,len((ListeScore)))
	for k in range(NombreDInstance):
		ListeScorek=ListeScore[k]
		if ListeScorek<mintmp:
			mintmp=k
	return ListeScorek, AntColony[k]
#    Ici on retourne la meilleure solution et son score    
        
        # Appliquer une recherche locale
#		
#		En fonction des phéromones qui ont été ajoutées on crée une nouvelle solution et on la compare par rapport à ce qu'on a déjà
#		On réajuste la solution pour voir si on peut la modifier s'il y a des trucs qui sont inutiles (sommets superflus, cycles inutiles)
#		
#        on fera ça dans un second temps hein
        # Mise à jour des phéromones 
		 
#        Pheromone=Rho(Pheromone)

#		On regarde comparativement le taux de phéromones des sommets et si l'un ou l'autre a un taux particulièrement élevé on agit comme suit 
#		On fait évaporer des phéromones de manière constante sauf si :
#		 
#			#deux possibilités si sommet très interessant :
#			on enlève des possibilités pour vérifier si on est pas bloqué dans un optimum local
#				On fait quelques recherches en fonction de ça
#				c'est mieux,-> on continue en enlevant les phéromones du sommet/on empêche son accès
#				C'est moins bien -> on le force dans chaque solution comme hypothèse pour améliorer le temeps de calcul
#
#        on fera ça dans un second temps hein



NombreDInstance=len(parserInstance.readInstance(instanceName))
print(ColonyAnt(instanceName,Rcapt,Rcom))	
  
if len(Acom)!=n:print("error longueur com")
if Acom[0][0]==1:
    NeighCom=Acom-np.eye(n)
else:
    NeighCom=Acom
      
if len(Acapt)!=n:print("error longueur capt")
if Acapt[0][0]==1:
    NeighCom=Acapt-np.eye(n)
else:
    NeighCom=Acapt
'''
Le critèr d'arrêt est : nombre max d'itération; ça fait longtemps qu'on a pas amélioré la sol; temps en secondes
On part dune solution trouvée au préalable et on l'améliore
On peut booster l'évaporation de phéromones pour sortir de minima locaux
Pour réajuster le temps de calcul on peut forcer l'existence de sommets dans la solution, point charnière
Les fourmis partent des cibles pour aller au puit. -> disparaissent une fois qu'elles se rejoigent
Une fois les solutions formées à la fin du parcours des fourmis faire en sorte de réajuster la solution obtenue, enlever les cycles jusqu'à ce qu'on ait un arbre
	
'''	
	
