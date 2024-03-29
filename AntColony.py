import numpy as np
import random as rd

from meta_VNS import parserInstance
from meta_VNS import constraints

#On choisit l'instance avec laquelle on veut travailler
Rcapt = 1
Rcom = 2
instanceName = '../Instances/captTRUNC90_10_10.dat'
print(instanceName)

Acapt, Acom, NeighCapt, NeighCom=parserInstance.parseData(instanceName, Rcapt, Rcom)

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
#    print("li",li)
    if len(li)>1500 :
        print("ABORT EVERYTHING",li)
        assert(False)
#    On regarde dans un premier tmeps tous les sommets vers lesquels on peut aller
#    if np.sum(Accessible)!=1 or np.sum(Accessible)!=0 :print("error proba",np.sum(Accessible))
#    Probabilité avec laquelle on va choisir le sommet sur lequel aller
#    print(OnVaOu(Accessible))
    assert(type(OnVaOu(Accessible))==int)
    li.append(OnVaOu(Accessible))
    return li
#    On en choisit un en proba en fonction de rho

def OnVaOu(l):
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

def Visibilite2(k,Sommets,NombreDInstance,p=0):
#    print("k",k,"Sommets",Sommets)
#    C'est cette fonction qu'on va modifier lorsqu'on voudra changer la probabilite de'allr d'un somet à un autre
    if not (type(k)==type(1)): print("probleme de type",k, type(k))
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
    
def Evaporation(x,a=0.6,b=0):
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
	return np.sum(solution)-1

def ScoreAlternatif(solution,NombreDInstance):
    return NombreDInstance-score(solution)

Pheromone=len(parserInstance.readInstance(instanceName))*[1]
    
sol=[]
def ColonyAnt(instanceName,Rcapt,Rcom):
	'''
	impmlement the ant colony heuristic
	'''
	# parse data	
	Acapt,Acom,NeighCapt,NeighCom = parserInstance.parseData(instanceName, Rcapt, Rcom)
	#Initialization
	NombreDInstance=len(parserInstance.readInstance(instanceName))
	assert(NombreDInstance==len(Acapt))
	global Pheromone
	
	#Iterations
	Stop=15#NombreDInstance
#    Stop permet d'arrêter un fourmi si elle n'a rien rapporté au bout de Stop tour
	NombreDeTour=1000
	# NombreDeTour est le nombre d'itération max qu'on s'est fixé
	LaSolutionConverge=0
	# LaSolutionConverge est incrémenté dès que la solution n'est pas changée
#    On a abandonné cette idée puisqu'on a besoin de mauvaises solution pour les phéromones
	CriteredArret=( NombreDeTour<0 and LaSolutionConverge>10)
	# Formule booleenne qui indique lorsqu on arrete les iterations pour renvoyer la derniere solution obtenue
	# CriteredArret = tant que on a un nombre de tour >0 et que la sol converge >10
	while not CriteredArret:
		global sol
		sol=[]
#		On a besoin de sol car sol est une liste de sommet, il faut savoir quel sommet on a mis en dernier, et Ant est une liste de booléen, pour se formaliser sur cce qu'on s'était dit initialement
		OnEstBloque=False
#        Booléen qui devient vrai dès qu'il faut changer de fourmi
		for i in range(int(4)):
#               Pour chaque cible i
			sol.append(i)
#                Les fourmis partent d'un point source à n'importe quel sommet en temps
			while not constraints.checkConstraints(np.array(SolutionToAnt(sol,NombreDInstance)), Acapt, Acom, NeighCom):
				if OnEstBloque : 
#					assert(False)
					break
				for iter in range(Stop):
					assert(type(sol[-1])==int)
					AccessibiliteDesCibles=MultList(Visibilite2(sol[-1],sol,NombreDInstance,p=0),Pheromone)
					j=np.sum(AccessibiliteDesCibles)
					if j==0:
						OnEstBloque=True
						break
					AccessibiliteDesCibles=[g * (1/j) for g in AccessibiliteDesCibles]
#                   C'est ici qu'on choisit la probabilité avec laquelle on se déplacae sur chaque somme
					sol=CreerUnChemin(sol,AccessibiliteDesCibles)
					assert(type(sol[-1])==int)
#                    On ajoute un nouveau sommet au chemin
			if constraints.checkConstraints(np.array(SolutionToAnt(sol,NombreDInstance)), Acapt, Acom, NeighCom):
			    return SolutionToAnt(sol,NombreDInstance)
        
Colonie=[]
EvolutionDesScores=[]
NombreDInstance=len(parserInstance.readInstance(instanceName))

def GenerationDeFourmi():
    global Pheromone
    NombreDeFourmi=50
    global Colonie
    global NombreDInstance
    #Colonie sera une liste de NombreDeFourmi fourmis de NombreDInstance éléments
    global EvolutionDesScores
    for fourmi in range(NombreDeFourmi):
        Colonie.append(ColonyAnt(instanceName,Rcapt,Rcom))
        EvolutionDesScores.append(score(Colonie[-1]))
    
    #    Maintenant on s'occupe de mettre en place des phéromones et de les faire s'évaporer
    
    for fourmi in range(NombreDeFourmi):
        assert(len(Pheromone)==len(Colonie[fourmi]))
        Pheromone=[x+y for (x,y) in zip(Pheromone,Colonie[fourmi])]
    Pheromone=Evaporation(Pheromone)
    
ScoreMin=[]
ScoreMoy=[]
for gen in range(10):    
    print("generation",gen)
    GenerationDeFourmi()
    ScoreMoy.append(np.mean(EvolutionDesScores[-NombreDInstance:len(EvolutionDesScores)]))
    ScoreMin.append(np.min(EvolutionDesScores[-NombreDInstance:len(EvolutionDesScores)]))

print(Colonie[EvolutionDesScores.index(min(EvolutionDesScores))])
print(score(Colonie[EvolutionDesScores.index(min(EvolutionDesScores))]))
print("ScoreMin",ScoreMin)
print("ScoreMoy",ScoreMoy)  

import matplotlib.pyplot as mp
mp.plot(ScoreMoy)
mp.plot(ScoreMin)

	
