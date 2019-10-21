# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:52:04 2019

@author: Pierre
"""

#
#fichier = open( "captANOR225_9_20.dat", "r") # ouverture en lecture
#txt = fichier.read()
#fichier.close()
#print(txt)
#print(type(txt))
#b = []
#line = txt.split("\n")[1] # 1ere ligne de donnees
#line = line.split(";") # le separateur utilise a l'ecriture
##b.append(float(line[0]))
##b.append(int(line[1]))
##print("b:", b)
#print(line)

import numpy as np
from math import sqrt

def distance(a,b):
#    Calcule la distance de deux points dans le plan
    if len(a)!=2 and len(b)!=2:
        print("Erruer d'argument")
    if type(a[0])!="int" or type(a[1])!="int" or type(b[0])!="int" or type(b[1])!="int" :
        print("erreur de type")
    else:
        try:
            return sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
        except:
            print("a",a)
            print("b",b)


def distance2(a,b):
#    Calcule la distance de deux points dans le plan
#    Cette fonction est adaptée pour des listes dans laquelle le premier terme ne représente aps les coordonnées
    if len(a)!=3 and len(b)!=3:
        print("Erruer d'argument")
    else:
        try:
            return sqrt((a[1]-b[1])**2+(a[2]-b[2])**2)
        except:
            print("a",a)
            print("b",b)

            
liste = []
for line in open(instanceName):
    try:
        tmp = [float(x) for x in line.split() if x]
        if tmp:
            liste.append(tmp)
    except:
        pass
n=len(liste)
#Ici on lit le fichier.dat et on le transforme en un liste de n élément où chaque élément est une liste de longueur 3

Acapt=np.zeros((n,n))    

def MakeAcapt(l=None):
#    Cette fonction fait la matrice Acapt telle que Acapt[i,j]=1 ssi dist(i,j)<=Rcapt pour une liste donnée
    global liste
    global Acapt
    if l==None:
        l= liste
    n = len(l)
    Acapt=np.zeros((n,n))    
    for i in range(n):
        for j in range(n):
            if distance2(l[i],l[j])<=Rcapt:
#                Initialement la matrice vaut 0 partout et on teste pour tous les couples de sommets si leur distance est inf à Rcapt
#                print(  distance2(liste[i],liste[j]),i,j)  
                Acapt[i,j]=1
    Acapt=Acapt-np.eye(n)
    print(Acapt)
#   On enlève le fait que on puisse se capter soi même pour une meilleure complexité
                        
    
#MakeAcapt()    

Acom=np.zeros((n,n))    

def MakeAcom(l=None):
#    Cette fonction fait la matrice Acapt telle que Acapt[i,j]=1 ssi dist(i,j)<=Rcapt pour une liste donnée
    global liste
    global Acom
    if l==None:
        l= liste
    n = len(l)
    Acom=np.zeros((n,n))    
    for i in range(n):
        for j in range(n):
            if distance2(l[i],l[j])<=Rcom:
#                Initialement la matrice vaut 0 partout et on teste pour tous les couples de sommets si leur distance est inf à Rcapt
#                print(  distance2(liste[i],liste[j]),i,j)  
                Acom[i,j]=1
    Acom=Acom-np.eye(n)
    print(Acom)
#   On enlève le fait que on puisse se capter soi même pour une meilleure complexité


def contraintecapt(lis):
#    Cette fonction fait partie de la fonction qui vérifie que la solution qui nous est donnée est bien réalisable vis à viq de cette contrainte, c'est à dire si on peut accéder à tous les sommets
    listemarque=lis
    n=len(lis)
    for k in range(n):
        if distance2(listemarque[k],[0,0,0])<=Rcapt:
            listemarque[k][0]='X'
#    Initialisation, on marque les sommets qui peuvent être captés par le puits
    OnPeutContinuer=True        
    if OnPeutContinuer:
#        Tant qu'on peut continuer
        OnPeutContinuer=False
        print("n",n)
        for g in range(n):
            print(listemarque[g][0])
            if listemarque[g][0]=='X' :
#                Pour chaque sommet marqué
#                print("ah")
                for l in range(n):
#                    print("g","l",g,l)
                    if Acapt[g][l]==1:
#                    Si on peut trouver un sommet marqué auquel on peut accéder 
#                        print("oh")
                        listemarque[l][0]='X'
                        OnPeutContinuer=True
#                        Si on peut se déplacer jusqu'à un autre sommet
    res=True
    print(listemarque)
    for i in listemarque:
        if i[0]!='X':
            res=False
        if not res :
            return res
    return res
#On teste pour voir si tous les sommets du graphe sont accessibles    

def contraintecom(lis):
#    Cette fonction fait partie de la fonction qui vérifie que la solution qui nous est donnée est bien réalisable vis à viq de cette contrainte, c'est à dire si on peut accéder à tous les sommets
    listemarque=lis
    n=len(lis)
    for k in range(n):
        if distance2(listemarque[k],[0,0,0])<=Rcom:
            listemarque[k][0]='X'
#    Initialisation, on marque les sommets qui peuvent être captés par le puits
    OnPeutContinuer=True        
    if OnPeutContinuer:
#        Tant qu'on peut continuer
        OnPeutContinuer=False
        print("n",n)
        for g in range(n):
            print(listemarque[g][0])
            if listemarque[g][0]=='X' :
#                Pour chaque sommet marqué
#                print("ah")
                for l in range(n):
#                    print("g","l",g,l)
                    if Acom[g][l]==1:
#                    Si on peut trouver un sommet marqué auquel on peut accéder 
#                        print("oh")
                        listemarque[l][0]='X'
                        OnPeutContinuer=True
#                        Si on peut se déplacer jusqu'à un autre sommet
    res=True
    print(listemarque)
    for i in listemarque:
        if i[0]!='X':
            res=False
        if not res :
            return res
    return res
#On teste pour voir si tous les sommets du graphe sont accessibles      

def longueurSolution(l):
#    Cette fonction donne la longueur de la solution
    if type(l)==type([0]):
        return len(l)
    else : print("errortype",type(l))
                

if __name__ == '__main__': 
    exemple1=   [[0,0,0],
                 [1,1,0],
                 [2,1,1],
                 [3,1,2],
                 [4,0,-1],
                 [5,-1,-1]]

    MakeAcom(exemple1)
    contraintecom(exemple1)

