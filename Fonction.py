#################################### Propriété #############################################

    #Tp2 Génie mathématiques, méthode de la décomposition de Cholesky et autres
    #Louis Calvo & Alexis Mangé
    #fichier 1/2

################################## Librairies importées ####################################

import numpy as np 
import time as t
from math import *
import matplotlib.pyplot as plt

################################## Fonctions réutilisées ###################################

def Cholesky(A):
    '''
    Cette fonction permet d'obtenir les matrices L et L_T tel que A = L*L_T.
    Il s'agit de la décomposition de Cholesky qui nous permettra de résoudre des systèmes de la formes AX=B
    '''

    nb_ligne, nb_colonne = np.shape(A)
    L = np.zeros([nb_ligne, nb_ligne])
    try:
        for k in range(nb_ligne):
            for i in range(k, nb_ligne):
                if i == k:
                    L[i, k] = np.sqrt(A[i, k] - np.sum(L[i, :k]**2))
                else:
                    L[i, k] = (A[i, k] - np.sum(L[i, :k]*L[k, :k])) / L[k, k]
        L_T = np.transpose(L)
        verif = 1
        return L, L_T, verif
    except:
        L_T = np.transpose(L)
        verif=0
        return L, L_T, verif

def random_mat(n):
    '''
    Cette fonction permet de générer deux matrices aléatoire A et B.
    Elle retourne ces deux matrices.
    '''

    A = np.random.rand(n, n)
    B = np.random.rand(n)
    A = A.dot(np.transpose(A))

    return A, B


def ResolutionSystTriginf(L,B):

    '''
    Cette fonction permet de résoudre un système à partir d'une matrice triangulaire inférieur (matrice L).
    Elle prend en argument L et B et retourne les solutions sous forme d'une matrice.
    '''

    Taug=np.c_[L,B]
    nb_ligne, nb_colonnes = np.shape(Taug)
    comptage=0
    Y = []
    v = Taug[0][nb_colonnes-1]/Taug[0][0]   # valeur de Xn qui nous permet de résoudre le reste du système
    Y.append(v)
    for i in range(1,nb_ligne):
        n = Taug[i][nb_colonnes-1]
        for j in range(0,nb_colonnes-1):
            if (j < i):
                n = n - Taug[i , j]*Y[comptage]  #on bascule toutes les valeurs de l'autre coté sauf le X qu'on cherche à calculer.
                comptage+=1
        n = n / Taug[i,i]  #on divise et on obtient la valeur d'une des inconnu du système
        v=n       
        comptage = 0
        Y.append(v)
    Y = np.asarray(Y)
    Y = Y.T
    return Y


def ResolutionSystTriSup(U,Y):

    '''
    Cette fonction permet de résoudre un système à partir d'une matrice triangulaire supérieur (matrice U)..
    Elle est légèrement différentes de ResolutionSystTriSup() car elle prend en argument U et une matrice Y contenant les solutions de LY=B
    '''

    global nb_ligne, nb_colonnes
    comptage=0
    Taug = np.c_[U,Y]
    nb_ligne, nb_colonnes = np.shape(Taug)
    liste_solutions = []
    v = Taug[nb_ligne-1][nb_colonnes-1]/Taug[nb_ligne-1][nb_colonnes-2]   # valeur de Xn qui nous permet de résoudre le reste du système
    liste_solutions.append(v)
    for i in range(2,nb_ligne+1):
        n = Taug[nb_ligne-i][nb_colonnes-1]
        for j in range(2,nb_colonnes+1):
            if ((nb_colonnes - j) > (nb_ligne - i)):
                #if abs(Taug[nb_ligne-i , nb_colonnes-j] - 0) <= 10**-10:
                n = n - Taug[nb_ligne-i , nb_colonnes-j]*liste_solutions[comptage]  #on bascule toutes les valeurs de l'autre coté sauf le X qu'on cherche à calculer.
                comptage+=1
        n = n / Taug[nb_ligne-i, nb_colonnes-i-1]  #on divise et on obtient la valeur d'une des inconnu du système
        v=n
        comptage = 0
        liste_solutions.append(v)
    
    liste_solutions.reverse()
    return liste_solutions


def DecompositionLU(A):


    '''
    Grace à cette fonction on peut obtenir la décomposition LU d'une matrice carrée A tel que A=LU.
    '''
    
    global nb_ligne, nb_colonnes
    nb_ligne, nb_colonnes = np.shape(A)
    #création d'une matrice L vièrge
    n = nb_ligne


    L = np.zeros_like(A)
    U = A.copy()
    #Set up pour faire une réduction de la matrice carrée A et réduction.
    #à chaque création du pivot il est ajouté à la matrice L
    pivot = 0

    for k in range(0,nb_colonnes):
        for i in range(k+1,nb_ligne):
            pivot = (U[i,k])/(U[k,k])
            L[i,i] = 1
            L[i,k] = pivot
            U[i,:] = U[i,:] - pivot * U[k,:] 
    L[0,0] = 1

    #print des résultats pour les vérifier, la matrice A est devenue la matrice U
    

    '''
    print("la fonction U : \n")
    print(U)
    print("la fonction L : \n")
    print(L)
    '''
    return (U,L)