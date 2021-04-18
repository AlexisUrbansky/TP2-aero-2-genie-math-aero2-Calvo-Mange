#################################### Propriété #############################################

    #Tp2 Génie mathématiques, méthode de la décomposition de Cholesky et autres
    #Louis Calvo & Alexis Mangé
    #fichier 2/2

################################## Librairies importées ####################################

import numpy as np 
import time as t
from math import *
import matplotlib.pyplot as plt
from Fonction import *    #Fichier python contenant les fonctions utilisées dans notre programme
 




###########################################################################################
#                                   Méthode Cholesky                                      #
###########################################################################################

ERREUR_C=list()
Y_C=list()
def ResolCholesky(A,B):
    '''
    Cette fonction regroupe les fonctions nécéssaires pour effectuer la méthode de Cholesky pour trouver les solution d'un système à 
    l'aide de matrices.
    Elle retourne le temps de calcul nécéssaire pour effectuer cette méthode.
    '''

    t1 = t.time()
    nb_ligne, nb_colonnes = np.shape(A)
    #définition de la matrice L
    L , LT, verif = Cholesky(A)
    
    if verif == 0:
        print(' On ne peut pas appliquer Cholesky')
        t2=t.time()
        tps_calcul = t2-t1
        Y_C.append(tps_calcul) 
    else:
        Y = ResolutionSystTriginf(L,B)
        X = ResolutionSystTriSup(LT,Y)
        erreur = np.linalg.norm(A.dot(X) - np.ravel(B))
        ERREUR_C.append(erreur)
        t2=t.time()
        tps_calcul = t2-t1
        Y_C.append(tps_calcul)
    return tps_calcul


###########################################################################################
#                                     Méthode LU                                          #
###########################################################################################

ERREUR_LU=list()
Y_LU=list()
def resolution_LU(A,B):

    '''
    Cette fonction regroupe les fonctions nécéssaires pour effectuer la méthode LU pour trouver les solution d'un système à 
    l'aide de matrices.
    Elle retourne le temps de calcul nécéssaire pour effectuer cette méthode.
    '''

    T1 = t.time()
    U,L = DecompositionLU(A)
    #### résolution de LY=B ####
    Y = ResolutionSystTriginf(L,B)
    #### Résolution UX=Y ####
    X = ResolutionSystTriSup(U,Y)
    X = np.asarray(X)
    erreur = np.linalg.norm(A.dot(X) - np.ravel(B))
    
    ERREUR_LU.append(erreur)
    T2 = t.time()
    tps_calcul = T2-T1
    Y_LU.append(tps_calcul)
    return X

###########################################################################################
#                              Méthode linalg.solve                                       #
###########################################################################################

Y_linalg = list()
ERREUR_Linalg=list()
def linalg_solve(A,B):

    '''
    Cette fonction permet de résoudre un système AX=B par la méthode linalg.solve présente de base sur python
    Elle retourne le temps de calcul nécéssaire pour effectuer cette méthode.
    '''

    T1=t.time()
    X = np.linalg.solve(A,B)
    T2=t.time()
    tps_calcul = T2-T1 
    Y_linalg.append(tps_calcul)
    erreur = np.linalg.norm(A.dot(X) - np.ravel(B))
    ERREUR_Linalg.append(erreur)
    return tps_calcul


###########################################################################################
#                              Méthode linalg.cholesky                                    #
###########################################################################################
Y_Linalgcholesky =list()
ERREUR_LinalgCholesky = list()
def linalg_cholesky(A,B):

    '''
    Cette fonction permet de résoudre un système AX=B par la méthode linalg.Cholesky présente de base sur python
    Elle retourne le temps de calcul nécéssaire pour effectuer cette méthode.
    '''

    T1=t.time()
    L  = np.linalg.cholesky(A)
    LT = np.transpose(L)
    #ResolCholesky(L,LT)
    Y = ResolutionSystTriginf(L,B)
    X = ResolutionSystTriSup(LT,Y)
    erreur = np.linalg.norm(A.dot(X) - np.ravel(B))
    T2=t.time()
    tps_calcul = T2-T1 
    Y_Linalgcholesky.append(tps_calcul)
    ERREUR_LinalgCholesky.append(erreur)
    return tps_calcul


###########################################################################################
#                              Création des différents graphiques                         #
###########################################################################################
X = list()  
for i in range(100, 1000 , 100):
    X.append(i)
    A, B = random_mat(i)
    linalg_solve(A,B)
    ResolCholesky(A,B)
    linalg_cholesky(A,B)
    resolution_LU(A,B)




##### Courbe temps de fonction de la taille de la matrice #####
plt.ylabel("temps en seconde")
plt.xlabel(" n ")
plt.title("Courbe du temps d'exécution en fonction de la taille n de la matrice")
plt.plot(X,Y_LU, label='LU')
plt.plot(X,Y_C, label='Cholesky')
plt.plot(X,Y_Linalgcholesky, label='linalg.Cholesky')
plt.plot(X,Y_linalg, label='linalg.solve')
plt.grid()
plt.legend()
plt.show()

###### Courbe Logarithmique ######
plt.loglog(X,Y_LU, label='courbe Ln de LU')
plt.loglog(X,Y_C, label='courbe Ln de Cholesky')
plt.loglog(X,Y_Linalgcholesky, label='courbe Ln de linlag.Cholesky')
#plt.loglog(X,Y_linalg, label='Courbe Ln de linalg.solve')
plt.grid()
plt.ylabel("temps en seconde")
plt.xlabel(" n ")
plt.title("Courbe logarithmique du temps en fonction de la taille de la matrice. ")
plt.legend()
plt.show()

###### Courbe semi-Logarithmique ######
plt.semilogy(X,Y_LU, label='courbe Ln de LU')
plt.semilogy(X,Y_C, label='courbe Ln de Cholesky')
plt.semilogy(X,Y_Linalgcholesky, label='courbe Ln de linlag.Cholesky')
#plt.loglog(X,Y_linalg, label='Courbe Ln de linalg.solve')
plt.grid()
plt.ylabel("temps en seconde")
plt.xlabel(" n ")
plt.title("Courbe semilogarithmique du temps en fonction de la taille de la matrice. ")
plt.legend()
plt.show()


##### Courbe des erreurs #####
plt.plot(X,ERREUR_LU, label='LU')
plt.plot(X,ERREUR_C, label='Cholesky')
plt.plot(X,ERREUR_LinalgCholesky, label='linalg.Cholesky')
plt.plot(X,ERREUR_Linalg, label='linalg.solve')
plt.grid()
plt.legend()
plt.ylabel("Erreur ||=AX-B||")
plt.xlabel(" n ")
plt.title("Courbe de l'erreur en fonction de la taille n de la matrice")
   
plt.show()

###### Courbe log des erreurs  #####
plt.semilogy(X,ERREUR_LU, label='LU')
plt.semilogy(X,ERREUR_C, label='Cholesky')
plt.semilogy(X,ERREUR_LinalgCholesky, label='linalg.Cholesky')
plt.semilogy(X,ERREUR_Linalg, label='linalg.solve')
plt.legend()
plt.grid()
plt.ylabel("Erreur ||=AX-B||")
plt.xlabel(" n ")
plt.title("Courbe semilogarithmique de l'erreur en fonction de la taille n de la matrice")
plt.show()

