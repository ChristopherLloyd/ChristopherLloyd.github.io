## transformation images


## bibliotheques

from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## génération aléatoire permutations


def alea_permut(n,nbshuffle):
    A=[]
    for k in range(n):
        A.append(k)
    for k in range(nbshuffle):
        np.random.shuffle(A)
    return(np.array(A))

## création images bitmap


    ## Image standard

def img_standard(n,p):
    X=np.ones((n,p))
    i=0
    for k in range(n):
        for l in range(p):
            X[k,l]=i
            i+=1
    return(X)

    ## affichage d'une image

def img_affiche(T):
    plt.imshow(T,cmap=plt.cm.gray,interpolation='nearest')
    plt.show()

## action d'une permutatio de type vecteur de longueur n*p sur un tableau de taille n,p

def agir(perm,T):
    A=T.copy()
    n,p=np.shape(T)
    for i in range(n):
        for j in range(p):
            elem=T[i,j]
            alpha=perm[i*p+j]
            jbis=alpha%p
            ibis=(alpha-jbis)/p
            A[ibis,jbis]=elem
    return(A)

def transform(perm,T,itere):
    A=T.copy()
    for k in range(itere):
        A=agir(perm,A)
    return(A)

## affichage d'une image avant après transformation

def img_affiche_av_ap(T,perm,itere):
    A=transform(perm,T,itere)
    plt.subplot(121)
    plt.imshow(T,cmap=plt.cm.gray,interpolation='nearest')
    plt.subplot(122)
    plt.imshow(A,cmap=plt.cm.gray,interpolation='nearest')
    plt.show()

## choix d'une permutation aléaoire et itération jusqu'à revenir sur identité : détermination de l'ordre

def transform_perm_lst(perm,E):
    R=E.copy()
    for k in range(0,len(E)):
        R[k]=(E[perm[k]])
    return(R)


def ordre_perm_img(perm):
    l=len(perm)
    L=[k for k in range(0,l)]
    E=L.copy()
    itere=1
    E=transform_perm_lst(perm,E)
    while E!=L:
        E=transform_perm_lst(perm,E)
        itere+=1
    return(itere)

## Affichage image standard, permutation, ordre pour 10x10

def img_affiche_av_ap_ord(T,perm1,perm2,perm3,itere):
    A=transform(perm1,T,itere)
    plt.figure(1)
    plt.subplot(221)
    plt.axis('off')
    plt.imshow(T,cmap=plt.cm.gray,interpolation='nearest')
    #
    plt.subplot(222)
    plt.axis('off')
    plt.title("Transfo. 1 : {} itérations".format( ordre_perm_img(perm1) ), fontsize=12)
    plt.imshow(A,cmap=plt.cm.gray,interpolation='nearest')
    #
    plt.subplot(223)
    plt.axis('off')
    plt.title("Transfo. 2 : {} itérations".format( ordre_perm_img(perm2) ), fontsize=12)
    plt.imshow(A,cmap=plt.cm.gray,interpolation='nearest')
    #
    plt.subplot(224)
    plt.axis('off')
    plt.title("Transfo. 3 : {} itérations".format( ordre_perm_img(perm3) ), fontsize=12)
    plt.imshow(A,cmap=plt.cm.gray,interpolation='nearest')
    #
    plt.tight_layout()
    plt.show()

# img_affiche_av_ap_ord(img_standard(10,10), alea_permut(100,5),alea_permut(100,5),alea_permut(100,5),1)







