## Bibliotheques

from math import *
import cmath as c

import numpy as np
import numpy.polynomial.polynomial as poly

import scipy as sc
import scipy.sparse.csgraph as g
import scipy.integrate as quad
import scipy.linalg as sla
#import scipy.optimize as opt

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

### Reference Numpy

## 1) tolist, .shape, .reshape(l,c) , fromfunction

a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b=a.tolist()                                        #a.tolist() a.toarray()
t=type(a)                                           #type de a
d=a.dtype                                           #type des objets dans a
f=np.array(a, dtype=complex)                        #declaration du type
g=a.astype(complex)                                 #copie avec conversion de types : .astype(complex)

n,p = np.shape(a)                                   #np.shape(a)
r=a.real                                            # .real, .imag, .conj()
i=a.imag                                            # np.abs(), np.angle()
cj=a.conj()
rho=np.abs(a)
theta=np.angle(a)
k=a+np.fromfunction(lambda x,y : x+y*1j,(3,4))      #np.fromfunction

## 2) v[debut : fin : pas] ; slicing = creation de views (pointeurs) sur les sous tableaux ; pas de copie
v=np.array(range(0,160,10))
t=np.array([[10*i+j for j in range(8)] for i in range(5)])
m=t.astype(complex)
m[::2,::2] += np.array([[i+j*1j for j in range(4)] for i in range(3)])

b=t[1::2,2::5]
b.fill(-1)
'''comme t est la base de b, toute modification de l'un agit sur l'autre : 
b ne fait que pointer vers un sous tableau du tableau pointe par b'''


## Fancy indexing !! : a[indices] (renvoie le tableau indices evalue par a) ; realise des copies
a = np.arange(0,100,10) #np.arange = np.array([k for k in range()])

''' FANCY INDEXING/WRITTING '''
indices = np.array([[5,2,1],[3,8,7]])
b = a[indices] #remplace dans le moul indices, les i par a[i]: la forme en sortie est celle de indices
l = a[[5,3,1,0,7,5]] #a est 'évalué' en les indices
a[[2,3,5,7]]=(222,333,555,777) #fancy indexing en mode ecriture (on aurait aussi u mettre des [])
b += b #ne modifie pas a car le fancy indexing cree une copie

    ## Fancy bidimensionnel, N-dimensionnel : FILTRE SUR LES INDICES

'''ENCORE PLUS PUISSANT !!!'''

c = (np.arange(12)**2)   #les carres de 0 a 11 
c.resize(3,4)            # en format (3,4)
I = np.fromfunction(lambda i,j : (i**2+j)%3, (5,5)).astype(int) #Attention, les indices doivent
J = np.fromfunction(lambda i,j : (j**3+2*i)%4, (5,5)).astype(int) #etre de type int
L=[I,J]

#on a deux possibilites pour le fancy indexing : 

extraction1 = c[I,J]
extraction2 = c[L]


        ## np.place(a, conditions,b) FILTRE SUR LES VALEURS
'''np.place(a,conditions,b) : ecrit les valeurs de b dans les positions de a verifiant les conditions'''
np.place(a,np.mod(a,3)==0,0) #remplace les valeurs idoines par 0


## 3) Redimensionnement : a.resahpe(dim) ; a.shape = (n1,n2,n3,...) ; np.resize(a,dim)
a = np.array(range(12))

'''RESIZE'''

a.reshape((3,4)) #SUR PLACE
c=np.resize(a,(2,3,5)) #NOUVEAU tableau et REPETE ou ARRETE l'ecriture si dimensions INCOHERENTES

cinq = np.resize(5,(2,3))
construction = np.resize([k for k in range(5)], (3,5))


'''a.flatten('F'), a.flat'''
a.flatten() #applatit en 1 ligne
a.flatten('F') #lis les colonnes d'abbord
x=sum(x for x in a.flat) #a.flat renvoie l'iterteur des elements de a


## utilitaires
t=a.T                       #transpose
np.delete(a,[2,0],axis=0)   #delete(a,indices,axis=k)
a2=np.append(a,a,axis=1)    #append "along axis k"
np.eye(4,5, k=1)            # lingnes, colonnes, k = decalge des 1 au dessus
m=np.diag([1,2,3,4,5])      #cree matrice diagonale
d=np.diag(m)                #extrait une diagonale
l=np.tril(m)                #tril
u=np.triu(m)                #triu
X=np.arange(5,31,2)         #arange
Y=np.linspace(5,31,11)      #linspace
v=np.vander(range(1,7))     #matrice de vandermonde


## fromfunction


def memoise(func):
    cache={}
    def wrapper(*args):
        if args not in cache:
            cache[args]=func(*args)
        return cache[args]
    return wrapper

@memoise
def combi(i,j):
    if i>j:
        return 0
    if i==j or i==0:
        return 1
    else:
        return combi(i,j-1)+combi(i-1,j-1)

t1 = np.fromfunction(lambda i,j : 1/(1+i+j),(5,5))
t2 = np.array([[combi(i,j) for j in range(5)] for i in range(5)])

## Fonctions universelles : ufunc : qui peuvent s'appliquer terme a terme aux tableaux

p=np.power(t2,t1) #terme a terme
x=10*p//(1+t2)
r=np.mod(t2,t1)
np.add(t1,2*t2,t1) #ajoute t2 à t1 et modifie t1 (dernier arg est le recepteur et conserve son datatype) sur place

## VECTORISATION DES FONCTIONS

def f(x):
    return (x+sin(x))/(x+cos(x))
    
'''np.vectorise'''
vf = np.vectorize(f)
t3 = vf(t1) # vf s'applique terme a terme

def g(x,y):
    return 0 if x<y else y

vg = np.vectorize(g)
t4 = vg(t1,t3)
t5 = vg(5,[[1,2,3],[4,5,6],[7,8,9]]) # Python s'adapte

''' np.apply_along_axis(func, axis, tab) '''
sumcol=np.apply_along_axis(sum, 0, t4)
sumlin=np.apply_along_axis(sum, 1, t4)

## Operations logiques, ensemblistes, comparaison, tri

a = np.array([True, False, False, True, True])
b = np.array([False, False, True, True, True])

n = np.logical_not(a)
u = np.logical_xor(a,b)

bool1 = np.array_equal(a,b)
bool2 = np.greater(a,b)
bool3 = np.allclose(a,b,1e-03) #Pour comprendre les erreurs

s=np.sort(a) #copie triee
a.sort() #sur place
u = np.unique(a) #unique :TRI avec suppression des doublons


#operations ensemblistes 1d : union1d, intersect1d, setdiff1d, setxor1d (donc sur les lignes)
u = np.union1d(a,b)
i = np.intersect1d(a,b)
d = np.setdiff1d(a,b)
delta = np.setxor1d(a,b)


## Comparaisons, tests: np.any, np.all, np.argwhere, np.where

a = np.array([[i<0.3*j for j in range(10)] for i in range(10)])
b= np.array([[2*i<j**2 for j in range(10)] for i in range(10)])

# any, all
t=np.any(a)
f=np.all(a)
d=np.all(c<5)
bvect=np.all(c<10, axis=0)

'''argwhere, np.where'''
mat = np.argwhere(c<2) #renvoie le tableau des coordonnees correspondant aux emplacements adequats
tab = np.where(c<2) #renvoie le tuple des incices selon chaque axes
choose = np.where(c<2, 3+c, False) #False est convertit en 0, ajoute 3 si <2, sinon met 0


## sommes produits differences

# cumsum, cumprod
a.sum()
a.prod()
a.cumsum()
a.cumprod()

## Algebre lineaire : utiliser scipy.linalg est plus efficace !!
k = np.resize([k for k in range(5)], (4,4))
K = np.dot(k.transpose(),k)
D=np.linalg.matrix_power(K,5)


