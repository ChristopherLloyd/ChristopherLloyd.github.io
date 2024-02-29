	## calculs ordre Sn

## bibliotheques

from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## coeficients binomiaux

def coBino(k,n):
    if k<0 or k>n:
        return(0)
    elif k==0 or k==n:
        return(1)
    else:
        m=min(k,n-k)
        p=1
        for i in range(m):
            p = p*(n-i)
        return (p/factorial(m))


## liste premiers et autres fonctions arithmétiques

def listprime(n):
    if n<2:
        return([])
    else:
        L=[2]
        k=3
        while len(L)<n:
            for d in L:
                if k%d==0:
                    break
                if d==L[-1]:
                    L.append(k)
            k+=2
        return(L)


def primeFactors(n):
    L= listprime(n)
    LP=[]
    for k in L:
        while n%k==0 and n!=1:
            LP.append(k)
            n=floor(n/k)
    return(LP)

def Factorise(n):
    F=[]
    a=0
    while n%2==0:
        a+=1
        n=floor(n/2)
    if a>0:
        F.append((2,a))
    for k in range(3,n+1,2):
        a=0
        while n%k==0 and n!=1:
            a+=1
            n=floor(n/k)
        if a>0:
            F.append((k,a))
    return(F)

def nbDiv(n):
    F=Factorise(n)
    p=1
    for (a,b) in F:
        p=p*(b+1)
    return(p)





## nombre elements Sn d'ordre p premier et graphe

def ordre_premier(p,n):
    som=0
    for k in range(1,floor(n/p)+1):
        som+=coBino(k*p,n)*factorial(k*p)/(p**k)/factorial(k)
    return(som)


def courbe_ordre_premier(p,debut,fin):
    plt.figure(0)
    X=[]
    Y=[]
    for k in range (debut,fin+1):
        Y.append(ordre_premier(p,k))
        X.append(k)
    plt.plot(X,Y)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("nombre d'éléments d'ordre {} ".format(p), fontsize=20)
    plt.xlabel("ordre du groupe allant de {} à {}".format(debut,fin), fontsize=20)
    plt.show()

def equiv_ordre_premier(p,debut,fin): # Nombre d'éléments d'ordre premier p dans Sn pour n entre debut et fin
    X=[]
    Y=[]
    for k in range (debut,fin+1):
        Y.append(log(ordre_premier(p,k)+1))
        X.append(k)
    plt.plot(X,Y,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("nombre d'éléments d'ordre {} ".format(p), fontsize=20)
    plt.xlabel("ordre du groupe", fontsize=20)
    plt.show()

def equiv_ordre_premier2(p,q,debut,fin): # comparaison du nombre d'éléments d'ordre p et q premiers dans Sn pour n entre debut et fin
    X=[]
    Y=[]
    Z=[]
    for k in range (debut,fin+1):
        Y.append(log(ordre_premier(p,k)+1))
        Z.append(log(ordre_premier(q,k)+1))
        X.append(k)
    plt.plot(X,Y,'ro',X,Z,'bo')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("nombre d'éléments d'ordre {} en rouge et {} en bleu ".format(p,q), fontsize=20)
    plt.xlabel("ordre du groupe", fontsize=20)
    plt.show()

def equiv_ordre_premier3(p,debut,fin): #comparaison entre nb d'éléments d'ordre p premiers dans Sn pour n entre debut et fin et fonction
    X,Y,M1,M2,M3=[],[],[],[],[]
    for k in range (debut,fin+1):
        Y.append(log(ordre_premier(p,k)+1))
        X.append(k)
        alpha = sqrt(2*np.pi/p)*(k)**(p/2)*(k/np.e)**(k-k/p)*(1+1/p*(p-1))**(k/p)
        beta = 1/sqrt(p)*(k/np.e)**(k-k/p)
        M1.append(log(alpha+1))
        M2.append(log(beta+1))
        #M3.append(log(sqrt(p)*(k/np.e)**(k-k/p)+1))
    plt.plot(X,Y,'k-',linewidth=3)
    plt.plot(X,M1,'k.',X,M2,'k.',linewidth=2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Comparaison équivalents, ordre = {} ".format(p), fontsize=20)
    #plt.xlabel("ordre du groupe", fontsize=20)
    plt.show()

def recurrence_ordre_fixe_log(p,m): # vérification concordance formules sommatoire et récurrence
    X=[]
    Y=[]
    L=[0 for k in range(m)]
    for i in range(p-1,m):
        L[i]=L[i-1]+factorial(i)/factorial(i-p+1)*(L[i-p]+1)
    for k in range (1,m+1):
        Y.append(log(ordre_premier(p,k)+1))
        X.append(k)
    R=np.log(np.array(L)+1)
    plt.plot(X,Y,'ro',X,R,'b+')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("nombre d'éléments d'ordre {} ".format(p), fontsize=20)
    plt.xlabel("ordre du groupe", fontsize=20)
    plt.show()
    return(R)



## génération des partitions

## majordre (heuristique, pas sur)


def majOrdre(n): # détermination empirique de l'ordre maximal des éléments de Sn (pour taille du tableau)
    if n<=4:
        return(n)
    L=listprime(n)
    maj=1
    s=0
    k=0
    for k in L:
        if s+k==n:
            return(maj*k*3)
        elif s+k<n:
            maj=maj*k
            s+=k
        else:
            return(maj*(n-s)*3)

## opérations sur les lises et tabuation

def prodlist(L): #produit des éléments d'une liste
    prod=1
    for k in L:
        prod=prod*k
    return(prod)

def prodfacto(L): #produit des factorielles des éléments d'une liste (liste de couples [élément,nbApparition] )
    prod=1
    for k in L:
        prod=prod*factorial(k[1])
    return(prod)

def prodpuiss(L): #produit des élément^nbApparition des éléments d'une liste (liste de couples [élément,nbApparition] )
    prod=1
    for k in L:
        prod=prod*(k[0]**k[1])
    return(prod)

def reduce(L): #on ne conserve qu'une apparition à chaque fois
    if L==[]:
        return([])
    C=[]
    X=L.copy()
    while X!=[]:
        a=X.pop(0)
        n=X.count(a)
        C.append(a)
        for k in range(n):
            X.remove(a)
    return(C)

def contract(L): #changement d'écriture de liste en liste couples [élément,nbApparition]
    if L==[]:
        return([])
    C=[]
    X=L.copy()
    while X!=[]:
        a=X.pop(0)
        n=X.count(a)
        C.append([a,n+1])
        for k in range(n):
            X.remove(a)
    return(C)

def support(L): #retire les cycles de longueur 1 et retourne [liste,nb de 1 qu'il y avait]
    n=L.count(1)
    X=L.copy()
    for k in range(n):
        X.remove(1)
    return([X,n])

def supportBis(L): #retire les 0 de L en fin de liste
    X=L.copy()
    while X[-1]==0:
        X.pop(-1)
    return(X)

def tabule(A,n,P): # tabule la permutation de Sn dont la dscd est représentée par A à la bonne colonne dans P
    [S,un]=support(P)
    R=reduce(S)
    C=contract(S)
    p=round(ppcmList(R))
    val=coBino(n-un,n)*factorial(n-un)/prodpuiss(C)/prodfacto(C)
    A[p-1]=A[p-1]+val
    

## ppcmlist

def pgcd(a,b):
    a,b=max(a,b),min(a,b)
    r=a%b
    while r!=0:
        b,r=r,b%r
    return(b)

def ppcm(a,b):
    return(a*b/pgcd(a,b))

def ppcmList(L):
    if L==[]:
        return(1)
    X=L.copy()
    while len(X)>1:
        a=X.pop(0)
        b=X.pop(0)
        X.append(ppcm(a,b))
    return(X[0])

## génération de l'itérateur des partitions

def accelAsc(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2*x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]
    

## génération liste des ordres dans Sn

def repOrdN(n):
    maj=majOrdre(n)
    A=[0 for k in range(maj+1)]
    for p in accelAsc(n):
        tabule(A,n,p)
    return(supportBis(A))

## génération tableau ordres lignes : Sn colonnes : ordres


def matSnOrd(m):
    M=[]
    for k in range(1,m+1):
        M.append(repOrdN(k))
    return(M)
    
##  traitement du tableau

def square(n):
    T=matSnOrd(n).copy()
    Tlen=len(T[-1])
    for K in T:
        while len(K)<Tlen:
            K.append(0)
    return(np.array(T))


def abscisses(L): #retourne la liste [1,card(L)]
    X=[]
    for k in range(1,len(L)+1):
        X.append(k)
    return(X)

def liste_ordrefixe(k,m): #nombre d'éléments d'ordre k dans Sn pour n de 1 à m
    S=square(m)
    L=[]
    n,p = np.shape(S)
    for i in range(0,n):
        L.append(S[i][k-1])
    return(L)

## graphes ordres éléments varient

def disp_ordreVarie(n): # dans Sn nombre d'éléments d'ordre variant 
    Y=repOrdN(n)
    X=abscisses(Y)
    plt.plot(X,Y,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("nombre d'éléments", fontsize=20)
    plt.xlabel("ordre des éléments", fontsize=20)
    plt.title("Ordre du groupe : {} ".format(n), fontsize=20)
    plt.show()
    return()

def disp_ordreVarie_log(n): # dans Sn nombre d'éléments d'ordre variant log(ordo)
    Y=np.log(np.array(repOrdN(n))+1)
    X=abscisses(Y)
    plt.plot(X,Y,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("ln(nombre d'éléments)", fontsize=20)
    plt.xlabel("ordre des éléments", fontsize=20)
    plt.title("Ordre du groupe : {} ".format(n), fontsize=20)
    plt.show()
    return()

def disp_ordreVarie_log_abs_renorm(n): # dans Sn nombre d'éléments d'ordre variant log(abs,ordo)
    Y=np.log(np.array(repOrdN(n))+1)
    X=np.log(abscisses(Y))
    plt.plot(X,Y,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("ln(nombre d'éléments)", fontsize=20)
    plt.xlabel("ln(ordre des éléments)", fontsize=20)
    plt.title("Ordre du groupe : {} ".format(n), fontsize=20)
    plt.show()
    return()

def disp_ordreVarie_abs_renorm(n): # Distribution de ln(ord(x)) est asmptotiquement normale de m = 1/2*ln(n)**2 , sigma = 1/3*ln(n)**3 !!
    Y=np.array(repOrdN(n))/factorial(n)
    y=np.amax(Y)
    Y=Y/y
    X=np.log(abscisses(Y))
    plt.plot(X,Y,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.ylabel("nombre d'éléments", fontsize=20)
    #plt.xlabel("ln(ordre des éléments)", fontsize=20)
    plt.title("Ordre du groupe : {} ".format(n), fontsize=20)
    plt.plot([1/2*log(n)**2,1/2*log(n)**2],[1,y]) #mean
    plt.plot([1/2*log(n)**2-1/3*log(n)*3,1/2*log(n)**2-1/3*log(n)*3],[0,y+1]) #mean - ecart type
    plt.plot([1/2*log(n)**2+1/3*log(n)*3,1/2*log(n)**2+1/3*log(n)*3],[0,y+1]) #mean + ecart type
    plt.show()
    return()


## graphes ordres éléments fixe

def disp_ordrefixe(k,n): # nombre d'éléments d'ordre 10 dans Sm pour m entre 1 et n
    Y=liste_ordrefixe(k,n)
    X=abscisses(Y)
    plt.plot(X,Y,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("nombre d'éléments d'ordre {} ".format(k), fontsize=20)
    plt.xlabel("ordre du groupe", fontsize=20)
    plt.show()
    return()

def disp_ordrefixe_log(k,n): # nombre d'éléments d'ordre k dans Sm pour m entre 1 et n log(ordo)
    Y=np.log(square(n)[:,k-1]+1)
    X=abscisses(Y)
    plt.plot(X,Y,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("ln (nombre d'éléments d'ordre {} ) ".format(k), fontsize=20)
    plt.xlabel("ordre du groupe", fontsize=20)
    plt.show()
    return()


## graphes 3d

def disp_tab_ordres(n): # image : ordonnées = ordre groupe de 1 à n, abscisses = ordre éléments, couleur = log(nombre d'éléments)
    Y=np.log(square(n)+1)
    plt.imshow(Y, cmap=None, norm=None, aspect='auto')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("ordre des éléments", fontsize=20)
    plt.ylabel("ordre du groupe", fontsize=20)
    plt.title("ln(ln(nombre d'éléments))", fontsize=20)
    plt.show()
    return()

def disp_surf_ordres(n):
    Z=np.log(np.log(square(n)+1)+1)
    x,y=np.shape(Z)
    Y,X=np.meshgrid(np.arange(1,y+1,1),np.arange(1,x+1,1))
    #return(X,Y)
    ax=plt.subplot(111, projection="3d")
    ax.plot_wireframe(X,Y,Z,cmap='binary')
    plt.show()
    

## ordre max

def disp_ordre_max(n): # ordre maximal atteint dans Sm pour m allant de 1 à n
    X=[]
    Y=[]
    for k in range(1,n+1):
        a=len(repOrdN(k))
        X.append(k+1)
        Y.append(a)
    plt.plot(X,Y,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("ordre max atteint".format(k), fontsize=20)
    plt.xlabel("ordre du groupe", fontsize=20)
    plt.show()
    return()

def disp_ordre_max_log(n): # ordre maximal atteint dans Sm pour m allant de 1 à n avec log(ornonnées)
    X=[]
    Y=[]
    for k in range(1,n+1):
        a=len(repOrdN(k))
        X.append(k+1)
        Y.append(log(a+1))
    plt.plot(X,Y,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("ln(ordre max atteint)".format(k), fontsize=20)
    plt.xlabel("ordre du groupe", fontsize=20)
    plt.show()
    return()


##
plt.grid(True)

## ordre des elements les plus abondants dans Sn

def plusAbondant(m): # détermination de l'ordre le plus représenté dans Sn entre 1 et m
    Y=[]
    for k in range(1,m+1):
        A=repOrdN(k)
        a=max(A)
        i=1+A.index(a)
        Y.append(i)
    return(Y)



def dispPlusAbondant(m):
    X=[]
    Y=[]
    for k in range(1,m+1): # graphe de l'ordre le plus représenté dans Sn entre 1 et m
        A=repOrdN(k)
        a=max(A)
        i=1+A.index(a)
        X.append(k+1)
        Y.append(i)
    plt.plot(X,Y,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("ordre le plus représenté", fontsize=25)
    plt.xlabel("ordre du groupe", fontsize=25)
    plt.show()
    return()

## graphes ordres éléments varient avec les premiers mis en valeur


def disp_ordreVariePrem(n): # nombre d'éléments d'ordre variant dans Sn avec les ordres premiers mis en valeur
    Y=repOrdN(n)
    X=abscisses(Y)
    P=listprime(n)
    Z=len(Y)*[0]
    for k in range(len(Y)):
        if (k+1) in P:
            Z[k]=Y[k]
    plt.plot(X,Y,'b+',X,Z,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("nombre d'éléments, les premiers sont en gras", fontsize=20)
    plt.xlabel("ordre des éléments", fontsize=20)
    plt.title("Ordre du groupe : {} ".format(n), fontsize=20)
    plt.show()
    return()


def disp_ordreVariePrem_log(n): # log(nombre d'éléments) d'ordre variant dans Sn avec les prdres premiers mis en valeur
    Y=repOrdN(n)
    X=abscisses(Y)
    P=listprime(n)
    Z=len(Y)*[0]
    for k in range(len(Y)):
        Y[k]=log(Y[k]+1)
        if (k+1) in P:
            Z[k]=Y[k]
    plt.plot(X,Y,'b+',X,Z,'ro')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("nombre d'éléments, les premiers sont en gras", fontsize=20)
    plt.xlabel("ordre des éléments", fontsize=20)
    plt.title("Ordre du groupe : {} ".format(n), fontsize=20)
    plt.show()
    return()

## ordre moyen dans Sn

def ordre_moyen(n): # détermination de l'ordre moyen des éléments de Sn
    X=repOrdN(n)
    x=len(X)
    Y=[]
    for k in range(0,x):
        Y.append((k+1)*X[k])
    return(sum(Y)/factorial(n))

def trace_ordre_moyen(min,max): # tracé pour n allant de min à max de l'ordre moyen des éléments de Sn
    X=[k for k in range(min,max+1)]
    Y=[log(ordre_moyen(n)) for n in range(min,max+1)]
    Z=[1/2*log(k)**(2) for k in range(min,max+1)]
    plt.plot(X,Y,'ro',X,Z,'b<')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.ylabel("ordre moyen des éléments", fontsize=20)
    #plt.xlabel("ordre du groupe", fontsize=20)
    #plt.title("Graphe de l'ordre moyens dans Sn pour n allant de {} à {}".format(min,max), fontsize=20)
    plt.show()

## moyenne de ln(ord) dans Sn
def ordre_moyen_log(n): # détermination de l'ordre moyen des éléments de Sn
    X=repOrdN(n)
    x=len(X)
    Y=[]
    for k in range(0,x):
        Y.append(log(k+1)*X[k])
    return(sum(Y)/factorial(n))

def trace_ordre_moyen_log(min,max): # tracé pour n allant de min à max de l'ordre moyen des éléments de Sn
    X=[k for k in range(min,max+1)]
    Y=[ordre_moyen_log(n) for n in range(min,max+1)]
    Z=[1/2*log(k)**(2) for k in range(min,max+1)]
    plt.plot(X,Y,'ro',X,Z,'b<')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.ylabel("ordre moyen des éléments", fontsize=20)
    #plt.xlabel("ordre du groupe", fontsize=20)
    #plt.title("Graphe de l'ordre moyens dans Sn pour n allant de {} à {}".format(min,max), fontsize=20)
    plt.show()


## Agancement graphique des résulatats

##  Courbes ordre premier fixé et n varie en log

##      agencement 1 : nmax=50, premiers=2,11

'''

plt.figure(1)

plt.subplot(121)
X=[]
Y=[]
for k in range (1,50+1):
    Y.append(log(ordre_premier(2,k)+1))
    X.append(k)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ordre = {} ".format(2), fontsize=20)
#plt.ylabel("ordre = {} ".format(2), fontsize=20)
#plt.xlabel("ordre du groupe", fontsize=20)


plt.subplot(122)
X=[]
Y=[]
for k in range (1,50+1):
    Y.append(log(ordre_premier(11,k)+1))
    X.append(k)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ordre = {} ".format(11), fontsize=20)
#plt.ylabel("ordre = {} ".format(11), fontsize=20)
#plt.xlabel("ordre du groupe", fontsize=20)

plt.tight_layout()
plt.show()

'''

##      agencement 2 : nmax=50, premiers=5,11,19,23
'''

plt.figure(2)


plt.subplot(221)
X=[]
Y=[]
for k in range (1,50+1):
    Y.append(log(ordre_premier(5,k)+1))
    X.append(k)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ordre = {} ".format(5), fontsize=20)
#plt.ylabel("ordre = {} ".format(5), fontsize=20)
#plt.xlabel("ordre du groupe", fontsize=20)

plt.subplot(222)
X=[]
Y=[]
for k in range (1,50+1):
    Y.append(log(ordre_premier(13,k)+1))
    X.append(k)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ordre = {} ".format(13), fontsize=20)
#plt.ylabel("ordre = {} ".format(13), fontsize=20)
#plt.xlabel("ordre du groupe", fontsize=20)

plt.subplot(223)
X=[]
Y=[]
for k in range (1,50+1):
    Y.append(log(ordre_premier(19,k)+1))
    X.append(k)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ordre = {} ".format(19), fontsize=20)
#plt.ylabel("ordre = {} ".format(19), fontsize=20)
#plt.xlabel("ordre du groupe", fontsize=20)

plt.subplot(224)
X=[]
Y=[]
for k in range (1,50+1):
    Y.append(log(ordre_premier(23,k)+1))
    X.append(k)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ordre = {} ".format(23), fontsize=20)
#plt.ylabel("ordre = {} ".format(23), fontsize=20)
#plt.xlabel("ordre du groupe", fontsize=20)

plt.tight_layout()
plt.show()

'''

##  Courbes ordre non premier fixé et n varie en log

'''
plt.figure(1)


plt.subplot(221)
Y=np.log(square(50)[:,10-1]+1)
X=abscisses(Y)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ordre = {} ".format(10), fontsize=20)
#plt.ylabel("ln (nombre d'éléments d'ordre {} ) ".format(k), fontsize=20)
#plt.xlabel("ordre du groupe", fontsize=20)

plt.subplot(222)
Y=np.log(square(50)[:,14-1]+1)
X=abscisses(Y)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ordre = {} ".format(14), fontsize=20)
#plt.ylabel("ln (nombre d'éléments d'ordre {} ) ".format(k), fontsize=20)
#plt.xlabel("ordre du groupe", fontsize=20)

plt.subplot(223)
Y=np.log(square(50)[:,26-1]+1)
X=abscisses(Y)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ordre = {} ".format(26), fontsize=20)
#plt.ylabel("ln (nombre d'éléments d'ordre {} ) ".format(k), fontsize=20)
#plt.xlabel("ordre du groupe", fontsize=20)

plt.subplot(224)
Y=np.log(square(50)[:,33-1]+1)
X=abscisses(Y)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("ordre = {} ".format(33), fontsize=20)
#plt.ylabel("ln (nombre d'éléments d'ordre {} ) ".format(k), fontsize=20)
#plt.xlabel("ordre du groupe", fontsize=20)

plt.tight_layout()
plt.show()

'''

##  ordre varie n fixe avec ln(abscisses=ordre des éléments dans Sn)

## ordre = 15,30,45,60
'''
plt.figure(1)


plt.subplot(221)
Y=np.array(repOrdN(15))/factorial(15)
y=np.amax(Y)
Y=Y/y
X=np.log(abscisses(Y))
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.ylabel("nombre d'éléments", fontsize=20)
#plt.xlabel("ln(ordre des éléments)", fontsize=20)
plt.title("Ordre du groupe : {} ".format(15), fontsize=20)
plt.plot([1/2*log(15)**2,1/2*log(15)**2],[0,y+1],'g') #mean
plt.plot([1/2*log(15)**2-1/3*log(15)*3,1/2*log(15)**2-1/3*log(15)*3],[0,y+1],'g--') #mean - ecart type
plt.plot([1/2*log(15)**2+1/3*log(15)*3,1/2*log(15)**2+1/3*log(15)*3],[0,y+1],'g--') #mean + ecart type

plt.subplot(222)
Y=np.array(repOrdN(30))/factorial(30)
y=np.amax(Y)
Y=Y/y
X=np.log(abscisses(Y))
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.ylabel("nombre d'éléments", fontsize=20)
#plt.xlabel("ln(ordre des éléments)", fontsize=20)
plt.title("Ordre du groupe : {} ".format(30), fontsize=20)
plt.plot([1/2*log(30)**2,1/2*log(30)**2],[0,y+1],'g') #mean
plt.plot([1/2*log(30)**2-1/3*log(30)*3,1/2*log(30)**2-1/3*log(30)*3],[0,y+1],'g--') #mean - ecart type
plt.plot([1/2*log(30)**2+1/3*log(30)*3,1/2*log(30)**2+1/3*log(30)*3],[0,y+1],'g--') #mean + ecart type

plt.subplot(223)
Y=np.array(repOrdN(45))/factorial(45)
y=np.amax(Y)
Y=Y/y
X=np.log(abscisses(Y))
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.ylabel("nombre d'éléments", fontsize=20)
#plt.xlabel("ln(ordre des éléments)", fontsize=20)
plt.title("Ordre du groupe : {} ".format(45), fontsize=20)
plt.plot([1/2*log(45)**2,1/2*log(45)**2],[0,y+1],'g') #mean
plt.plot([1/2*log(45)**2-1/3*log(45)*3,1/2*log(45)**2-1/3*log(45)*3],[0,y+1],'g--') #mean - ecart type
plt.plot([1/2*log(45)**2+1/3*log(45)*3,1/2*log(45)**2+1/3*log(45)*3],[0,y+1],'g--') #mean + ecart type

plt.subplot(224)
Y=np.array(repOrdN(60))/factorial(60)
y=np.amax(Y)
Y=Y/y
X=np.log(abscisses(Y))
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.ylabel("nombre d'éléments", fontsize=20)
#plt.xlabel("ln(ordre des éléments)", fontsize=20)
plt.title("Ordre du groupe : {} ".format(60), fontsize=20)
plt.plot([1/2*log(60)**2,1/2*log(60)**2],[0,y+1],'g') #mean
plt.plot([1/2*log(60)**2-1/3*log(60)*3,1/2*log(60)**2-1/3*log(60)*3],[0,y+1],'g--') #mean - ecart type
plt.plot([1/2*log(60)**2+1/3*log(60)*3,1/2*log(60)**2+1/3*log(60)*3],[0,y+1],'g--') #mean + ecart type


plt.tight_layout()
plt.show()
'''
## ordre varie n fixe avec abscisses=ordre des éléments dans Sn
'''
plt.figure(1)


plt.subplot(121)
Y=repOrdN(15)
X=abscisses(Y)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.ylabel("nombre d'éléments", fontsize=20)
#plt.xlabel("ordre des éléments", fontsize=20)
plt.title("Ordre du groupe : {} ".format(15), fontsize=20)

plt.subplot(122)
Y=repOrdN(20)
X=abscisses(Y)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.ylabel("nombre d'éléments", fontsize=20)
#plt.xlabel("ordre des éléments", fontsize=20)
plt.title("Ordre du groupe : {} ".format(20), fontsize=20)



plt.tight_layout()
plt.show()

'''
## ordre varie n fixe avec abscisses=ordre des éléments dans Sn -> ln(ord)

'''
plt.figure(1)


plt.subplot(221)
Y=np.log(np.array(repOrdN(15))+1)
X=abscisses(Y)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.ylabel("ln(nombre d'éléments)", fontsize=20)
#plt.xlabel("ordre des éléments", fontsize=20)
plt.title("Ordre du groupe : {} ".format(15), fontsize=20)

plt.subplot(222)
Y=np.log(np.array(repOrdN(30))+1)
X=abscisses(Y)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.ylabel("ln(nombre d'éléments)", fontsize=20)
#plt.xlabel("ordre des éléments", fontsize=20)
plt.title("Ordre du groupe : {} ".format(30), fontsize=20)

plt.subplot(223)
Y=np.log(np.array(repOrdN(45))+1)
X=abscisses(Y)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
#plt.ylabel("ln(nombre d'éléments)", fontsize=20)
#plt.xlabel("ordre des éléments", fontsize=20)
plt.title("Ordre du groupe : {} ".format(45), fontsize=20)

plt.subplot(224)
Y=np.log(np.array(repOrdN(60))+1)
X=abscisses(Y)
plt.plot(X,Y,'ro')
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
#plt.ylabel("ln(nombre d'éléments)", fontsize=20)
#plt.xlabel("ordre des éléments", fontsize=20)
plt.title("Ordre du groupe : {} ".format(60), fontsize=20)


plt.tight_layout()
plt.show()
'''
## 




