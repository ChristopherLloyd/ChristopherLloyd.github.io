## Encodage utf8 (par défault pour python 3)
#coding:utf-8

## Bibli

from math import *
import numpy as np



## Decorateur de Memoisation (programmation dynamique)

def memoise(func):
    cache={}
    def wrapper(*args):
        if args not in cache:
            cache[args]=func(*args)
        return cache[args]
    return wrapper

"""
* signale que ce qui suit est un tuple (longueur indeterminee) de plusieurs arguments et pas un seul argument tuple, tandis que args est un seul élément tuple lorsque non précede de *
"""

##

@memoise
def pascal(p,n):
    if n<p:
        return 0
    if n==p or p==0:
        return 1
    else:
        return pascal(p-1,n-1)+pascal(p,n-1)
    
def pgcd(a,b):
    a,b=max(a,b),min(a,b)
    r=a%b
    while r!=0:
        b,r=r,b%r
    return(b)

def factorise(n):
    """retroune liste des (diviseur, exposant)"""
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


## liste premiers

def listPrime(n):
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

'''
Dans crible_ncarre on retourne les p<n**2
qui ne sont pas dans la liste des multiples de i<n de l'intervalle ]i, n*n[
car un non premier <= n*n admet un diviseur strict dans [2,n]
'''
def crible_ncarre(n):
    return([p for p in range(2,n*n) if p not in [j for i in range(2,n) for j in range(2*i,n*n,i)]])


def primeFactors(n):
    L= listPrime(n)
    LP=[]
    for k in L:
        while n%k==0 and n!=1:
            LP.append(k)
            n=floor(n/k)
    return(LP)

## opérations sur les lises et tabuation

def prodList(L):
    """produit des éléments d'une liste"""
    prod=1
    for k in L:
        prod=prod*k
    return(prod)

def prodFacto(L):
    """produit des factorielles des éléments d'une liste (liste de couples [élément,nbApparition] )"""
    prod=1
    for k in L:
        prod=prod*factorial(k[1])
    return(prod)

def prodPuiss(L):
    """produit des élément^nbApparition des éléments d'une liste (liste de couples [élément,nbApparition] )"""
    prod=1
    for k in L:
        prod=prod*(k[0]**k[1])
    return(prod)

def reduire(L):
    """on ne conserve qu'une apparition à chaque fois"""
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

def contract(L):
    """changement d'écriture de liste en liste couples [élément,nbApparition]"""
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

## Mergesort: Trifusion recursif

def merge(a,b):
    """fusionne deux listes triées"""
    p,q=len(a),len(b)
    c=[None]*(p+q)
    i=j=0
    for k in range(p+q):
        if j>=q:
            c[k:]=a[i:]
            break
        elif i>=p:
            c[k:]=b[j:]
            break
        elif a[i]<b[j]:
            c[k]=a[i]
            i+=1
        else:
            c[k]=b[j]
            j+=1
    return c

def mergeSort(t):
    """trie une liste"""
    n=len(t)
    if n<2:
        return t
    a=mergeSort(t[:n//2])
    b=mergeSort(t[n//2:])
    return merge(a,b)

## Generateur de partitions
"""If you have a partition of n, you can reduce it to a partition of n-1 in a canonical way by subtracting one from the smallest item in the partition. E.g. 1+2+3 => 2+3, 2+4 => 1+4. This algorithm reverses the process: for each partition p of n-1, it finds the partitionS of n that would be reduced to p by this process. Therefore, each partition of n is output exactly once, at the step when the partition of n-1 to which it reduces is considered."""

def partitions_gen_rec(n):
    if n == 0:# cas de base de la récursion: 0 est la somme de la liste vide
        yield []
        return
    for p in partitions_gen_rec(n-1):# modifier partitions de n-1 pour fabriquer partitions de n
        yield [1] + p 
        if p and (len(p) <= 1 or p[1] > p[0]):#if [] : False ; if [a,...]: True
            yield [p[0] + 1] + p[1:]

def parcourir_iter(L):
    res=[]
    for k in L:
        res.append(k)
    return(res)

## Generateur permutations (ou anagrammes) d'un mot en ordre lexico
def lexico_permute_string(s):
    ''' Generate all permutations in lexicographic order of string `s`

        This algorithm, due to Narayana Pandita, is from
        https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order

        To produce the next permutation in lexicographic order of sequence `a`

        1. Find the largest index j such that a[j] < a[j + 1]. If no such index exists, 
        the permutation is the last permutation.
        2. Find the largest index k greater than j such that a[j] < a[k].
        3. Swap the value of a[j] with that of a[k].
        4. Reverse the sequence from a[j + 1] up to and including the final element a[n].
    '''

    a = sorted(s)
    n = len(a) - 1
    while True:
        yield ''.join(a)

        #1. Find the largest index j such that a[j] < a[j + 1]
        for j in range(n-1, -1, -1):
            if a[j] < a[j + 1]:
                break
        else:
            return

        #2. Find the largest index k greater than j such that a[j] < a[k]
        v = a[j]
        for k in range(n, j, -1):
            if v < a[k]:
                break

        #3. Swap the value of a[j] with that of a[k].
        a[j], a[k] = a[k], a[j]

        #4. Reverse the tail of the sequence
        a[j+1:] = a[j+1:][::-1]


##


