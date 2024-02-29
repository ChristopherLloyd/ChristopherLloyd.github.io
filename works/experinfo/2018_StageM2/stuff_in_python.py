#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:22:25 2019

@author: christopher-lloyd
"""

## bibliotheques
from math import *
import numpy as np
import numpy.polynomial.polynomial as poly

import scipy as sc
import scipy.sparse.csgraph as g
#import scipy.integrate as quad
#import scipy.linalg as sla
#import scipy.optimize as opt

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

## augmentation de la taille pile de recursion

import sys
sys.setrecursionlimit(5000)


## Classes
    ## Définition class Pile1

class Pile1:
    def __init__(self):
        self.lst = []
    
    def empty(self):
        return self.lst == []
    
    def push(self, x):
        self.lst.append(x)
    
    def pop(self):
        if self.empty():
            raise ValueError("pile vide")
        return self.lst.pop()

    ## utilisation de Pile1

p=Pile1()
for i in range(1,11):
    p.push(i)
while not p.empty():
    print(p.pop(), end=' - ')
print()

    ## Classe pile2 perso

class Cell:
    def __init__(self,x):
        self.val = x
        self.next = None

class Pile2:
    def __init__(self):
        self.lst = None
    
    def empty(self):
        return self.lst is None
    
    def push(self,x):
        c=Cell(x)
        c.next=self.lst
        self.lst = c
    
    def pop(self):
        if self.empty():
            raise ValueError("pile vide")
        c = self.lst
        self.lst = c.next
        return c.val

    ## utilisation de Pile2

p=Pile2()
for i in range(1,100):
    p.push(i)
while not p.empty():
    print(p.pop(), end=' - ')
print()

    ## classe liste doubement chainée

class Cell_d :
    
    def __init__(self,x):
        self.val = x
        self.next = None
        self.prec = None

class Liste_d:
    
    def __init__(self):
        self.first = None
        self.last = None
        self.len = 0
        
    def empty(self):
        return self.len is 0
    
    def addend(self, x):
        c=Cell_d(x)
        self.len+=1
        if self.len is 1:
            self.last = c
            self.first = c
        else :
            self.last.next, c.prec = c, self.last
            self.last = c
    
    def addbeg(self,x):
        c=Cell_d(x)
        self.len+=1
        if self.len is 1:
            self.last = c
            self.first = c
        else :
            self.first.prec, c.next = c, self.first
            self.first = c
    
    def delend(self):
        if self.len is 0:
            raise ValueError("empty")
        else:
            self.last, self.last.prec = self.last.prec, None
    
    def delbeg(self):
        if self.len is 0:
            raise ValueError("empty")
        else:
            self.first, self.first.next = self.first.next, None

    ##  classes Heritage

class Personne:
    def __init__(self, nom):
        self.nom = nom
        self.prenom = "Martin"
    def __str__(self):
        """Méthode appelée lors d'une conversion de l'objet en chaîne"""
        return "{0} {1}".format(self.prenom, self.nom)

class AgentSpecial(Personne):
    """Classe définissant un agent spécial.
    Elle hérite de la classe Personne"""
    
    def __init__(self, nom, matricule):
        """Un agent se définit par son nom et son matricule"""
        Personne.__init__(self, nom)    # On appelle explicitement le constructeur (__init__) de Personne : heritage
        self.matricule = matricule      # possede un attribut en plus de la classe Personne
        
    def __str__(self):
        """Méthode appelée lors d'une conversion de l'objet en chaîne"""
        return "Agent {0}, matricule {1}".format(self.nom,self.matricule)


#issubclass et isinstance font ce qu'elles disent 
vrai1 = issubclass(AgentSpecial, Personne) and issubclass(Personne, object)   #objet : classe grand-mere de toutes celles que l'on peut
                                                                            #creer et qui definit nottament les methodes __init__ etc.
vrai2 = isinstance(Personne('Machin'),Personne)


## TP Dictionnaire ordonné : POO

class DictionnaireOrdonne:
    
    def __init__(self,base={}, **donnees):
        """constructeur, on peut l'appeller sans arguments, ou avec une base et rajouter des arguments cle = valeur """
        self._cles = [] #attribut : liste cles
        self._valeurs = [] #attribut : liste valeurs
        #on verifie que base est un dictionnaire
        if type(base) not in (dict, DictionnaireOrdonne):
            raise TypeError("Le type attendu est un dictionnaire")
        #on recupere les donnes de base et de donnees
        for cle in base:
            self[cle]=base[cle] #surcharge __setitem__ defini plus loin 
        for cle in donnees:
            self[cle]=donnees[cle]
    
    def __repr__(self):
        """representation de l'objet, sera renvoye lorsqu'on saisit le dico dans l'interpreteur ou applique repr(dico) """
        chaine = "{"
        premier_passage = True
        for cle, valeur in self.items():
            if not premier_passage :
                chaine += ", " #on ajoute la virgule comme separateur avant d'ecrire 'cle : valeur' sauf le premier ou on met pas la virgule
            else:
                premier_passage = False
            chaine += repr(cle) + ": " + repr(valeur)
        chaine +="}"
        return chaine
    
    def __str__(self):
        """fonction appelle lorsqu on fait print(dico) ou str(dico), on redirige vers repr"""
        return repr(self)
    
    def __len__(self):
        return len(self._cles)
    
    def __contains__(self, cle):
        """surcharge de l'operateur in  (pour: cle in dico)"""
        return cle in self._cles
    
    def __getitem__(self, cle):
        """surcharge des [] dans : dico[cle]"""
        if cle not in self._cles:
            raise KeyError("La cle {0} est absente du dico".format(cle))
        else:
            indice = self._cles.index(cle)
            return self._valeurs[indice]
    
    def __setitem__(self, cle, valeur):
        """methode speciale appelle lorsqu'on essaye de modifier une association ou d'en rajouter"""
        if cle in self._cles :
            indice = self._cles.index(cle)
            self._valeurs[indice] = valeur
        else :
            self._cles.append(cle)
            self._valeurs.append(valeur)
    
    def __delitem__(self, cle):
        """methode appelle quand on modifie une cle par la fonction : del dico[cle] """
        if cle not in self._cles:
            raise KeyError("la cle {} est absente".format(cle))
        else:
            indice = self._cles.index(cle)
            del self._cles[indice]
            del self._valeurs[indice]
    
    def __iter__(self):
        """methode de parcours de l'objet. On renvoie l'iterateur des cles"""
        return iter(self._cles)
    
    def __add__(self, autre_objet):
        """sucharge de dico + dico """
        if type(autre_objet) is not type(self):
            raise TypeError("Impossible de concdatener {} et {}".format(type(self), type(autre_objet)))
        else:
            nouveau = DictionnaireOrdonne(self) #on fait une copie de self en utilisant self comme base
            for cle, valeur in autre_objet.items(): #on ajoute les associations de l'autre dico
                nouveau[cle]=valeur
        return nouveau
    
    def items(self):
        """on construit un generateur renvoyant les couples (cle, valeur)"""
        for cle, valeur in zip(self._cles, self._valeurs):
            yield (cle, valeur)
    
    def values(self):
        """renvoie la liste des valeurs"""
        return(self._values)
    
    def keys(self):
        """renvoie liste des cles"""
        return self._cles
    
    def reverse(self):
        """methode qui retourne la liste"""
        new_cles, new_valeurs = [], []
        for i in range(len(self)-1,-1, -1):#on utilise len
            new_cles.append(self._cles[i])
            new_valeurs.append(self._valeurs[i])
        self._cles = new_cles
        self._valeurs = new_valeurs
    
    def sort(self):
        """methode qui trie selon les cles"""
        ncles = sorted(self._cles) #on trie les cles
        nval = []
        for cle in ncle:
            nval.append(self[cle]) #on utilise __getitem__
        self._cles = ncles
        self._valeurs = nval


## Recursif et Memoisation

    ## tours de hanoi : jolie generalisation et recursion

def hanoi(n, i=1, j=2, k=3):
    if n==0:
        return None
    hanoi(n-1, i, k, j)
    print("Deplacer le disque {} de la tige {} vers la tige {}. " .format(n, i, k))
    hanoi(n-1, j, k)

    ## mergesort : tri fusion : recursif

def merge(a,b):
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

def mergesort(t):
    n=len(t)
    if n<2:
        return t
    a=mergesort(t[:n//2])
    b=mergesort(t[n//2:])
    return merge(a,b)

    ## memoisation avec un decorateur

def memoise(func):
    cache={}
    def wrapper(*args):           # * signale que ce qui suit est un tuple (longueur indeterminee) de plusieurs arguments et pas un seul argument tuple
        if args not in cache:     # args est un seul tuple lorsque non précede de *
            cache[args] = func(*args)
        return cache[args]         # permet de remonter du bas de l'arbre vers le haut en combinant les resultats des sous problemes successifs : programmation dynamique
    return wrapper

@memoise
def fib(n) :
    if n<2 :
        return n
    return fib(n-1)+fib(n-2)

#@controler_temps(5)
def pascalnul(p,n):
    if n==p or p==0:
        return 1
    else:
        return pascalnul(p-1,n-1)+pascalnul(p,n-1)

@memoise
def pascal(p,n):
    if n==p or p==0:
        return 1
    else:
        return pascal(p-1,n-1)+pascal(p,n-1)

##  Decorateurs en general
    ## decorateur a parametres : nb_secs

import time
"""la fonction time() de ce module mesure le tps ecoule depuis janvier 1970"""

'''La fonction controler_temps prends en parametre nb_secs 
et renvoie le decorateur associe a ce parametre'''

def controler_temps(nb_secs):
    """renvoie un decorateur destiner a controler que le temps mis par la fonction qu'il decorera est inferieur a nb_secs"""
    def decorateur(fonction_a_executer):
        """ce decorateur va remplacer fonction_a_executer par fonction modifiee"""
        def fonction_modifiee(*parametres_non_nommes, **parametres_nommes):
            """fonction renvoyee par deco pour remplacer fonction_a_executer"""
            tps_avant = time.time()
            valeur_renvoyee = fonction_a_executer(*parametres_non_nommes, **parametres_nommes)
            tps_apres=time.time()
            tps_exec=tps_apres-tps_avant
            if tps_exec>nb_secs:
                print("La fonction {} a mis {} sec pour s'executer".format(fonction_a_executer, tps_exec))
            return valeur_renvoyee #fonction_modifiee renvoie meme valeur que l'originale, elle ne fait qu'afficher du tps avant
        return fonction_modifiee #le decorateur renvoie la fonction modifiee
    return decorateur #le fabricant de decorateur renvoie le decorateur

@controler_temps(4) #ceci est le decorateur obtenu pour le parametre 4
def attendre():
    print("a vos marques... pret... partez!! ")
    input("Appuyez sur entree")

    ## controler les types
'''ceci est un constructeur a nombre variable de parametres'''

def controler_types(*t_args, **t_kwargs):
    def decorateur(fonction_a_executer):
        def fonction_modifiee(*args, **kwargs):
            if len(args) != len(t_args):
                raise TypeError("il y a {} arguments au lieu de {}".format(len(args), len(t_args)))
            for i, arg in enumerate(args):
                if t_args[i] is not type(args[i]):
                    raise TypeError("L'argument libre numero {} est du type {} au lieu de {}".format(i, type(args[i]), t_args[i]))
            for cle in kwargs:
                if cle not in t_kwargs:
                    raise TypeError("l'argument {} n'a aucun type".format(repr(cle)))
                if type(kwargs[cle]) is not t_kwargs[cle]:
                    raise TypeError("{} est du type {} au lieu de {}".format(cle, type(kwargs[cle]), t_kwargs[cle]))
            return fonction_a_executer(*args, **kwargs)
        return fonction_modifiee
    return decorateur

@controler_types(int, i=int, j=int, k=int)
def perroquet(n, i=0, j=0, k=0):
    return (n,i,j,k)

    ## decorateur de classe : n'autorise qu'une instanciation de la classe
def singleton(classe_definie):
    instances = {} # Dictionnaire de nos instances singletons 
    def get_instance(*args):
        if classe_definie not in instances: # On crée notre premier objet de classe_definie
            instances[classe_definie] = classe_definie(*args) #cle = constructeur de classe = fonction d'instanciation : valeur= instance de la classe singleton
        return instances[classe_definie] #on retourne la valeur : instance qui serait produite à l'appel de la : clé
    return get_instance #on retourne la fonction destinee a remplacer la fontion d'instanciation de la classe decoree

"""
##### forme generale de la definition d'un decorateur #####

def maDeco(func):
    def wrapper(*args, **kwargs):
        a executer avant la fonction
        func(*args, **kwargs)
        a executer apres la fonction
    return wrapper


##### Equivalent de l'expression @decorateur #####

@decorateur
def fonction_a_decorer:
    pass

<=>

def fonction_a_decorer:
    pass

fonction_a_decorer = decorateur(fonction_a_decorer)

##### VOIR : help('class') #####

Classes can also be decorated: just like when decorating functions,

   @f1(arg)
   @f2
   class Foo: pass

is equivalent to

   class Foo: pass
   Foo = f1(arg)(f2(Foo))


"""

############################### Boite à outils ###############################

## chaines de caractere ; supportent : st[indice], len(st), + , * , for c in chaine

#test pour savoir si b facteur de c
Est_un_facteur = b in c


#formatage des str
adresse = "{code_postal}{nom_ville} ({pays})".format(code_postal=75003,nom_ville=" Paris ", pays="France")
print(adresse)

# methodes split et join
ma_chaine = 'Bonjour.à.tous'
ma_liste = ma_chaine.split('.')     # string -> list en separant sur le motif separateur entre '' en arg du split
ma_chaine2 = '.sep.'.join(ma_liste) # list -> string en concatenant une liste de string avec .sep. comme separateur

#TUPLISATION: conversion conteneur d'arguments pour donner separement a fonction
a = [1,2,3,4]
print(a, *a) # arg, *arg, **arg (nombre de * doit croitre)

# str.replace(old,new,numreplacemax)
"acvacvacvjfhacv".replace("acv","_-_",3)

# str.index(ch)
"acvcvjfhev".index("jfh")

#sorted trie un iterable (ici liste ) avec la fonction d'ordre ici key : ordre lexico sur les minuscules
st = sorted("This is a test string from Andrew".split(), key=str.lower) #par defaut, split()=split(" ")
print(st) #chaine des mots ordonnes

# zip, enumerate 
for c,d in zip("zipeclair","zipeclair"):
    print(c,d)
for indice, valeur in enumerate('qdljbvb'):
    print(indice,valeur)

#break interromp une boucle
#continue reprend au debut de la boucle (a l'etape suivante si boucle for)


## Listes
"""attention aux copies dans les tableaux"""
import copy as cop
A=[[1],[2,2]]
L=cop.deepcopy(A) #la modification d'un 2 dans L n'affecte pas celle de A

def crible_ncarre(n):
    #on retourne les p qui ne sont pas dans la liste des multiples de i<n qui st dans ]i, n*n[
    #car un non premier <= n*n admet un diviseur strict dans [2,n]
    return([p for p in range(2,n*n) if p not in [j for i in range(2,n) for j in range(2*i,n*n,i)]])

'''
[expression for indice_1 in iterable_1 [if condition_1]
            for indice_2 in iterable_2 [if condition_2]
            ...
            for indice_n in iterable_n [if condition_n]]



Le chronologie de l'évaluation se fait comme suit:
    le n-uplet (indice_1,...,indice_n) parcours toutes les valeurs possibles dans l'ordre "lexico" :
    indice_1,...,indice_n-1 sont fixes et on fait varier indice_n puis
    on incremente indice_n-1 pour refaire varier indice_n etc.

Remarque : iterable_j et condition_j peuvent dependre de indice_i<j puisque 
           i est fixe au moment du parcours de j

'''

## Erreurs, declanchemet rattrapage

"""definition heritage de Exception"""
class MonException(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

exn1 = MonException('expression de lexeption exn1')

class MonExceptionTriviale(Exception):
    pass

exn2 = MonExceptionTriviale()

"""
rattrappage:  

try:
    expression a executer
except: type_de_exception as exception_retournee:
    instructions si exception levee
else:
    instructions si rien a ete rattrappe et si aucune autre erreur survenue
finally:
    instructions tjrs executees meme en cas de return rencontre avant

"""
from numpy.random import randint

try :
    if randint(2) == 1:
        #a=0/0
        raise MonException("1")
except MonException as e:
    print('erreur rattrapée :', e.message, "   "+ str(type(e)) )
else:
    print('aucune erreur rattrapée')
finally:
    print('toujours exécuté meme si yavais un return')


# assert expression_logique, AssertionError
annee = input (" Saisissez une année supérieure à 0 :")
try :
    annee = int( annee ) # Conversion de l'annee
    assert annee > 0
except ValueError :
    print (" Vous n'avez pas saisi un nombre .")
except AssertionError :
    print ("L'annee saisie est inferieure ou egale à 0.")


## Ensembles
ens1 = set([k for k in range(10)])
ens2={2,53,11,9,7,7,8,4} #il ne retiendra qu'un deul des deux 7

'''consulter help('set')'''

#surcharge des operateurs
inter = ens1&ens2
union = ens1|ens2
diff = ens1-ens2
delta = ens1^ens2
inclu = ens1 <= ens2

#quelques methodes
ens1.add(23)
x=ens1.pop()
ens1.isdisjoint(ens2)


##cGraphes !!!!!

    ## Dictionaires, cles = sommets, valeurs = liste/ensemble des voisins
graph = {'A': ['B', 'C'],
             'B': ['C', 'D'],
             'C': ['D'],
             'D': ['C'],
             'E': ['F'],
             'F': ['C']}
    ## chemin de 1 vers 2 dans graph s'il existe par backtracking

def find_path(graph, start, end, path=[]):
        path += [start]
        if start == end:
            return path
        if start not in graph:                              #peut etre qu'un sommet pointe vers un sommet qui n'est pas enregistre en tant que tel comme clé
            return None
        for node in graph[start]:
            if node not in path:                            # on a pas envie de boucler à l'infini donc on visite les pas encore vus
                newpath = find_path(graph, node, end, path) #on essaye en emprutant la voie par node
                if newpath: return newpath                  #if iterable s'evalue en vrai ssi l'iterable est non vide
        return None

find_path(graph, 'A', 'D')

#test booleen sur iterable, entier, None,.. : faux si vide, vrai sinon.
if [] or {} or () or 0 or None: 
    print(1) 
else: 
     print(2)

#evaluation conditionelle compressée
a = ('expression1' if 'test'=='vrai' else 'expression2')


## Calcule tous les chemins de depart a arrivee

def find_all_paths(graph, start, end, path=[]):
        path += [start]       #on ajoute le nouveau sommet au chemin
        if start == end:            #on a trouve un chemin de start a end, on renvoye le chemin trouve, un singleton chemin
            return [path]
        if start not in graph:      # un sommet point vers son voisin start inconnu: impasse
            return []
        paths = []                  #on initialise l'ensemble des chemins elementaires trouves, et qui commencent par path, à vide
        for node in graph[start]:   #on cherche à partir de chacun des voisins
            if node not in path:    #on ne veut pas boucler sur un cycle : chemins élémentaires
                newpaths = find_all_paths(graph, node, end, path) #on regarde les chemins aboutissant a partir du chemin choisi, renvoie une liste de chemins
                paths.extend(newpaths) #on rajoute chaque nvx chemin trouvé a la liste de tous les chemins
        return paths

    ## Calcule chemin plus court

def find_shortest_path(graph, start, end, path=[]):
        path += [start]
        if start == end:
            return path
        if start not in graph:
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = find_shortest_path(graph, node, end, path) #on parcours depuis chacun des voisins
                if newpath:                                          #si on a trouve un nouveau chemin (ie pas None)
                    if not shortest or len(newpath) < len(shortest): #si shortest est None ou si on a trouve un plus court
                        shortest = newpath                           #on remplace shortest par ce plus court
        return shortest


## Polynomes : numpy.polynomial.polynomial as poly 

#classe Polynomial
#attributs : coefs, domain, window
p=poly.Polynomial([1,2,5,6,5,8,9,5,5,5,5,4,56,4,5,5,4,5])

#quelques operations
q=p+p
s=p**3
u,v = divmod(s,p)                            #division euclidienne couple quot, reste
x=poly.polyval([complex(1,2),2.5],v.coef)    #evaluation sur complexe et reel
r=poly.polyroots(u)                          #racines dans domain
m=poly.polycompanion(u.coef)                 #matrice compagnon
pprime=poly.polyder(p.coef)                  #array derivée
pprimeint=poly.polyint(pprime)               #array primitive
coef=poly.polyfromroots([5,2,3,6])           #racines aux coefs




## Dictionnaires

dico = {'a':1,'b':2,'c':2}  #creation
dib=dict([(1,'a'),(3,'r')]) #creation autre facon
dico2 = dico.copy()         #copie

dico['d']=4                 #ajout cle : valeur ou modification de la valeur associee a la cle
x = dico['b']               #acces veleur de la cle
del dico['d']               #supprime
y = dico.pop('a')           #supprime l'asociation et renvoye la valeur

# .keys()   .values() et zip
for cle, valeur in zip(dico.keys(),dico.values()):
    print(cle,valeur)

# l'iterable par defaut est celui des cles
for k in dico:
    print(k,dico[k])

#  .items()
for couple in dico.items():
    print (couple)

#parametres en nombre variables pour les fonctions, non nommés puis nommés

def fonction_inconnue(*en_tuple,**en_dico):
    print("j'ai recu {} et {}".format(en_tuple,en_dico))

def fonction_inconnue1(*parametres):#l'e * sert a prevenir que ce qui suit est un tuple de plusieurs parametres
    print("J'ai reçu : {}.".format(parametres))

def fonction_inconnue2(**parametres_nommes): # ** transforme la syntaxe param=valeur en le dictionnaire param : valeur
    print("J'ai reçu en paramètres nommés : {}.".format(parametres_nommes))



## Temps

import time

t1=time.time()#time stamp de l'instant ou elle est appelle ie nb de sec ecoulees depuis 1970
obj = time.localtime()#objet date avec par defaut le timestamp
obj2 = time.localtime(1)#avec 1 sec
t2=time.mktime(obj)#pour recalculer le timestamp a partir de l'objet date

time.sleep(3.5) #pause de 3.5 sec

## random

import random

a=random.random() #entre 0 et 1
b=random.randrange(1,11,3)
c=random.choice(['a','d','f','r','t'])
lst = ['a','d','f','r','t']
random.shuffle(lst)


#####################################################################

## Bibli
from math import *
import numpy as np
import scipy as sc
import sympy as sym
import matplotlib.pyplot as plt


## https://github.com/sympy/sympy/wiki/Tutorial

# built in formal constants
sym.E, sym.I, sym.pi

#create symbols
x=sym.Symbol('x')
y= sym.Symbol('y')

# manipulate expressions: expansion and substition
((x+y)**2).expand()
((x+y)**2).subs(x, 1)
((x+y)**2).subs(x, y)
expr = 3 + x + x**2 + y*x*2

# extract coefficients
a, b = sym.symbols("a, b")
expr.coeff(x,n=0)
expr.as_coefficients_dict()

#differentiate diff(func, var, times)
sym.diff(sym.sin(3*x), x, 2)


#limits, limit(function, variable, point):
sym.limit((sym.tan(x+y)-sym.tan(x))/y, y, 0) #as y goes to 0
sym.limit(1/x, x, sym.oo) #as x goes to infinity


#series expansion, function.series(var, point, order):
sym.cos(x).series(x, 0, 10)

#integration formal and compute definte integral
sym.integrate(sym.log(x), x)
sym.integrate(sym.sin(x), (x, 0, sym.pi/2))

#algebraic equations
sym.solve([sym.Eq(x + 5*y, 2), sym.Eq(-3*x + 6*y, 15)], [x, y])

# pattern matching, returns a dictionary or None
p = sym.Wild('p', exclude=[x])
q = sym.Wild('q', exclude=[x])
(5*x**2 + 3*x).match(p*x**2 + q*x)



## Inversion de Lagrange pour la série B

def lagrange(n):
    f = 2*sym.exp(x)+sym.exp(-x)-x-3
    g = 2*x + 1 - sym.exp(-x)
    res = g
    for k in range(1,n):
        
        res+= sym.Rational(1,factorial(k))*sym.diff(f**k * sym.diff(g, x), x, k-1)
    res=(res-x)/2
    return res.series(x,0,n+1)

#lagrange(10): x + 2*x**2 + 19*x**3/3 + 149*x**4/6 + 1634*x**5/15 + 46061*x**6/90 + 793346*x**7/315 + O(x**8)

def lagrange2(n):
    f = 2*sym.exp(x)+sym.exp(-x)-x-3
    h=f
    g = 2*x + 1 - sym.exp(-x)
    dg=sym.diff(g,x)
    res = g
    for k in range(1,n):
        res+= sym.Rational(1,factorial(k))*sym.diff(h*dg, x, k-1)
        h*=f
    res=(res-x)/2
    return res.series(x,0,n+1)

#lagrange2(8): x + 2*x**2 + 19*x**3/3 + 149*x**4/6 + 1634*x**5/15 + 46061*x**6/90 + 793346*x**7/315 + 16147441*x**8/1260 + O(x**9)

def lagrange4(x,n):
    f = (2*sym.exp(x)+sym.exp(-x)-x-3).series(x,0,n+1)
    h=f
    g = 2*x + 1 - sym.exp(-x)
    dg=(sym.diff(g,x)).series(x,0,n+1)
    res = g.series(x,0,n+1)
    for k in range(1,n):
        res+= (sym.Rational(1,sym.factorial(k))*sym.diff((h*dg).series(x,0,k+n), x, k-1)).series(x,0,n+1)
        h*=f
    res=(res-x)/2
    return res.series(x,0,n+1)

#lagrange4(sym.Symbols('x'),12): x + 2*x**2 + 19*x**3/3 + 149*x**4/6 + 1634*x**5/15 + 46061*x**6/90 + 793346*x**7/315 + 16147441*x**8/1260 + 758401817*x**9/11340 + 10092008627*x**10/28350 + 600357880699*x**11/311850 + 39473091815683*x**12/3742200 + O(x**13)

## Resultats et Verification rayon B: beta =(3-2*sqrt(3)+2*log((1+sqrt(3))/2))

A=[1, 4, 38, 596, 13072, 368488, 12693536, 516718112, 4133744896, 218419723296, 12917771042560, 845303896024192, 60630669028889088]
B=[1, 2*2, 19*3/3, 149*4/6, 1634*5/15, 46061*6/90, 793346*7/315, 16147441*8/1260, 758401817*9/11340, 10092008627*10/28350, 600357880699*11/311850, 39473091815683*12/3742200]

def l(a):
    return log(a)/log(1/(3-2*sqrt(3)+2*log((1+sqrt(3))/2)))

Bl=[l(a) for a in B]

#l1=np.vectorize(l)
#Bl= l1(B)
#plt.plot(Bl)
plt.plot([log(a) for a in A])
plt.show()


## Etude numérique des serie D

#create symbols
x=sym.Symbol('x')
y= sym.Symbol('y')

# Conjecture du premier point critique pour D(x)=y(x):
Eq0=(x**3+x**2)*y**6-x**2*y**5-4*x*y**4+(8*x+2)*y**3-(4*x+6)*y**2+6*y-2
Eq0_y=(x**3+x**2)*6*y**5-x**2*5*y**4-4*x*4*y**3+(8*x+2)*3*y**2-(4*x+6)*2*y+6
DiscE=16*x**10*(175232*x**5+252288*x**4+29128*x**3+41675*x**2+7572*x+324)

#delta est la petite racine réelle de DiscE/x**10
# 0.063321613 < rayon < 0.063321614
# 15.792395 < 1/rayon < 15.792396

##Algorithme de Newton sur D à la main:

Eq=(x**3+x**2)*y**6-x**2*y**5-4*x*y**4+(8*x+2)*y**3-(4*x+6)*y**2+6*y-2
Eq1=Eq.subs(y,1+x*y)
Eq2=Eq.subs(y,1+x+x**2*y)
Eq3=Eq.subs(y,1+x+3*x**2+x**3*y)
Eq4=Eq.subs(y,1+x+3*x**2+15*x**3+x**4*y)
Eq5=Eq.subs(y,1+x+3*x**2+15*x**3+105*x**4+x**5*y)
Eq6=Eq.subs(y,1+x+3*x**2+15*x**3+105*x**4+923*x**5+x**6*y)
Eq7=Eq.subs(y,1+x+3*x**2+15*x**3+105*x**4+923*x**5+9417*x**6+x**7*y)
Eq8=Eq.subs(y,1+x+3*x**2+15*x**3+105*x**4+923*x**5+9417*x**6+105815*x**7+x**8*y)
Eq9=Eq.subs(y,1+x+3*x**2+15*x**3+105*x**4+923*x**5+9417*x**6+105815*x**7+1267681*x**8+x**9*y)
Eq10=Eq.subs(y,1+x+3*x**2+15*x**3+105*x**4+923*x**5+9417*x**6+105815*x**7+1267681*x**8+15875631*x**9+x**10*y)

AnaLin=[1,1,3,15,105,923,9417,105815,1267681,15875631,205301361]

#def implicite(E,x,y,n) ABANDONNE


## Etude numérique des serie C

EqC0=2*y**3+(x+2)*y**2+(2*x-1)*y+x
EqC0_y=2*3*y**2+(x+2)*2*y+(2*x-1)


## dessins de courbes algébrique singulière plan globale

# https://plot.ly/python/
# https://matplotlib.org/users/pyplot_tutorial.html
# http://apprendre-python.com/page-creer-graphiques-scientifiques-python-apprendre


# equations algebriques

def ellipse(x,y):
    return x**2-4*y**2-4*y

def pertcardio(x,y):
    return (x**2+y**2-2*x)**2-(x**2+y**2)

def heart(x,y):
    return (x*2+y**2)**3-3*x**2*y**3

# parametrisations par t

## Classe des diagrammes de cordes linéaires, génération des analytiques

class DiagLin:
    def __init__(self,data):
        if type(data)==str: #from word to lst and dic
            word=data
            lst=list(word)
            dic=dict([(k,[]) for k in word])
            for (n,k) in enumerate(word):
                dic[k].append(n)
        elif type(data)==list: #from lst to word and dic
            lst=data
            word=''
            for c in lst:
                word+=c
            dic=dict([(k,[]) for k in word])
            for (n,k) in enumerate(word):
                dic[k].append(n)
        elif type(data)==dict: #from dic to lst and word
            dic=data
            lst=[None]*(2*len(dic))
            for k in dic:
                for n in dic[k]:
                    lst[n]=k
            word=''
            for c in lst:
                word+=c
        else: # pas le bon type
            raise TypeError
        for k in dic: # cordiag implique 2 lettres de chaque
            if len(k)!=2:
                raise ValueError("Il y a une chorde à {} brins".format(len(k)) )
        self.word=word
        self.dic=dic
        self.list=lst

    def __len__(self):
        return len(self._cles)
    
    def interlace_graph(self):
        



## Graphes !!!!!

# https://www.python.org/doc/essays/graphs/
# https://www.python-course.eu/graphs_python.php
# https://networkx.github.io/documentation/networkx-1.10/tutorial/tutorial.html
# https://www.tutorialspoint.com/python/python_graphs.htm

    ## Dictionaires, cles = sommets, valeurs = liste/ensemble des voisins
graph = {'A': ['B', 'C'],
             'B': ['C', 'D'],
             'C': ['D'],
             'D': ['C'],
             'E': ['F'],
             'F': ['C']}
    ## chemin de 1 vers 2 dans graph s'il existe par backtracking

def find_path(graph, start, end, path=[]):
        path += [start]
        if start == end:
            return path
        if start not in graph:                              #peut etre qu'un sommet pointe vers un sommet qui n'est pas enregistre en tant que tel comme clé
            return None
        for node in graph[start]:
            if node not in path:                            # on a pas envie de boucler à l'infini donc on visite les pas encore vus
                newpath = find_path(graph, node, end, path) #on essaye en emprutant la voie par node
                if newpath: return newpath                  #if iterable s'evalue en vrai ssi l'iterable est non vide
        return None

find_path(graph, 'A', 'D')

#test booleen sur iterable, entier, None,.. : faux si vide, vrai sinon.
if [] or {} or () or 0 or None: 
    print(1) 
else: 
     print(2)

#evaluation conditionelle compressée
a = ('expression1' if 'test'=='vrai' else 'expression2')


## Calcule tous les chemins de depart a arrivee

def find_all_paths(graph, start, end, path=[]):
        path += [start]       #on ajoute le nouveau sommet au chemin
        if start == end:            #on a trouve un chemin de start a end, on renvoye le chemin trouve, un singleton chemin
            return [path]
        if start not in graph:      # un sommet point vers son voisin start inconnu: impasse
            return []
        paths = []                  #on initialise l'ensemble des chemins elementaires trouves, et qui commencent par path, à vide
        for node in graph[start]:   #on cherche à partir de chacun des voisins
            if node not in path:    #on ne veut pas boucler sur un cycle : chemins élémentaires
                newpaths = find_all_paths(graph, node, end, path) #on regarde les chemins aboutissant a partir du chemin choisi, renvoie une liste de chemins
                paths.extend(newpaths) #on rajoute chaque nvx chemin trouvé a la liste de tous les chemins
        return paths

    ## Calcule chemin plus court

def find_shortest_path(graph, start, end, path=[]):
        path += [start]
        if start == end:
            return path
        if start not in graph:
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = find_shortest_path(graph, node, end, path) #on parcours depuis chacun des voisins
                if newpath:                                          #si on a trouve un nouveau chemin (ie pas None)
                    if not shortest or len(newpath) < len(shortest): #si shortest est None ou si on a trouve un plus court
                        shortest = newpath                           #on remplace shortest par ce plus court
        return shortest

#########################################################
