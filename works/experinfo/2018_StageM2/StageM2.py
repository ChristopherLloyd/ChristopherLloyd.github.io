## Bibli
from math import *
import numpy as np
import scipy as sc
import sympy as sym
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\Christopher\\Desktop\\Informatique\\ClassesFonctionsUtiles')
from arithmetique import memoise, pascal
from graphes import Graph

## Intro Sympy: https://github.com/sympy/sympy/wiki/Tutorial

# built in formal constants
sym.E, sym.I, sym.pi

#create symbols
x=sym.Symbol('x')
y= sym.Symbol('y')
A=sym.Symbol('A')
z=sym.Symbol('z')

# manipulate expressions: expansion and substition
((x+y)**2).expand()
((x+y)**2).subs(x, 1)
((x+y)**2).subs(x, y)
expr = 3 + x + x**2 + y*x*2
EqA=(z**3+z**2)*A**6-z**2*A**5-4*z*A**4+(8*z+2)*A**3-(4*z+6)*A**2+6*A-2
EqAmoins1=EqA.subs(A,1+y).expand()

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

def lagrangeB(n):
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

#lagrangeB(8): x + 2*x**2 + 19*x**3/3 + 149*x**4/6 + 1634*x**5/15 + 46061*x**6/90 + 793346*x**7/315 + 16147441*x**8/1260 + O(x**9)

def lagrangeB_David(x,n):
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

#lagrangeB_David(sym.Symbols('x'),12): x + 2*x**2 + 19*x**3/3 + 149*x**4/6 + 1634*x**5/15 + 46061*x**6/90 + 793346*x**7/315 + 16147441*x**8/1260 + 758401817*x**9/11340 + 10092008627*x**10/28350 + 600357880699*x**11/311850 + 39473091815683*x**12/3742200 + O(x**13)



B_n=[1, 2, 19/3, 149/6, 1634/15, 46061/90, 793346/315, 16147441/1260, 758401817/11340, 10092008627/28350, 600357880699/311850, 39473091815683/3742200]

Buiss=[1, 4, 38, 596, 13072, 368488, 12693536, 516718112, 24268858144, 1291777104256, 76845808729472, 5052555752407424]


## Resultats et Verification rayon B: beta =(3-2*sqrt(3)+2*log((1+sqrt(3))/2))
beta =(3-2*sqrt(3)+2*log((1+sqrt(3))/2))

def l(a):
    return -log(a)/log(beta)

plt.plot([log(b) for b in Buiss])
plt.show()


## Etude numérique des serie C

#EqC0=2*y**3+(x+2)*y**2+(2*x-1)*y+x
#EqC0_y=2*3*y**2+(x+2)*2*y+(2*x-1)

'''calcule le nombre de diagrammes de cordes analytiques connexes enracines, par taille = nombre de cordes non enracines > 0'''
def dace(n):
    res=0
    for k in range(n):
        res+=pascal(n-1,n-1+k)*pascal(n-1-k,2*n+k)*2**k
    return res//n

# [1, 4, 27, 226, 2116, 21218, 222851, 2420134, 26954622, 306203536, 3534170486, 41326973520]


## Etude numérique de la serie D

#create symbols
x=sym.Symbol('x')
y= sym.Symbol('y')
m= sym.Symbol('m')

# Conjecture du premier point critique pour D(x)=y(x):
Eq0=(x**3+x**2)*y**6-x**2*y**5-4*x*y**4+(8*x+2)*y**3-(4*x+6)*y**2+6*y-2
Eq0_y=(x**3+x**2)*6*y**5-x**2*5*y**4-4*x*4*y**3+(8*x+2)*3*y**2-(4*x+6)*2*y+6
Eq0_y_test=sym.diff(Eq0, y, 1)
test=(Eq0_y-Eq0_y_test).expand()
DiscE=16*x**10*(175232*x**5+252288*x**4+29128*x**3+41675*x**2+7572*x+324)

Eq0m=(1-16*x)**6*Eq0.subs(y,1/(1-16*x)-m)

## calcul de a_0
alpha_small=0.063321613
alpha_large=0.063321614
tau_small=1.08698766
tau_large=1.086987665

G=-(Eq0-6*y)/6

G_yy=sym.diff(G, y, 2)
G_x=sym.diff(G, x, 1)

Gyy=-0.0542425 #G_yy.subs([(x,alpha_large),(y,tau_small)])
Gx= -0.000132727
a_0=sqrt(alpha_large*Gx/(Gyy*2*np.pi))


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

## Asymptotique des coefficients de C

import sympy.polys.polytools as sp
from sympy import poly
from sympy.abc import x,y

# Create symbols
x=sym.Symbol('x')
y= sym.Symbol('y')

# Equation
eqC=2*y**3+(x+2)*y**2+(2*x-1)*y+x
peqC=sp.poly_from_expr(eqC)
disc=sp.Poly(eqC, y).discriminant()



## Definitions des courbes algébrique à tracer

# https://plot.ly/python/
# https://matplotlib.org/users/pyplot_tutorial.html
# http://apprendre-python.com/page-creer-graphiques-scientifiques-python-apprendre

#create symbols
x=sym.Symbol('x')
y= sym.Symbol('y')

#equations local
v0  = y**3-x
v12 = (x+y)**2-x**8
v34 = (y-x**2)**2-x**6
v5 = y**2 + 2*y*x**2 + x**4 + x**5
v6 = y+x**2
v7 = y-x
#equations global
ellipse = x**2-4*y**2-4*y
pertcardio= (x**2+y**2-2*x)**2-(x**2+y**2)
heart=(x*2+y**2)**3-3*x**2*y**3


# tracé
X=np.linspace(-0.3,0.3,1+30*10)
P0= [(x0,y0) for x0 in X for y0 in sym.solve(v0.subs(x,x0),y)]
#P12=[(x0,y0) for x0 in X for y0 in sym.solve(v12.subs(x,x0),y)]
#P34=[(x0,y0) for x0 in X for y0 in sym.solve(v34.subs(x,x0),y)]
#P5= [(x0,y0) for x0 in X for y0 in sym.solve(v5.subs(x,x0),y)]
#P6= [(x0,y0) for x0 in X for y0 in sym.solve(v6.subs(x,x0),y)]
#P7= [(x0,y0) for x0 in X for y0 in sym.solve(v7.subs(x,x0),y)]

## Tracé des courbe algébriques

for p in P0:
    plt.plot(p)

## Generation des diagrammes ou diagrammes analytiques:

'''
Remarque: un diagramme de corde c'est un mot de Dick dont les parentheses fermantes sont nuérotées de telle sorte que le numéro de chaque parenthese fermante est supérieur ou egal au nombre de parentheses ouvrantes qui le précède
'''

def gen_cordiag(n):
    if n==0:
        yield []
        return
    for k in range(2*n-1):
        for subdiag in gen_cordiag(n-1):
            yield([n]+subdiag[:k]+[n]+subdiag[k:])


## Test si diagramme est anlytique
'''
On  parcours les listes à l'envers car la methode pop est en O(fin):
on pop le plus loin d'abord sinon ca perturbe les indices.
ATTENTION  ce programme modifie l'entrée.
'''


def analyDiag(l):
    if len(l)<10: return True
    length = len(l)
    for k in range(length-1,-1,-1): # isolé
        if l[k-1]==l[k]:
            l.pop(k),l.pop(k-1)
            return analyDiag(l)
    for k in range(length-2,-1,-1): # fourche
        if l[k-1]==l[k+1]:
            l.pop(k+1),l.pop(k-1)
            return analyDiag(l)
    for k in range(length-1,0,-1):
        for j in range(k-3,-1,-1):
            if (l[j]==l[k] and l[j+1]==l[k-1]):# faux jumeaux (paralleles)
                l.pop(k),l.pop(j)
                return analyDiag(l)
            elif (l[j+1]==l[k] and l[j]==l[k-1]): #vrai jumeaux (se croisent)
                l.pop(k),l.pop(j+1)
                return analyDiag(l)
    return False

## Calcul du nombre d'analitiques linéaires

def ACD(n):
    r=0
    d=None
    for diag in gen_cordiag(n):
        d=diag.copy()
        if analyDiag(d):
            r+=1
    return r

#ACD(n) : 1,1,3,15,105,923,9417,105815,1267681
# La serie génératrice est donc correcte

## Non analytiques pour 5

dic5=dict((tuple(diag),analyDiag(diag)) for diag in gen_cordiag(5))


for (k,v) in dic5.items():
    if not v:
        print(k)

## de Gauss

'''attention, ne fonctionne que pour les diagrammes engendrés par gen_cordiag
dont les premières rencontres sont dans l'ordre decroissant'''

def isGauss(d):
    n,n2=len(d)//2,len(d)
    if n<2:
        return True
    elif d==(2*[k for k in range(n,0,-1)]):
        return False
    else:
        ex=d.index(n,1,n2)
        if ex<2*n:
            d1=d[1:ex]
            d2=d[ex+1:]
            d3=d1.copy()
            d3.reverse()
            return isGauss(d1+d2) and isGauss(d1+d3)
        else:
            return isGauss(d[1:ex])

def Gauss(n):
    r=0
    for d in gen_cordiag(n):
        if isGauss(d):
            r+=1
    return r





















