## Bibli
from math import *
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

## Definitions des courbes algébrique à tracer

#create symbols
x=sym.Symbol('x')
y= sym.Symbol('y')

# graphique 1: singularités locales, fenetre: [-0.3,0.3]^2
# couleurs respectivement: noir vert bleu rouge orange violet
v0  = y**3-x
v12 = (x+y)**2-x**8
v34 = (y-x**2)**2-x**6
v5 = y**2 + 2*y*x**2 + x**4 + x**5
v6 = y+x**2
v7 = y-x

# graphique 2: courbes globales fenetre : [-2,4]*[-2,2]
# couleurs respectivement : noir bleu rouge
ellipse = x**2+4*y**2-4*y
pertcardio= (x**2+y**2-2*x)**2-(x**2+y**2)
coeur=(x*2+y**2-1)**3-3*x**2*y**3


## tracé: problème avec les nombres complexes, et je m'y prends mal

#X=np.linspace(-0.3,0.3,1+30*10)
#P0= [(x0,y0) for x0 in X for y0 in sym.solve(v0.subs(x,x0),y)]
#P12=[(x0,y0) for x0 in X for y0 in sym.solve(v12.subs(x,x0),y)]
#P34=[(x0,y0) for x0 in X for y0 in sym.solve(v34.subs(x,x0),y)]
#P5= [(x0,y0) for x0 in X for y0 in sym.solve(v5.subs(x,x0),y)]
#P6= [(x0,y0) for x0 in X for y0 in sym.solve(v6.subs(x,x0),y)]
#P7= [(x0,y0) for x0 in X for y0 in sym.solve(v7.subs(x,x0),y)]

