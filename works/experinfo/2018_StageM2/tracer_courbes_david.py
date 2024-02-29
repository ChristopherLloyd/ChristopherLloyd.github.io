## Createur

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 08:01:08 2018

@author: coulette
"""

## Bibli

import numpy as np
import matplotlib.pyplot as plt

## Fonction de tracer equation

def plot_equations(expr_list,xrangedef,yrangedef,color_list=None,save=False,figfilename='fig.pdf'):
    print(xrangedef)
    xvals = np.linspace(xrangedef[0],xrangedef[1],xrangedef[2])
    yvals = np.linspace(yrangedef[0],yrangedef[1],yrangedef[2])
    X,Y  = np.meshgrid(xvals,yvals)
    fig = plt.figure(figsize=(4,3),dpi=200)
    if (len(color_list)==len(expr_list)):
        for expr,col in zip(expr_list,color_list):
            eq = lambda x,y : eval(expr)
            V = eq(X,Y)
            plt.contour(X,Y,V,[0],colors=col,linewidths=1)
    else:
        for expr in expr_list:
            eq = lambda x,y : eval(expr)
            V = eq(X,Y)
            plt.contour(X,Y,V,[0],colors=col,linewidths=3)
    #plt.grid()
    #plt.xlabel(r'x')
    #plt.ylabel(r'y')
    plt.axis('off')
    if (save):
        plt.savefig(figfilename)
    else:
        plt.show()
    #plt.close(fig)

## Donées des deux graphiques

# graphique 1: singularités locales, fenetre: [-0.3,0.3]^2
# couleurs respectivement: noir vert bleu rouge orange violet
#x,y = sym.Symbol('x'), sym.Symbol('y')
#v0  = y**3-x
#v12 = (x+y)**2-x**8
#v34 = (y-x**2)**2-x**6
#v5 = y**2 + 2*y*x**2 + x**4 + x**5
#v6 = y+x**2
#v7 = y-x

#ellipse=x**2+4*y**2-4*y
#cardioide=(x**2+y**2-2*x)**2-(x**2+y**2)
#coeur=(x**2+y**2-1)**3-3*x**2*y**3

## Tracer graphiques local

xrangdef_graph1 = (-0.3,0.3,6000)
yrangdef_graph1 = (-0.3,0.3,6000)
    
expr_list_graph1=['y**3-x','(x+y)**2-x**6','(y-x**2)**2-x**6','y**2 + 2*y*x**2 + x**4 + x**5','y+x**2','y-x']
clist_graph1=['k','orange','b','r','g','purple']

plot_equations(expr_list_graph1,xrangdef_graph1,yrangdef_graph1,color_list=clist_graph1)

## Tracer graphiques global
# graphique 2: courbes globales fenetre : [-2,4]*[-2,2]
# couleurs respectivement : noir bleu rouge
xrangdef_graph2 = (-2.0,4.0,5000)
yrangdef_graph2 = (-2.0,2.0,5000)
expr_list_graph2 = ['x**2+4*y**2-4*y','(x**2+y**2-2*x)**2-(x**2+y**2)','(x**2+y**2-1)**3-3*x**2*y**3']
clist_graph2=['k','b','r']

plot_equations(expr_list_graph2,xrangdef_graph2,yrangdef_graph2,clist_graph2)
