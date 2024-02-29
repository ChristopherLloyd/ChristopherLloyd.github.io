## bibliotheques

from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##

def Distrib(n,p):
    X=[]
    Y=[]
    for k in range (1,n//p+1):
        Y.append(log(coBino(k*p,n)*factorial(k*p)/p**k/factorial(k)))
        X.append(k)
    plt.plot(X,Y,'ro')
    plt.show()

##

def compare(n,p):
    W=[]
    X=[]
    Y=[]
    Z=[]
    for k in range(1,n+1):
        X.append(k)
        Y.append(exp(k**(1/p))/p)
        s=0
        for i in range(1,k//p+1):
            s+=factorial(k//p)*p**(k//p)/(factorial(i)*p**i*factorial(k-i*p))
        Z.append(s)
    plt.plot(X,Y,'ro',X,Z,'bo')
    plt.show()

##

def stir(m):
    X=[k for k in range(1,m+1)]
    Y=[log(factorial(k)-sqrt(2*np.pi*k)*(k/e)**(k)) for k in range(1,m+1)]
    plt.plot(X,Y,'ko')
    plt.show()

##

def distrib2(n,p):
    X=[k for k in range(1,n//p+1)]
    Y=[factorial(n//p-k)/factorial(n-k*p) for k in range(1,n//p+1)]
    Z=[p**(-1/2)*(sqrt(2*(n//p)*np.pi)/p**(n//p-k))**p for k in range(1,n//p+1)]
    plt.plot(X,np.log(Y),'ro',X,np.log(Z),'bo')
    plt.show()

##

def primesequ(m):
    L=[2]
    for i in range(1,m):
        L.append(listprime(L[-1])[-1])
    plt.plot([k for k in range(1,m+1)],L,'ro')
    plt.show()
    return(L)
##

def parti(n):
    a=[]
    for k in accelAsc(n):
        a.append(k)
    return(a)
    

##
'''
n=1000000

(sqrt(n*log(n))) =                               3700
(1/2*log(n)**2)  =                               95
(1/2*log(n)**2-sqrt(1/3*log(n**3))) =            92
(1/2*log(n)**2+sqrt(1/3*log(n**3))) =            99

s=0
for k in range(1,1000001):
    s+=log(k)
print(s)
'''
##
plt.subplot(111, projection="3d")
plt.plot([1,2,3],[4,5,6],[7,8,9])
plt.show()

##
x=np.linspace(-1,1,50)
y=np.linspace(-1,1,50)
X,Y=np.meshgrid(x,y)
Z=1/(1+X**2+Y**2)
plt.contourf(X,Y,Z,cmap='hot')
plt.colorbar()
plt.show()

##
ax=plt.subplot(111, projection="3d")
ax.plot_wireframe(X,Y,Z)
plt.show()

##
ax=plt.subplot(111, projection="3d")
ax.plot_surface(X,Y,Z,cmap='binary')
plt.show()















