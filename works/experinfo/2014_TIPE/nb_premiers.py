## bibliotheques

from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##

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

##

def print_prime(n):
    Y=listprime(n)
    Z1=[k*(log(k)+2) for k in range(1,n+1)]
    Z0=[k*log(k) for k in range(1,n+1)]
    X=[k for k in range(1,n+1)]
    plt.plot(X,Y,'ro',X,Z0,'b',X,Z1,'g')
    plt.show()

##

def print_sum_prime(n):
    A=listprime(n)
    Y=[A[0]]
    for k in range(1,len(A)):
        Y.append(Y[-1]+A[k])
    Z=[(k+1)**2/2*(log(k+1)+3/2) for k in range(1,n+1)]
    X=[k for k in range(1,n+1)]
    plt.plot(X,Y,'ro',X,Z,'b')
    plt.show()

##
plt.grid('on')
pas=0.01*np.e
X=np.arange(pas,10,pas)
Y=1/np.log(X)
plt.plot(X,Y)
plt.show()

















