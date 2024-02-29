## Bibli

import sympy as sym
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np

## Fonctions

def lagrange2(x,n):
    f = 2*sym.exp(x)+sym.exp(-x)-x-3
    h=f
    g = 2*x + 1 - sym.exp(-x)
    dg=sym.diff(g,x)
    res = g
    for k in range(1,n):
        res+= sym.Rational(1,sym.factorial(k))*sym.diff(h*dg, x, k-1)
        h*=f
    res=(res-x)/2
    return res.series(x,0,n+1)

def lagrange3(x,n):
    f = (2*sym.exp(x)+sym.exp(-x)-x-3).series(x,0,n+1)
    h=f
    g = 2*x + 1 - sym.exp(-x)
    dg=(sym.diff(g,x)).series(x,0,n+1)
    res = g.series(x,0,n+1)
    for k in range(1,n):
        res+= (sym.Rational(1,sym.factorial(k))*sym.diff(h*dg, x, k-1)).series(x,0,n+1)
        h*=f
    res=(res-x)/2
    return res.series(x,0,n+1)

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


## Tests

x = sym.Symbol('x')

nmax = 10

dt_lag2 = []
dt_lag3 = []
dt_lag4 = []

for n in range(1,nmax):
    tsl2 = timer()
    r2 = lagrange2(x,n)
    tel2 = timer()
    print(" lagrange2 n = "+str(n)+ " time : "+str(tel2-tsl2))
    dt_lag2.append(tel2-tsl2)
    sym.pprint(r2)
    
    tsl3 = timer()
    r3 = lagrange3(x,n)
    tel3 = timer()
    print(" lagrange3 n = "+str(n)+ " time : "+str(tel3-tsl3))
    dt_lag3.append(tel3-tsl3)
    sym.pprint(r3)

    tsl4 = timer()
    r4 = lagrange4(x,n)
    tel4 = timer()
    print(" lagrange4 n = "+str(n)+ " time : "+str(tel4-tsl4))
    dt_lag4.append(tel4-tsl4)
    sym.pprint(r4)

## graphique performances temporelles

fig =plt.figure()
plt.plot(range(1,nmax),dt_lag2,'-o',label = 'lag2')
plt.plot(range(1,nmax),dt_lag3,'-o',label = 'lag3')
plt.plot(range(1,nmax),dt_lag4,'-o',label = 'lag4')
plt.legend()
plt.grid()
#plt.savefig('lagrange_scaling.pdf')
#plt.close(fig)

fig =plt.figure()
plt.plot(np.log(range(1,nmax)),np.log(dt_lag2)-np.log(dt_lag2[0]),'-o',label = 'lag2')
plt.plot(np.log(range(1,nmax)),np.log(dt_lag3)-np.log(dt_lag2[0]),'-o',label = 'lag3')
plt.plot(np.log(range(1,nmax)),np.log(dt_lag4)-np.log(dt_lag2[0]),'-o',label = 'lag4')
plt.legend()
plt.grid()
#plt.savefig('lagrange_scaling_loglog.pdf')
#plt.close(fig)

plt.show()