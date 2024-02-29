##
from math import *
import numpy as np
import matplotlib.pyplot as plt
import time

## working

def partitions_gen_rec(n):
    if n == 0:# cas de base de la récursion: 0 est la somme de la liste vide
        yield []
        return
    for p in partitions_gen_rec(n-1):# modifier partitions de n-1 pour fabriquer partitions de n
        yield [1] + p 
        if p and (len(p) <= 1 or p[1] > p[0]):#if [] : False ; if [a,...]: True
            yield [p[0] + 1] + p[1:]

## Explanation
#If you have a partition of n, you can reduce it to a partition of n-1 in a canonical way by subtracting one from the smallest item in the partition. E.g. 1+2+3 => 2+3, 2+4 => 1+4. This algorithm reverses the process: for each partition p of n-1, it finds the partitionS of n that would be reduced to p by this process. Therefore, each partition of n is output exactly once, at the step when the partition of n-1 to which it reduces is considered.

##

def parcourir_iter(L):
    res=[]
    for k in L:
        res.append(k)
    return(res)

##

def chronometrer(myfunc,n):
    a = time.clock()
    myfunc(n)
    b = time.clock()
    return b-a