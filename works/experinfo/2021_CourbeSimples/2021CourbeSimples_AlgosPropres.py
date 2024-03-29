

###### ###### MULTILOOPS IN SURFACES ###### ######

"""
    This module deals with multiloops in a surface of genus g>0 and 1 puncture.
    
    The surface of genus g>0 (with 1 puncture), is obtained by gluing a (truncated) 4g-gon.
    The identification pattern of the 4g-gon is encoded by a chord diagram :
    a cyclic word in which each letter appears twice (lower and upper case).
    For instance 'abABcdCD...' is the standard chodiag for a closed surface.
    A chord diagram for a sphere with k-holes is 'aAbBcCdD...'.

    A loop is given from its cutting sequence wrt the chord diagram, or equivalently 
    as a product in the standard presentation of the fundamental group, 
    it is thus encoded as a linear word in the letters a, A, b, B, etc.
    A multiloop is a collection of loops, encoded as a sequence of loops.
    
    The main functions of this module are :
    
    multicurve_basis_decomposition : list of str  --> list of list of str
    Given a multiloop in a closed surface minus 0 or 1 point, 
    it decomposes its trace function as a linear combination of multicurves.
    
    chordiagram :
    
    intersection_matrix :
    
    
"""

###### LIBRARIES ######

import numpy as np
import matplotlib.pyplot as plt
# from typing import List, Callable, Tuple


###### ###### DECOMPOSITION OF MULTILOOPS IN THE MULTICURVE BASIS ###### ######

"""
    The main function of this section is multicurve_basis_decomposition.
    
    It relies on smooth_an_intersection and proceeds inductively.

    The function smooth_an_intersection relies on linked_words 
    The function linked_words relies on on sing_cross_ratio

"""


"""
    The function cyclorder(x,y,z) takes three elements x, y, z 
    in a (cyclic) poset and returns their cyclic order 1,0,-1
    
    The function across(W,Z,cycle) returns 2,1,0,-1,-2 which equals 
    double the symplectic intersection number of the chords W and Z 
    taken in this order.
    
"""

def cyclorder(x,y,z):
    return np.sign((z-y)*(z-x)*(y-x))

def across(W, Z, cycle):
    pu = cycle.index(W[0])
    pv = cycle.index(W[1])
    px = cycle.index(Z[0])
    py = cycle.index(Z[1])
    return(cyclorder(pu,px,pv)-cyclorder(pu,py,pv))

#"""
#    The function sign_across_ratio returns 1,0,-1 
#    depending on whether the chords from W[0] to W[1] and from Z[0] to Z[1] :
#    are disjoint, share one point on the boundary, or in the interior.
#    This is given by the sign of a cross ratio.
#    The cycle pattern is assumed injective (each element appears once).
#"""

#def sign_across_ratio(W, Z, cycle):
#    pu = cycle.index(W[0])
#    pv = cycle.index(W[1])
#    px = cycle.index(Z[0])
#    py = cycle.index(Z[1])
#    return np.sign((pv-pu)*(pv-px)*(py-pu)*(py-px))


"""
    The function is_linked_pair returns 1,0,-1 depending on whether 
    w1[s1-1], w2[s2-1], w1[s1:], w2[s2:] are linked on the boundary 
    the linking being defined with respect to the cycle.
    
    This corresponds to the relative position of the axes associated to 
    the infinite linear words with ...+w1+(w1[:s1]) extending to the left 
    and w1[s1:]+w1+... extending to the right.
    
    The answer is 1 if they cross, -1 if they are disjoint, 0 if tangent.
    (This can be determined by looking only length(l1)+length(l2) letters 
    since u^infty = v^infty iff uv = vu.)
"""


def is_linked_pair(word1, word2, s1, s2, cycle):

    w1s = word1[s1-1].swapcase()
    w2s = word2[s2-1].swapcase()

    if w1s == w2s :
        return 0

    k = 0
    while((word1[(s1+k)%len(word1)] == word2[(s2+k)%len(word2)]) &\
        (k <= (len(word1)+len(word2)))):
        k += 1

    w1t = word1[(s1+k)%len(word1)]
    w2t = word2[(s2+k)%len(word2)]

    return abs(across((w1s, w1t), (w2s, w2t), cycle))-1

"""
    The function find_intersection inspects linear cyclic permutations of words
    and returns the first pair of indices (s1, s2) such that :
    w1[s1-1], w2[s2-1], w1[s1:], w2[s2:] are linked on the boundary 
    the linking being defined with respect to the cycle.
    If the loops do not intersect it returns None.
    
"""


def find_intersection(word1, word2, cycle):

    for s1 in range(len(word1)):
        for s2 in range(len(word2)):
            if is_linked_pair(word1, word2, s1, s2, cycle)==1:
                return(s1, s2)

    return(None)

"""
    The function find_intersecting_loops finds two intersecting loops in the multiloop
    and returns the indices of the loops followed by the indices of the linked pairs.
"""

def find_intersecting_loops(multiloop, cycle):
    for i, loopi in enumerate(multiloop):
        for j, loopj in enumerate(multiloop):
            fi = find_intersection(loopi, loopj, cycle)
            if not fi :
                continue
            else : 
                return((i,j), fi)
    return(None)

"""
    The function reduce(word) returns the reduced cyclic representative 
    in the free group generated by the letters of the word.
    
    The function inverse(word) returns the inverse of an element,
    in the free group on small letters whose inverse are capitals.
"""

def simplification(word):
    for p, char in enumerate(word):
        if word[p-1] == char.swapcase():
            return(p)
    return None

def reduce(word):
    p = simplification(word)
    if p == None:
        return word
    elif p == 0:
        return reduce(word[1:-1])
    else:
        return reduce(word[(p+1):]+word[:(p-1)])

def inverse(word):
    return word[::-1].swapcase()


"""
    The function smooth_intersection takes a multiloop : list of str
    along with the cycle pattern, and returns a list of list of str
    with two new multiloops obtained by smoothing an intersection.
    
    If there is no intersection, the first line raises a TypeError.
    This TypeError will be catched in the multicurve_basis_decomposition.
"""

def smooth_intersection(multiloop, cycle):
    
    ((i,j), (si, sj)) = find_intersecting_loops(multiloop, cycle)
    
    if i==j :
        loop = multiloop.pop(i)
        si, sj = min(si,sj), max(si,sj)
        
        sloop0 = loop[si:sj]
        sloop2 = loop[sj:] + loop[:si]
        sloop1 = sloop0 + inverse(sloop2)
        
        sm1 = multiloop + [reduce(sloop1)]
        sm2 = multiloop + [reduce(sloop0), reduce(sloop2)]

    else :
        loopi = multiloop[i]
        loopj = multiloop[j]
        del mulitloop[max(i,j)]
        del multiloop[min(i,j)]
        
        cloopi = loopi[si:]+loopi[:si]
        cloopj = loopi[sj:]+loopj[:sj]
        
        sloop1 = cloopi + cloopj
        sloop2 = cloopi + inverse(cloopj)
        
        sm1 = multiloop + [reduce(sloop1)]
        sm2 = multiloop + [reduce(sloop2)]

    return (sm1, sm2)


"""
    The function multicurve_basis_decomposition : list of str  --> list of list of str
    takes a multiloop in a genus g surface minus 1 point (a cycle of 2g-identifications)
    and decomposes its trace function in the linear basis of multicurves.
"""

def multicurve_basis_decomposition(multiloop, cycle):
    
    multiloops = [multiloop]
    multicurves = []
    
    while multiloops:
        multiloopop = multiloops.pop()
        try:
            smoothings = smooth_intersection(multiloopop, cycle)
            multiloops.extend(smoothings)
        except TypeError:
            multicurves.append(multiloopop)
    
    return multicurves

###### ###### SUPPLEMENTARY FUNCTIONS ###### ######

""" 
    The function list_linked_pairs returns the list of all such pairs.
    The function intersection number computes the number of linked pairs.
    The function intersection_matrix returns the intersection matrix of a multiloop.
"""

def list_linked_pairs(word1, word2, cycle):

    pairs = []

    for s1 in range(len(word1)): 
        for s2 in range(len(word2)):
            if is_linked_pair(word1, word2, s1, s2, cycle)==1:
                pairs.append((s1, s2))

    return(pairs)

def intersection_number(word1, word2, cycle):
    return len(list_linked_pairs(word1, word2, cycle))

def intersection_matrix(multiloop, cycle):
    nloops = len(multiloop)
    mat = np.zeros(nloops, dtype=int)
    for i in range nloops:
        for j in range nloops:
            mat[i,j]=intersection_number(multiloop[i], multiloop[j], cycle)
    return mat

"""
    The function is_primitive returns True if the word is primitive
    and the primitive root together with the power if not.
"""

def is_primitive(loop):
    for k in range(1,len(loop)):
        if loop[k:]+loop[:k] == loop:
            return(loop[:k], len(loop)//k)
    return True

"""
    The function perm_cycle of an iterable returns 
    the list of all its cyclic permutations.
"""

def perm_cycle(word):
    return ["".join([word[i - j] for i in range(len(word))]) \
            for j in range(len(word),0,-1)]

"""
    The function is_puncture returns the boolean telling whether 
    the loop is a cyclic permutation of the cycle or its inverse
    (and thus encircles the puncture once)
"""

def is_puncture(loop, cycle):
    return loop in perm_cycle(cycle) + perm_cycle(inverse(cycle))


###### ###### CHORDIAGRAM OF LOOP AND INTERLACE GRAPHS OF MULTILOOPS ###### ######

"""
    The function chordiag computes the chord diagram of a multiloop.
    The function chordiag of a multiloop
"""

def chordiag_loop(loop, cycle):
    n = len(loop)
    cycloops = perm_cycle(loop)
    pass

def chordiag_multiloop(loop, cycle):
    n = len(loop)
    cycloops = perm_cycle(loop)
    pass

###### ###### THE PUNCTURED TORUS ###### ######

""" !!! IN THIS SECTION THE CYCLE IS 'abAB' !!! """

"""
    The function pentore returns the pair of integers corresponding to the slope 
    of the loop in the once-punctured torus (assuming it has no intersections). 
"""

def pentore(loop):
    na=loop.count("a")
    nb=loop.count("b")
    nA=loop.count("A")
    nB=loop.count("B")
    return (na-nA,nb-nB)

"""
    The function param_multicurve returns ((sa,sb)), no, n0) where
    n0 is the number of trivial loops in the multicurve
    no is the number of boundary components in the multicurve
    sa,sb is the total slope of the non-boundary components of 'multicurve'.
"""

def param_multicurve(multicurve, cycle = 'abAB'):
    n0, no = 0,0
    sa, sb = 0,0

    for curve in multicurve:
        if not curve:
            n0+=1
        elif is_puncture(curve, cycle):
            no+=1
        else:
            na,nb = pentore(curve)
            sa+=np.sign(nb)*na
            sb+=np.abs(nb)
    
    return((sa,sb), no, n0)

"""
    The function affiche_loop takes a multiloop on the once-punctured torus and 
    prints the Newton polygon given by its decomposition in the multicurve basis.
"""


def NewPol_multiloop(multiloops, cycle='abAB'):
    
    multicurves = multicurve_basis_decomposition(multiloops, cycle)
    
    nuage_X=[]
    nuage_Y=[]
    
    for multicurve in multicurves:
        (sa,sb) = param_multicurve(multicurve, cycle)[0]
        nuage_X.append(sa)
        nuage_Y.append(sb)
    
    plt.scatter(nuage_X,nuage_Y)
    plt.show()
    
    return list(zip(nuage_X, nuage_Y))


