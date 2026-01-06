
import networkx as nx
import numpy as np

import random

from get_graph import get_graph_by_name


def simple_overlap(G):
    """Just counts the number of neighbors the two nodes share
    """

    n_nodes = len(G)

    
    S = np.zeros((n_nodes, n_nodes))
    for i, u in enumerate(G):
        for j, v in enumerate(G):
            if j < i:
                # stepping down i
                # if j is ever lower than i, we have already processed
                # at least once, so skip over
                continue
                

            if u == v:
                # how to handle?
                # should just be count of nodes in nb right
                # I feel like I would prefer that than a diagonal of 0s

                S[i, i] = len(set(G.neighbors(u)))
                continue

            Nu = set(G.neighbors(u))
            Nv = set(G.neighbors(v))
            common = len(Nu & Nv)
            # assuming symmetric
            S[i, j] = common
            S[j, i] = common
    
    return S


def simple_overlap2(G):
    """Just counts the number of neighbors the two nodes share
    """
    nodes = list(G)
    n = len(nodes)

    # precompute
    # (I think just in case nx is fully recomputing)
    N = {
        u: set(G.neighbors(u))
        for u in nodes
    }

    S = np.zeros((n, n))
    for i, u in enumerate(G):
        S[i, i] = len(N[u])
        for j in range(i + 1, n):
            v = nodes[j]
            common = len(N[u] & N[v])
            # assuming symmetric
            S[i, j] = common
            S[j, i] = common
    
    return S




def P_global(u, v, S):
    """
    given neighborhood overlap statistic S
    common to assume the likelihod of an edge (u,v)
    is simply proportional to S[u,v]
    """
    return S[u, v] / S[u].sum()

def P_row(u, v, S):
    """
    given neighborhood overlap statistic S
    common to assume the likelihod of an edge (u,v)
    is simply proportional to S[u,v]
    """
    return S[u, v] / S.sum()


def sorensen_index(G):
    nodes = list(G)
    n = len(nodes)

    N = {
        u: set(G.neighbors(u))
        for u in nodes
    }

    S = np.zeros((n, n))
    for i, u in enumerate(G):
        du = G.degree(u)
        S[i, i] = 2.0 * len(N[u]) / (du + du)
        for j in range(i + 1, n):
            v = nodes[j]
            common = len(N[u] & N[v])

            dv = G.degree(v)
            sorensen = 2.0 * common / (du + dv)
            # assuming symmetric
            S[i, j] = sorensen
            S[j, i] = sorensen
    
    return S


def salton_index(G):
    """Noramlizes by product of the degrees
    """
    nodes = list(G)
    n = len(nodes)

    N = {
        u: set(G.neighbors(u))
        for u in nodes
    }

    S = np.zeros((n, n))
    for i, u in enumerate(G):
        du = G.degree(u)
        # note sqrt(du*du) = du
        S[i, i] = 2.0 * len(N[u]) / du
        for j in range(i + 1, n):
            v = nodes[j]
            common = len(N[u] & N[v])

            dv = G.degree(v)
            salton = 2.0 * common / np.sqrt(du*dv)
            # assuming symmetric
            S[i, j] = salton
            S[j, i] = salton
    
    return S


def jaccard_index(G):
    """Noramlizes by product of the degrees
    """
    nodes = list(G)
    n = len(nodes)

    N = {
        u: set(G.neighbors(u))
        for u in nodes
    }

    S = np.zeros((n, n))
    for i, u in enumerate(G):
        S[i, i] = len(N[u]) / len(N[u])  # should just be 1
        for j in range(i + 1, n):
            v = nodes[j]
            common = len(N[u] & N[v])
            either = len(N[u] | N[v])
            jaccard = common / either
            # assuming symmetric
            S[i, j] = jaccard
            S[j, i] = jaccard
    
    return S


def RA_index(G):
    """
    RA = Resource Allocation index
    
    Main value add here is if neighborhood is zero
    then dont need to compute


    These give more weight to common neighbors that have low degree
    The intuition being that shared low-degree neighbor
    is more informative than a shared high-degree neighbor
    """
    nodes = list(G)
    n = len(nodes)

    N = {u: set(G.neighbors(u)) for u in nodes}
    deg = dict(G.degree())

    S = np.zeros((n, n))
    for i, u in enumerate(nodes):
        #S[i, i] = len(N[u])
        for j in range(i + 1, n):
            v = nodes[j]
            common = N[u] & N[v]

            if len(common) == 0:
                continue
            
            score = sum(1.0 / deg[w] for w in common)
            S[i, j] = score
            S[j, i] = score
    
    return S


def AA_index(G):
    nodes = list(G)
    n = len(nodes)

    N = {u: set(G.neighbors(u)) for u in nodes}
    deg = dict(G.degree())

    S = np.zeros((n, n))
    for i, u in enumerate(nodes):
        #S[i, i] = len(N[u])
        for j in range(i + 1, n):
            v = nodes[j]
            common = N[u] & N[v]

            if len(common) == 0:
                continue
            
            score = sum(1.0 / np.log(deg[w]) for w in common)
            S[i, j] = score
            S[j, i] = score
    
    return S


if __name__ == '__main__':

    G = get_graph_by_name('florentine_families')
    H = get_graph_by_name('karate_club')

    S1 = simple_overlap(G)
    S = simple_overlap2(G)
    print(S1)
    print('----------------')
    print(S)
    np.testing.assert_array_equal(S1, S)

    nodes = list(G.nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    for _ in range(5):
        print('-------------------------------')
        u, v = random.sample(nodes, 2)
        u_index = node_to_idx[u]
        v_index =  node_to_idx[v]
        print(u, '<-?->', v)
        p_global = P_global(u_index, v_index, S)
        print('global:', p_global)

        p_row = P_row(u_index, v_index, S)
        print('row:', p_row)

    
    print('Sorensen:')
    S_sorensen = sorensen_index(G)
    print(S_sorensen)
    print(S_sorensen.shape)

    S_salton = salton_index(G)
    print('Salton:')
    print(S_salton)

    S_jaccard = jaccard_index(G)
    print('Jaccard:')
    print(S_jaccard)

    S_ra = RA_index(G)
    print('RA:')
    print(S_ra)

    S_aa = AA_index(G)
    print('AA:')
    print(S_aa)


