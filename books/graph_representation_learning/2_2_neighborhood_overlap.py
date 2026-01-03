
import networkx as nx
import numpy as np

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




def simple_edge_likelihood(G):
    """
    given neighborhood overlap statistic S
    common to assume the likelihod of an edge (u,v)
    is simply proportional to S[u,v]
    """
    S = simple_overlap2(G)

    P = np.zeros(S.shape)




if __name__ == '__main__':

    G = get_graph_by_name('florentine_families')
    H = get_graph_by_name('karate_club')

    S1 = simple_overlap(G)
    S = simple_overlap2(G)
    print(S1)
    print('----------------')
    print(S)
    np.testing.assert_array_equal(S1, S)

