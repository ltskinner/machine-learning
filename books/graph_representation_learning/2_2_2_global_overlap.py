
import networkx as nx
import numpy as np

import random

from get_graph import get_graph_by_name


def katz_index(G, b=0.01):
    """
    b controls how much weight is given to short vs long paths
        - small b < 1 down weights the importance of long paths

    note that typically abs(b) < 1 / rho(A)
        - where rho(A) is the spectral radius (largest abs eigenvalue)
        - if this is not true, then I - bA may be singular or numericall unstable
    
    nx.to_numpy_array(G)
        produces a dense array

    nx.adjacency_matrix(G)
        produces a sparse matrix
        saves significant memory by only storing non-zero entreis
        use on massive graphs
    
    """
    A = nx.to_numpy_array(G, dtype=float)

    assert A.shape[0] == A.shape[1]
    I = np.identity(A.shape[0])

    S_katz = np.linalg.inv(I - b*A) - I
    return S_katz


def LHN_similarity(G, b=0.01):
    A = nx.to_numpy_array(G, dtype=float)
    n = A.shape[0]

    w, v = np.linalg.eig(A)
    lambda_1 = np.max(np.abs(w))

    nodes = list(G.nodes())
    D = np.zeros((n, n))
    for i, u in enumerate(nodes):
        D[i,i] = G.degree(u)

    I = np.eye(n)

    m = len(G.edges())

    alpha = 1  # just a global normalization constant, so 1 is fine

    #Dinv = np.linalg.inv(D)
    Dinv = np.linalg.pinv(D)  # mp handles zeros better

    S_LNH = (2 * alpha * m * lambda_1) * (
        Dinv @ np.linalg.inv(I - (b / lambda_1) *A) @ Dinv
    )

    return S_LNH



def random_walk_similarity(G, u_index, v_index):
    A = nx.to_numpy_array(G, dtype=float)
    n = A.shape[0]

    I = np.eye(n)

    nodes = list(G.nodes())
    D = np.zeros((n, n))
    for i, w in enumerate(nodes):
        D[i,i] = G.degree(w)
    

    Dinv = np.linalg.pinv(D)  # mp handles zeros better

    P = A @ Dinv


    c = .85 # 1/n  # "not sensible" lmao

    #core = (1 - c)*np.linalg.pinv(I - c*P)

    eu = np.zeros(n)
    eu[u_index] = 1
    #qu = core @ eu

    # see: https://stackoverflow.com/questions/31256252/why-does-numpy-linalg-solve-offer-more-precise-matrix-inversions-than-numpy-li
    qu = (1 - c) * np.linalg.solve(I - c*P, eu)
    # this is n-dimensional

    ev = np.zeros(n)
    ev[v_index] = 1
    #qv = core @ ev
    qv = (1 - c) * np.linalg.solve(I - c*P, ev)


    S_RW_uv = qu[v_index] + qv[u_index]

    return S_RW_uv





if __name__ == '__main__':

    G = get_graph_by_name('florentine_families')
    H = get_graph_by_name('karate_club')

    S_katz = katz_index(G, b=0.01)
    print('S_katz:')
    print(S_katz)

    S_LHN = LHN_similarity(G, b=0.01)
    print('S_LHN:')
    print(S_LHN)


    nodes = list(G.nodes)
    node_to_idx = {u: i for i, u in enumerate(nodes)}

    for _ in range(5):
        print('-------------------------------')
        u, v = random.sample(nodes, 2)
        u_index = node_to_idx[u]
        v_index =  node_to_idx[v]

        res = random_walk_similarity(G, u_index, v_index)
        print(f'{u_index} <- {res} -> {v_index}')



