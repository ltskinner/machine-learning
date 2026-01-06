
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









if __name__ == '__main__':

    G = get_graph_by_name('florentine_families')
    H = get_graph_by_name('karate_club')

    S_katz = katz_index(G, b=0.01)
    print('S_katz:')
    print(S_katz)

    S_LHN = LHN_similarity(G, b=0.01)
    print('S_LHN:')
    print(S_LHN)



