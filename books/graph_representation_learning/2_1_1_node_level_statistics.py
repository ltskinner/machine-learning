
import networkx as nx
import numpy as np

from get_graph import get_graph_by_name


def get_node_degree(A):
    # assumes A is square
    degrees = []
    for u_i, u in enumerate(A):
        # for each node
        d_u = A[u_i, :].sum()
        degrees.append(d_u)
    
    # or
    # degrees = A.sum(axis=1)
    return degrees

def get_node_degree_ones(A):
    # d = A \cdot \bar{1}

    n = A.shape[0]  # assumes square
    ones = np.ones(n, dtype=A.dtype)

    degrees = A.dot(ones)  # A @ ones
    return degrees.tolist()


def test_node_degree(G):
    degrees = []
    for v in G:
        degrees.append(G.degree[v])
    return degrees

def get_directed_degrees(A):

    n = A.shape[0]  # assumes square
    ones = np.ones(n, dtype=A.dtype)


    # out degree = \sum_{v \in V} A[u,v]
    out_degrees1 = A.sum(axis=1)  # rows
    # or
    out_degrees2 = A.dot(ones)  # A @ ones
    assert out_degrees1.tolist() == out_degrees2.tolist()

    # in degree = \sum_{v \in V} A[v,u]
    in_degrees1 = A.sum(axis=0)  # columns
    # or
    in_degrees2 = A.T.dot(ones)  # A @ ones
    assert in_degrees1.tolist() == in_degrees2.tolist()

    return out_degrees2, in_degrees2


def get_node_centrality_iterative(A):
    # aka eigenvector centrality
    # seek the prinicpal eigenvector of A (lambda_max)
    # here we are doing the power method / power iteration

    # start with guess of e^(0)
    # apply e^(k+1) = Ae^(k)
    # normalize and repeat
    # eventually converge to principal eigenvector

    n = A.shape[0]

    # "initial" eigenvector estimate
    e = np.ones(n) / np.sqrt(n)   # start normalized

    max_iter = 100
    tol = 1e-9
    for _ in range(max_iter):
        # with each loop the abs value change should trend toward zero
        e_new = A @ e             # matrix-vector multiply
        e_new = e_new / np.linalg.norm(e_new)

        if np.linalg.norm(e_new - e) < tol:
            break
        e = e_new

    eig_centrality = e
    return eig_centrality

def get_node_centrality_eigen_decomp(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # largest eigenvalue may not be index 0
    idx = np.argmax(eigenvalues.real)

    # real part of principal eigenval
    e = eigenvectors[:, idx].real

    # normalize
    e = e / np.linalg.norm(e)
    eig_centrality = e
    return eig_centrality



if __name__ == '__main__':
    G = get_graph_by_name('florentine_families')
    print(G.nodes)

    peruzzi = 'Peruzzi'    # 3
    guadagni = 'Guadagni'  # 12

    print(G.degree[peruzzi], G.degree[guadagni])

    A = nx.to_numpy_array(G)
    expected = test_node_degree(G)
    print(expected)
    degrees = get_node_degree(A)
    assert expected == degrees

    degrees = get_node_degree_ones(A)
    assert expected == degrees

    get_directed_degrees(A)


    eig_centrality1 = get_node_centrality_iterative(A)
    print(eig_centrality1)

    eig_centrality2 = get_node_centrality_eigen_decomp(A)
    assert np.round(eig_centrality1, decimals=2).tolist() == np.round(eig_centrality2, decimals=2).tolist()

    print(round(eig_centrality2[3], 2), round(eig_centrality2[12], 2))

