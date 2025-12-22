
import networkx as nx
import numpy as np

from get_graph import get_graph_by_name



def WL(G, K=100):
    # 1. assign an initial label to each node (in most graphs, this is simply the degree)
    labels = {}
    for v in G:
        labels[v] = G.degree[v]
    
    # 2. iteravely assign new label to each node
    # by hashing the multiset of current labels within the nodes neighborhood
    for i in range(1, K):
        signatures = {}

        for v in G:
            nb_labels = tuple(sorted(labels[u] for u in G.neighbors(v)))

            v_prev_label = labels[v]
            signatures[v] = (v_prev_label, nb_labels)

        # you must know all signatures before you can deduplicate them
        hashes = {}
        deduplicated_signatures = set(signatures.values())
        for sig_label_pseudo_hash, signature in enumerate(sorted(deduplicated_signatures)):
            hashes[signature] = sig_label_pseudo_hash

        new_labels = {}
        for v in G:
            new_signature = signatures[v]
            new_labels[v] = hashes[new_signature]

        if new_labels == labels:
            print(f'WL completed after {i} iterations')
            break

        labels = new_labels


    return labels


"""
graphlets


random-walk kernel
shortest path kernel

"""


def binom(n, k):
    return (n * (n-1)) // k


def count_k3_triangle_graphlet_simple(G):
    tri = 0
    for u, v in G.edges():
        tri += len(set(G.neighbors(u)) & set(G.neighbors(v)))
    return tri // 3

def count_k3_triangle_graphlet_ordered(G):
    tri = 0

    index = {}
    for c, u in enumerate(G):
        index[u] = c

    for u, v in G.edges():
        if index[u] < index[v]:
            # ok
            pass
        else:
            # escape to avoid double counting
            continue

        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))
        common = Nu & Nv

        # only count w with index > v to avoid triple counting
        for w in common:
            if index[v] < index[w]:
                tri += 1
            else:
                # do nothing, dont double count
                pass

    return tri


def count_k3_wedge_graphlet(G):
    wedges = 0
    for u, tu in nx.triangles(G).items():

        du = G.degree[u]
        if du < 2:
            continue

        wu = binom(du, 2) - tu
        wedges += wu

    return wedges

def normalize(arr):
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm

def k3_graphlet_kernel(G, H):
    # get_similarity
    # k3_graphlet_kernel
    phi_G = np.array([
        count_k3_triangle_graphlet_simple(G),
        count_k3_wedge_graphlet(G)
    ])
    phi_G_hat = normalize(phi_G)

    phi_H = np.array([
        count_k3_triangle_graphlet_simple(H),
        count_k3_wedge_graphlet(H)
    ])
    phi_H_hat = normalize(phi_H)

    kernel = np.dot(phi_G_hat.T, phi_H_hat)
    # 1.0 -> identical k=3 structure
    # 0.0 -> orthogonal (very different motif composition)
    # -1.0 -> opposute structure (super rare lol)
    return kernel


if __name__ == '__main__':
    G = get_graph_by_name('florentine_families')
    H = get_graph_by_name('karate_club')

    wl_res = WL(G)
    print(wl_res)


    triangles_simple = count_k3_triangle_graphlet_simple(G)
    triangles_ordered = count_k3_triangle_graphlet_ordered(G)
    triangles_official = sum(nx.triangles(G).values()) // 3
    print('triangles:', triangles_simple, triangles_ordered, triangles_official)

    wedges = count_k3_wedge_graphlet(G)
    print('wedges:', wedges)

    kernel = k3_graphlet_kernel(G, H)
    print(kernel)
