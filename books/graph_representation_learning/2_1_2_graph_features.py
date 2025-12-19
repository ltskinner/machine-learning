
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
            multiset = set()
            nb_labels = sorted(labels[u] for u in G.neighbors(v))

            signatures[v] = (labels[v], tuple(nb_labels))


        hashes = {
            signature: index
            for index, signature in enumerate(sorted(set(signatures.values())))
        }
        new_labels = {v: hashes[signatures[v]] for v in G}

        if new_labels == labels:
            print(f'WL completed after {i} iterations')
            break

        labels = new_labels


    return labels


if __name__ == '__main__':
    G = get_graph_by_name('karate_club')

    wl_res = WL(G)
    print(wl_res)
