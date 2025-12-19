
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


if __name__ == '__main__':
    G = get_graph_by_name('florentine_families')
    #G = get_graph_by_name('karate_club')

    wl_res = WL(G)
    print(wl_res)
