
import networkx as nx



def get_graph_registry():
    registry = {
        'karate_club': nx.karate_club_graph,
        'florentine_families': nx.florentine_families_graph,
    }
    return registry



def get_graph_by_name(name='karate_club'):
    registry = get_graph_registry()
    if name not in registry.keys():
        raise ValueError(f'`{name}` not one of the options: {registry.keys()}')
    G = registry[name]()
    return G
    



if __name__ == '__main__':

    graph_names = [
        'karate_club',
        'florentine_families'
    ]
    for name in graph_names:
        print(f'------- {name}')
        G = get_graph_by_name(name)
        print(G)
        print(G.nodes)
        print(G.edges)

        key = list(G.nodes)[1]
        print(G.adj[key])
        print(G.degree[key])

        adjacency_matrix = nx.to_numpy_array(G)
        print(adjacency_matrix)
        print(adjacency_matrix.shape)

        #for v in G:
        #    print(v)


