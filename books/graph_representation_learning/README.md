# Graph Representation Learning

[https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)

## [Chapter 1: Introduction](./CHAPTER_1.md)

| Term | Definition | Example |
| - | - | - |
| `multi-relational graph` | different relation types as a "layer" of edges, summarized as an `adjacency tensor` | $A \in \mathbb{R}^{\|V\| \times \|R\| \times \|V\| } $ |
| `heterogeneous graphs` | nodes with multiple types | $V = V_1 \cup V_2 \cup ... \cup ... V_{k} $ where $V_i \cap V_j = \emptyset, \forall i \neq j $. the most common constraint is that certain edges only connect nodes of certain types, i.e. $(u, \tau_{i}, v) \in E \implies u \in V_j, v \in V_k $ |
| `multi-partite graphs` | special case of heterogeneous graphs where edges can only connect nodes that have different types | $(u, \tau_{i}, v) \in E \implies u \in V_j, v \in V_k \land j \neq k $ |
| `multiplex graphs` | graphs decomposed into **k layers**; each node is assumed to belong to each layer; each layer corresponds to a unique relation type |  |
| `attribute` | = feature of information associated with a graph |  |
| `problem frames` | node classification; relation prediction; clustering/community detection; graph classification; graph clustering |  |
| `node classification` | predict label $y_u $ for all nodes $u \in V $ |  |
| `homophily` | tendency for nodes to share attributes with their neighbors in the graphs |  |
| `structural equivalence` | nodes in similar local neighborhood structures will have similar labels |  |
| `heterophily` | presumes that nodes will be preferentially connected to nodes with different labels |  |
| `relation prediction` | use partial information to infer missing edges $E \setminus E_{train} $ | aka: link prediction, graph completion, relational inference |
| `community detection` | infer latent community structures given only the input graph $G = (V, E) $ |  |
| `graph classification` | given multiple different graphs, make independent predictions specific to each graph (such as malicious software classification) |  |
| `graph clustering` | learn an unsupervised measure of similarity between pairs of graphs |  |

## [Chapter 2: Background and Traditional Approaches](./CHAPTER_2.md)

| Term | Definition | Example |
| - | - | - |
|  |  |  |
