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
| `node degree` | first-order graph statistic; indicates **local importance** high degree -> highly connected node; low degree -> peripheral node |  |
| `out degree` | how many edges originate from this node; influencers, senders, initiators, etc |  |
| `in degree` | how many edges point to this node; nodes that receive information, popularity, aggregation points, "consumers" or "sinks" in a flow |  |
| `node centrality` | A more powerful measure of inportance than degree. *centrality* comes in many forms |  |
| `eigenvector centrality` | Node/eigenvector centrality **is** the principal eigenvector of the adjacency matrix (the score of each node comes from the corresponding index in the principal eigenvector) | the centrality of node u is proportional to the sum of the centralities of its neighbors (important neighbors boost your importance) |
| `eigenvector centrality normalization` | 1. makes values numerically stable; 2. provides meaningful scale (probabilities, max=1, etc); 3. Required for iterative algorithms to prevent divergence; 4. Allows comparisons across graphs or time; 5. Ensures a unique interpretable solution |  |
| `clustering coefficient` | Measures the proportion of closed triangles in a nodes local neighborhood | Real world networks tend to have far higher clustering coefficients than would be expected for randomly sampled networks |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

### Stats progression

- 1. degree vector
- 2. Laplacian
- 3. Normalized laplacian
- 4. Random walk transition matrix
- 5. k-step walk probabilities
- 6. Spectral decomposition (fully connects to LA)
- 7. Graph Fourier transform
- 8. Convolutions on graphs
