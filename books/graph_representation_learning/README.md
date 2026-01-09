# Graph Representation Learning

Neat course: [https://bdpedigo.github.io/networks-course/landing.html](https://bdpedigo.github.io/networks-course/landing.html)

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

"How do we preserve the linear-algebraic meaning of this object without materializing it?"

- assume graphs are sparse unless proven otherwise
  - sparse graphs will not typicall have adjacency matrices
- sparse vs dense
  - 2m / n(n-1)
    - m = edges
    - n = nodes
    - $< 10%$ is sparse
    - $> 10%$ is dense
- adjacency lists enable constructing "rectangular linear operators" that repolace nxn adj matrices while preserving the same algebraic semantics
  - node <-> edge <-> node
- `rectangular linear operators`
  - maps between different spaces like R^3 and R^4

| Term | Definition | Example |
| - | - | - |
| `node degree` | first-order graph statistic; indicates **local importance** high degree -> highly connected node; low degree -> peripheral node |  |
| `out degree` | how many edges originate from this node; influencers, senders, initiators, etc |  |
| `in degree` | how many edges point to this node; nodes that receive information, popularity, aggregation points, "consumers" or "sinks" in a flow |  |
| `node centrality` | A more powerful measure of inportance than degree. *centrality* comes in many forms |  |
| `eigenvector centrality` | Node/eigenvector centrality **is** the principal eigenvector of the adjacency matrix (the score of each node comes from the corresponding index in the principal eigenvector) | the centrality of node u is proportional to the sum of the centralities of its neighbors (important neighbors boost your importance) |
| `eigenvector centrality normalization` | 1. makes values numerically stable; 2. provides meaningful scale (probabilities, max=1, etc); 3. Required for iterative algorithms to prevent divergence; 4. Allows comparisons across graphs or time; 5. Ensures a unique interpretable solution |  |
| `clustering coefficient` | Measures how tightly clustered a nodes neighborhood is. Measures the proportion of closed triangles in a nodes local neighborhood | Real world networks tend to have far higher clustering coefficients than would be expected for randomly sampled networks. Alternatively, its related to the ratio between the actual numbetr of triangles and the total possible triangles of the `ego graph` |
| `ego graph` | the subgraph containing a single node, its neighbors, and all the edges between nodes in its neighborhood |  |
| `graphlet` | Small, induced subgraph (subset of nodes and their direct connections) used as universal building blocks for network analysis, e.g. cycles of particular length | structural significance |
| `graphlet kernel` | Count small subgraph patterns and compare the count vectors |  |
| `motif` | `graphlets` that occur significantly more often in a real network compared to random graphs (there are also `anti-motifs` whic occur significantly less) | have statistical significance |
| `kernel function` | Similarity functions whose pairwise evaluations form a PSD matrix and thus correspond to `inner products` in some feature space | linear, polynomial, RBF/gaussian, laplacian, sigmoid |
| `kernel method` | Learning algorithms reformulated to depend only on inner products, which are in turn replaced by kernel evaluations throughought the model/algorithm |  |
| `Weisfeiler-Lehman Kernel` | Purely discrete method. Creates `multi-sets` of the neighborhoods and either there will be *exactly* identical multi-sets, or there will not be. There is no concept of "similarity" of the intersection of the multiset - things are either identical or they are not |
| `WL Graph Hash` | Combines the node level hashes for a single representation of an entire graph - allows assessing `structural symmetry` between entire graphs |  |
| `graph isomorphism` | Two graphs to have the exact same structure (same number of vertices, edges, connections) (like identical puzzle pieces being viewed at different angles) |  |
| `local overlap measures` | `simple overlap` = $N(u) \cap N(v) $; `Sorensen` = $ \frac{2S}{d_u + d_v}$; `Salton` = $\frac{2S}{\sqrt{d_u d_v}} $; `Jaccard` = $\frac{N(u) \cap N(v)}{N(u) \cup N(v)} $ | These metrics are extemely effective heuristics for link prediction. In many cases, they achieve competitive performance compared to advanced deep learning approaches. The primary limitation is they only consider local node neighborhoods |
| `importance-based local overlap measures` | `Resource Allocation (RA)` = $ S_{RA}[v_1, v_2] = \sum_{u \in N(v_1) \cap N(v_2)} \frac{1}{d_u} $; `Adamic-Adar (AA)` = $S_{AA}[v_1, v_2] = \sum_{u \in N(v_1) \cap N(v_2)} \frac{1}{\log{(d_u)}} $ | Both measures give more weight to common neighbors that have low degree. The intuition is that a shared low-degree neighor is more informative than a shared high-degree neighbor. |
| Value of Global vs Local | In some cases, two nodes may have no local overlap in their neighborhoods but they can be members of the same `community` |  |
| `katz index` |  | Most basic global overlap stat. Here, we count the number of paths *of all lengths* between a pair of nodes. Issue: strongly biased by node degree (higher degree = more paths through it) |
| `Leicht, Holme, and Newmann (LHN) similarity` |  | Alleviates node degree bias by considering the ratio between the actual observed paths and expected paths |
| `stochastic matrix` | Square matrix with non-negative entries, where each row (or column) sums to one. Represents probabilities of transitioning between states in a system |  |
| `stationary probability` | A stationary probability distribution is a probability (vector) that remains unchanged over time. Aka, unchanged under the Markov transition operator, capturing the long-run visitation frequency of each state. | Something like akin to eigenvector centrality as $x = \lambda A x $, but for probability with $\pi $ as a single probability, and $P$ being a `stochastic matrix` $\pi = P\pi $ |
| `unnormalized laplacian` | $L = D - A $. Characteristics: symmetric, PSD, V non-negative eigenvalues |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

### Katz and LNH Specific notes

| Term | Definition | Example |
| - | - | - |
| `series` | a sum of infinitely many terms; infinite summation |  |
| `finite sum` | the sum of k (finite, not infinite) |  |
| `scalar geometric series` | $1 + r + r^2 + r^3 + ... $ |  |
| `geometric series of matrices` | Means we have the $1 + A + A^2 + ... $ but they are matrices instead of some other data structure | Importantly, this means there is a `closed form inverse`, and convergence depends on $\rho (A) < 1 $. Note, this must start at i=0 |
| `walk` | nodes/edges may repeat |  |
| `path` | no repeated nodes (simple path) |  |
| `bipartite graph` | If the nodes of a graph can be split into two disjoint sets $V = V_1 \cup V_2 $ and $V_1 \cap V_2 = \emptyset $ | `Bipartite graphs` = no odd cycles. Bipartiteness is a global yes/no invariant about the graph, it is not hierarchical (not decomposable) |
| `odd cycles` | Like a triangle, which why graphlets start at k=3 |  |
| `convergence` | At limit, values approach a finite, real number. Not diverging to +-infinity and not oscillating |  |
| `spectral radius` | literally $\rho(A) $ is an operator on A to produce the radius. $\rho(A) = \operatorname{max} \lambda_{i} $. This is the largest absolute magnitude `eigenvalue`. Answers "How far from the origin does the spectrum extend?" |  |
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
