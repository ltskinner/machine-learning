# Chapter 5. Graph and Heterogeneouse Network Transformations

- simple (homogeneous) graphs
- elaborate (heterogeneous) graphs/information networks

## 5.1 Embedding Simple Graphs

- motivation for mining graphs is that info may exist both at:
  - the instance level (nodes)
  - the relation/interaction level (edges)

`network embedding` is a mechanism for converting networks into a tabular/matrix representation

- each node is a vector of fixed length d
  - set of nodes represented as a table w/ d columns
- the conversion aims to:
  - preserve structural properties of a nertowk
  - preserve node similarity
  - convert node similarity into vector similarity

The process of graph embedding corresponds to a mapping:

$f : G \rightarrow \mathbb{R}^{|V| \times d} $

- a mapping that takes G as input and projects it to a Vxd dimensional vector space
- where each node $u \in V$ gets assigned a d-dim vector of real numbers
- rationale for construction of algs that approximate f is to learn a representation in the form of a feature matrix, given that many existing data mining approaches operate on feature matrices. The feature matrix can be used in combination with standard ml to solve network analysis tasks (e.g. node classification or link prediction)

Early gen graph embeddings were significantly influenced by embedding construction from text data

- skip gram (used in word2vec) was adapted to learn node representations
  - idea is to use random walks as documents and individual nodes as words
  - for a given node, nearby nodes in the same random walk establish a context used to construct positive examples
  - embedding vectors are extracted from the hidden layer of the trained NN

### 5.1.1 DeepWalk Algorithm

- one of the first algs that treats nodes as words, and short random walks as sentences
  - learns distribution of node cooccurrences
  - learns individual node representations
    - by maximizing likelihood of observing neighborhood nodes in a set of short random walks (given the representation of the node)
- node2vec is a variant of this

Optimization problem:

$\argmax_{\Phi} [\log P r(\{v_{i-w}, ..., v_{i-1}, v_{i+1}, ..., v_{w+w} \} | \Phi(v_{i})  ) ] $

- ok this is really interesting
  - walks are like sentences basically
    - (not the sentence as a single unit - the sentence as a series of words)
  - generate inf training data on random walks like we have with text on internet

### 5.1.2 Node2vec Algorithm

- random walk to calc features that express similarities between node pairs

For a user-defined number of columns d, the alg returns:

- matrix: $F^* \in \mathbb{R}^{|V|\times d} $
- defined as:

$F^{*} = \operatorname{node2vec}(G) = \argmax{F \in \mathbb{R}^{|V|\times d} } \sum_{u \in V} ( -\log(Z_{u}) + \sum_{n \in N(u)} F(n) \cdot F(u) ) $

- where, N(u) denotes the network neighborhood of node u
- F(n) \cdot F(u) denotes the dot product
- Z_u is explained below
- Each matrix F is a collection of d-dimensional feature vectors
  - i-th row of matrix corresponding to the feature vector of the i-th node in the network
- Write F(u) to denote the row of matrix F corresponding to node u
- Resulting maximal matrix is denoted F*

Goal of node2vec is to construct feature vectors F(u) in such a way that feature vectors of nodes that share a certain neighborhood will be similar

- inner sum $\sum_{n \in N(u)} F(n) \cdot F(u)$ which is maximized calcs the similarities between node u and all nodes in neighborhood
  - it is large if feature vectors of nodes are colinear, or have large norm

Zu calced as:

$Z_u = \sum_{v \in V} e^{F(u) \cdot F(v)} $

- value of \log(Z_u) decreases when the norms of feature vectors F(v) increase
  - this penalizes collections of feature vectors with large norms

There are also probabilistic processes for selecting nodes from network

- unlike PageRank random walker, the transition probabilisties for traversing from node n1 to node n2 depend on node n0 that the walker visited before node n1, making a second order random walk. Non-normalized transitoin probabilities are set using two parameters

- $P(n_2 | n_0 \implies n_1) = $
  - $ \frac{1}{p} $ if n2 = n0
  - $1$ if n2 can be reached from n1
  - $\frac{1}{q} $ otherwise

here:

- p is the *return* parameter
  - low value means random walker is more likely to backtrack steps and closer to a BFS
- q is the *in-out*
  - low value encourages walker to move away from starting point and closer to DFS

To calculate maximizing matrix F*, a set of random walks of limited size is simulated starting from each node in the network to generate samples of sets N(u)

SGD is used to find maximizing matrix. The matrix of feature vectors is estimated at each sampling of neighborhoods N(u) for all nodes in the network. The resulting matrix F* maximizes the expression for the simulated neighborhood set

### 5.1.3 Other Random Walk-Based Graph Embedding Algorithms

- LINE
  - extends DeepWalk and node2vec
  - takes into account local and global network structure, but can efficiently handle large networks
- PTE
  - exploits heterogeneous networks w/ texts for supervised embedding construction
  - leverage documents labels to improve the quality of the final embeddings
- NetMF
  - generalization of DeepWalk, node2vec, LINE, and PTE
  - reformulates algs as matrix factorization problem
    - indicates that node embedding algs can be described as part of a more general theoretical framework
- struc2vec
  - two main ideas
    - reps of two nodes must be close if the two nodes are structurally similar
    - latent node rep should not depend on any node or edge attribute (including the node labels)
      - the structural identity of a given pair of nodes must be independent of their 'position' in the network
    - struc2vec attempts to determine node similarity by varying neighborhood size
- HARP
  - hierarchical representation learning for networks
  - address weaknesses of DeepWalk and node2vec (random init node embed)
    - leads to local optima
  - HARP creates hierarchy of nodes
    - aggregates nodes in each layer using graph goarsening
    - embeddings of higher level graphs are used to initialize lower levels, all the way down to original graph
  - can be used in conjunction with random walk based

## 5.2 Embedding Heterogeneous Information Networks

`heterogeonous information networks` describe heterogenous types of entities and different types of relations

### 5.2.1 Heterogeneous Information Networks

graphs dont hold any real data - information networks hoever are graphs where nodes have properties

#### Definition 5.1 Heterogeneous Information Network

A heterogenous informatoin network is a tuple $(V, E, \mathcal{A}, \mathcal{E}, \tau, \phi)$, where

- $G = (V, E) $ is a directed graph
- $\mathcal{A} $ is a set of object types
- $\mathcal{E} $ is a set of edge types
- function $\tau : V \rightarrow \mathcal{A} $, establish relations between nodes
- function $\phi : E \rightarrow \mathcal{E} $, establish relation between edges
  - by the following conditions:
    - if edge e1 = (x1, y1) and e2 = (x2, y2) belonging to the same edge type
      - (i.e. $\phi(e_1) = \phi(e_2) $)
    - then their start points and their end points belong to the same node type
      - (i.e. $\tau(x_1) = \tau(x_2) $ and $\tau(y_1) = \tau(y_2) $)

Sun and Han node that sets $\mathcal{A} \mathcal{E} $ (along with the restrictions imposed by the definition of a heterogeneous information network) can be seen as a network as well, with edges connecting two node types if there exists an edge type whose edges connect vertices of the two node types. This meta-level desc of a network is a *network schema*

#### Definition 5.2 Network Schema

A network schema of a heterogeneous information network $G = (V, E, \mathcal{A}, \mathcal{E}, \tau, \phi)$, denoted $T_{G} $, is a directed graph with vertices $\mathcal{A} $ and edges $\mathcal{E} $, where edge type $t \in \mathcal{E} $ whose edges donnect different vertices of type $t_1 \in \mathcal{A} $ to vertices of type $t_2 \in \mathcal{A} $, defines an edge in $\mathcal{E} $ from type $t_{1} $ to $t_{2} $

### 5.2.2 Examples of Heterogeneous Information Networks

An information network includes both the network structure and the data attached to individual nodes or edges

Forms of networks:

- Bibliographic networks or citation networks
- Online social networks
- Biological networks

### 5.2.3 Embedding Feature-Rich Graphs with GCNs

Real-world graphs often consist of nodes, which are further described by a set of features

Graph-convolutional neural networks (GCNs) exploit this extra info. Sort of like looking for presence of features within the neighborhood (akin to pixel regions)

- For a given node, its neighboring nodes' features are aggregated into a single representation which, apart from local topology,  contains feature information from all the considered neighbors.
- This aggregation starts from the most distant nodes, using predefined aggregation operators such as max- and mean-pooling
- the representation of each node closer to the node of interest is obtained by aggregating its neighbors feature vectors

Some well known GCNs

- Spectral GCN
  - NN arch, one of the first
  - generalize convolutions over structured grids (e.g. images) to spectral graph theory
  - efficient neighborhood-based feature aggregation achieves sota classification performance on many datasets
- Laplacian GCN
  - similar to spectral GCN
  - shows how the harmonic spectrum of a graph Laplacian can be used to construct NN with relatively low number of parameters
- GCN for node classification
  - semi-supervised learning on graphs by using GCNs offer SOTA classification performance
- GraphSAGE
  - many node embedding methods do not generalize well to unseen nodes (when graph dynamics are considered)
  - **inductive graph embedding learner**
  - first samples a given nodes neighborhood, then aggregates features of sampled nodes

### 5.2.4 Other Heterogeneous Network Embedding Approaches

- metapath2vec
  - uses set of pre-defined paths for sampling, yielding embeddings that are SOTA on node class tasks
- OhmNet
  - heterogeneous biological setting
  - outperforms tensor factorization methods
- HIN2Vec
  - works with probability that there is a meta-path between nodes u and v
  - generates positive tuples using homogeneous random walks disregarding the type of nodes and links
  - for each positive instance, generates several negative instances by replacing node u with a random node
- Heterogeneous data embeddings
  - images, text, video could all form a heterogenous graph
  - when embedded properly, offers sota annotation performance
