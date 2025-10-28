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
