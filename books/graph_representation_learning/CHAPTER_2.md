# Chapter 2. Background and Traditional Approaches

Methods prior to modern deep learning approaches

## 2.1 Graph Statistics and Kernel Methods

Traditionally, extract statistics or features (heuristic or domain knowledge) and use those as inputs to standard ML classifiers

- node-level statistics
- generalization of node-level to graph-level

### 2.1.1 Node-level Statistics and Features

What properties of a node distinguish it from other nodes? What are useful properties and statistics that we can use to characterize the nodes in this graph?

- Node Degree
- Node Centrality

#### Node Degree

$d_u $ for a node $u \in V $ - counts the number of edges incident to a node

$d_u = \sum_{v \in V} A [u, v] $

- for directed and weighted graphs, one can differentiate between different notions of degree
  - corresponding to outgoing edges or incoming edges (by summing over rows or cols of above equation)
- this is an essential statistic

#### Node Centrality

Does better job assessing *importance* of node

One popular and important measure of centrality: `eigenvector centrality` which takes into account how important a nodes neighbors are.

Defined as $e_u $ which includes a recurrence relation in which the nodes centrality is proportional to the average centrality of its neighbors:

$e_u = \frac{1}{\lambda} \sum_{v \in V} A[u, v]e_{v} \forall u \in V $ (where $\lambda$ is a constant)

Can rewrite in vector notation with **e** as the vector of node centralitites. Exposes that the recurrent defines the std eigenvector equation for the adjacency matrix

$\lambda e = A e $

Further, assuming that positive values of centrality are required, apply *Perron-Frobenius Theorem* to determine that the vector of centrality values e is given by the eigenvector corresponding to the largest eigenvalue of A

- One view of eigenvector centrality is that it ranks the likelihood that a node is visited on a random walk of infinite length on the graph
- Power iteration for eigenvector estimation illustrates this

$\bar{e}^{(t+1)} = A \bar{e}^{t} $

- if we start power iteration w vector $e^{(0)} = (1, 1, ..., 1)^T $
- then after first iteration e^1 will contain the degrees of all the nodes
- in general, at iteration $t \geq 1$, e^t will contain the number of length-t paths arriving at each node
  - thus, by iterating the process indefinitely we obtain a score that is proportional to the number of times a node is visited on paths of infinite length

The connection between node importance, random walks, and the spectrum of the adjacency matrix will return often throughout the book

Other centrality types:

- `betweeness centrality`
  - how often a node llies on the shortest path between two other nodes
- `closeness centrality`
  - average shortest path length between a node and all other nodes

#### The clustering coefficient

What features are useful for distinguishing between other nodes in the graph?

`clustering coefficient` - measures the proportion of closed trianges in a nodes local neighborhood

The `local variant` of the clustering coefficient is (Watts and Strogatz)

$c_{u} = \frac{|(v_1, v_2) \in E : v_1, v_2 \in N(u)|}{(\frac{d_u}{2})} $

- numerator counts number of edges between neighbours of node u
  - where $N(u) = \{u \in V : (u, v) \in E  \} $ to denote the *neighborhood*
- denominator calcs how many pairs of nodes are in u's neighborhood
- results
  - 1 means all of u's neighbors are also neighbors of each other
  - 0 means neighbors of u are not neighbors

Interesting property of real-world networks (social and biological) - they tend to have far higher clustering coeffs than one would expect if edges were sampled randomly

#### Closed triangles, ego graphs, and motifs

- instead of viewing as measure of local clustering, can view as a count of the number of closed triangles within each nodes local neighborhood
  - related to the ratio between the actual number of triangles and the total possible number of triangles within a nodes `ego graph` (the subgraph containing node, neighbors, and all edges between nodes in its neighborhood)

ego graph can be generalied to notion of counting arbitrary `motifs` or `graphlets` within a nodes ego graph

- instead of just counting triangles, consider more complex structures
  - cycles of particular length
  - counts of how often these different motifs occur in their ego graph

in doing this, transform task of node-level stats and features into graph-level task

### 2.1.2 Graph-level Features and Graph Kernels

Many methods here fall under group of `graph kernel methods`, which are approaches to designing features for graphs or implicit kernel functions for use in ML models

There are many approaches whicha are not discussed here

- Bag of nodes
- The Weisfeiler-Lehman kernel
- Graphlets and path-based methods

#### Bag of Nodes

simplest approach: aggregate node level stats

basically histograms for each feature we compute on each node, across the entire graph

downside: misses critical global properties

#### The Weisfeiler-Lehman Kernel

- use strategy of `iterative neighborhood aggregation`
  - extract node level features that contain more info than just their local ego graph
  - aggregate these richer features into graph-level representation

Core idea of WL:

- 1. assign initial lavel $l^{(0)}(v) $ to each node
  - in m ost graphs, label is simply the degree, i.e. $l^{(0)}(v) = d_v \forall v in V $
- 2. Next, iteratively assign a new label to each node by hashing the multi-set of the current labels within the nodes neighborhood:
  - $l^{(i)}(v) = \operatorname{HASH}(\{ \{l^{(i - 1)}(u \forall u \in N(v))  \} \}   ) $
  - where double-braces are used to denote a multiset, and HASH function maps each unique multi-set to a new unique label
- 3. After running K iterations of re-labeling (step 2), have now labeled $l^{(K)} (v)$ for each node that summarizes the structure of its K-hop neighborhood
  - can then compute histograms or other summary stats over the labels as a feature representation for the graph
  - aka, the WK kernel is computed by measuring the difference between the resultant label sets for two graphs

WL kernel is popular with theoretical properties

- one popular way to approx graph isomorphism is to check whether or not two graphs have the same label set after K rounds of WL
  - this approach is known to solve the isomorphism problem for a broad set of graphs

#### Graphlets and path-based methods

Count the occurrence of different, small subgraph structures `graphlets`

The `graphlet` kernel enumerates all possible graph structures of a particular size, and counting how many times they occur in the full graph

- This poses a challenging combinatorial problem
  - however, various approximations have been proposed

One alternative approach is to use `path-based methods`

- rather than enumerate graphlets, examine the different kinds of paths that occur in the graph
  - `random walk kernel`
  - `shortest path kernel`

Characterizing graphs based on walks and paths is very powerful

## 2.2 Neighborhood Overlap Detection

Node-level statistics do not quanitify *relationships* between nodes

Simplest neighborhood overlap measure - just counts number of neighbors that two nodes share:

$S[u, v] = |N(u) \cap N(v) | $

Where S[u, v] denotes the value quanitfying the relationship between nodes. Let $S \in \mathbb{R}^{|V| \times |V| } $ denote the *similarity matrix* summarizing pairwise node statistics

Given overlap statistic S[u, v], a common strategy is to assume that the likelihood of an edge (u,v) is simply proportional to S[u,v]:

$P(A[u, v] = 1) \propto S[u,v] $

### 2.2.1 Local overlap measures

`local overlap` statistics are simply functions of the number of common neighbors two nodes share

These are surprisingly "extremely" effective heuristics for link prediction - often achieve competitive performance even compared to advanced learning approaches

`Sorensen index` normalizes count of common neighbors by the sum of node degrees

$S_{Sorensen}[u, v] = \frac{2|N(u) \cap N(v)| }{d_u + d_v} $

Normalization of some kind is usually very important; otherwise the overlap measure would be highly biased towards predicting edges for nodes with large degrees

$S_{Salton} = \frac{2|N(u) \cap N(v)| }{\sqrt{d_u d_v}} $

$S_{Jaccard} = \frac{|N(u) \cap N(v)| }{|N(u) \cup N(v)|} $

There are also measures that go beyond counting - these seek to consider the *importance* of common neighbors in some way

`Resource Allocation (RA)` counts the inverse degree of the common neighbors

$S_{RA}[v_1, v_2] = \sum_{u \in N(v_1) \cap N(v2)} \frac{1}{d_u} $

Adamic-Adar (AA) uses the inverse log of the degrees

$S_{AA}[v_1, v_2] = \sum_{u \in N(v_1) \cap N(v2)} \frac{1}{\log(d_u)} $

### 2.2.2 Gloval overlap measures

These dont just consider the local neighborhood - in some cases, two nodes may have no overlap in their neighborhoods but still are members of the same community

`Katz index` is the most basic global overlap stat. Here, count the number of paths of *all lengths* between a pair of nodes

$S_{Katz}[u, v] = \sum_{i=1}^{\inf} \beta^{i}A^{i}[u,v] $

where $\beta \in \mathbb{R}^+ $ is user defined param controlling how much weight is given to short vs long paths. A small value of $\beta < 1 $ would down-weight the importance of long paths

#### Geometric Series of Matrices

Katz index is one example of a `geometric series of matrices`, variants of which occur frequently in graph analysis and graph representation learning

The solution to a basic `geometric series of matrices` is given by the following Theorem 1

Theorem 1.

- Let **X** be a real-valued *square matrix*
- let $\lambda_1 $ denote the *largest eigenvalue* of X

Then:

$(I - X)^{-1} = \sum_{i=0}^{\infty} X^i $

- iff and only if $\lambda_1 < 1 $ and $(I - X) $ is *non-singular* (non-invertible)
  - singular =
    - det(A) = 0
    - not invertible
    - at least one eigenvalue = 0

Proof:

- let $s_n  = \sum_{i=0}^{n} X^i $
  - then we have that:
    - $X s_n = X\sum_{i=0}^{n}X^i $
    - $X s_n = \sum_{i=1}^{n+1}X^i $
  - and
    - $s_n - X s_n = \sum_{i=0}^n X^i - \sum_{i=1}^{n+1}X^i $
      - I think this step uses let s_n = ...
    - $s_n (I - X) = I - X^{n+1} $
    - $s_n = (I - X^{n+1})(I - X)^{-1} $
      - (note this step takes the $(I - X)^{-1} $ of each side)
- And
  - if $\lambda_1 < 1 $
  - we have that $\lim_{n \rightarrow \infty} X^n = 0 $
  - so
    - $\lim_{n \rightarrow \infty} s_n = \lim_{n \rightarrow \infty} (I - X^{n+1})(I - X)^{-1} $
    - $\lim_{n \rightarrow \infty} s_n = I(I - X)^{-1} $
    - $\lim_{n \rightarrow \infty} s_n = (I - X)^-1 $

Based on Theorem 1, we can see that the solution to the Katz index is given by

$S_{\operatorname{Katz}} = (I - \beta A)^{-1} - I $

Where $S_{\operatorname{Katz}} \in \mathbb{R}^{|V| \times |V|  } $ is the full matrix of node-similarity values
