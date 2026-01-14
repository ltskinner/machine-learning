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

$c_{u} = \frac{|(v_1, v_2) \in E : v_1, v_2 \in N(u)|}{(\binom{d_u}{2})} $

- numerator counts number of edges between neighbours of node u
  - where $N(u) = \{v \in V : (u, v) \in E  \} $ to denote the *neighborhood*
- $\binom{n}{k} $
  - aka $\frac{n(n-1)}{k} $
  - aka "number of neighbor pairs"
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

There are many approaches which are not discussed here

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

- one popular way to approx `graph isomorphism` is to check whether or not two graphs have the same label set after K rounds of WL
  - this approach is known to solve the isomorphism problem for a broad set of graphs

WL is discrete (labels, hashes); not differentiable; not good for continuous attributes, dynamics graphs, learning end-to-end representations | If two nodes end up with the same value, WL failed to distingush them. They are considered to have identical structural equivalence (under 1-WL) (aka, a hash collision). In the Florentine example, they all have different WL values so they are all structurally distinct. In Karate club, there are several with the same label 6, so they are 1-WL-equivalent - think `structural symmetry`.

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

#### Not Garbage Katz Index Notes

The Katz index is the count of the number of walks of all lengths between a pair of nodes

Remember, by definition: $A[u,v]$ counts the walks of **length 1** from u to v

The Adjacency Matrix **is** a linear operator that extends walks by one edge

- Multiplying by A = "take one more step"
- Raising to power k = "take k steps"
- Entries in the result count *how many ways* one can do that

>> Matrix multiplication **is** the operation of "count all valid ways to chain steps"

The `walks` under consideration here consider repetition - edges can be walked more than one time in the steps

The key theorem Katz relies on is:

$\|A^k \| \approx \rho(A)^k $ as $l \rightarrow \infty $

where $\rho(A) = \operatorname{max}(\lambda_i) $

- eigenvalues < 1 -> decay
- eigenvalues = 1 -> persist
- eigenvalues > 1 -> explode

So to converge:

- $\rho(\beta A) < 1 $
- aka $\beta \rho(A) < 1 $
- aka $\beta \lambda_1 < 1 $
- aka $\beta < \frac{1}{\lambda_1} $

There is a `convergence` criterion that $\beta < 1 / \lambda_1 $

In practice, we just set $\beta $ to be low, like $\beta = 0.01 $ instead of precomputing the largest eigenvalue

##### Closed Form Solution

IFF:

- \lambda_i < 1 for each eigenvalue of A (especially the largest eigenvalue)
  - aka: the convergence criterion has been met
- (I - A) is non-singular (invertible)

Then, the `closed form solution` to a `geometric series of matrices` is:

$\sum_{k=0}^{\infty} A^k = (I - A)^{-1} $

So

$S_{\operatorname{Katz}} = (I - \beta A)^{-1} - I $

Where the $- I $ at the end nulls out the 0-length walks of a node to itself

>> So this is critical actually

- The closed form solution to geometric series of matrices, the series must begin at i=0
  - Obviously, Katz does not
  - So in that proof above, we effectively make the series start at 0, instead of 1
  - Then, after gathering the closed form, we then subtract the scaffold identity corresponding to i=1

#### Leicht, Holme, and Newman (LHN) Similarity

One issue with the Katz index is that it is strongly biased by node degree. Eqn 2.14 will generally give higher overall similarity scores when considering high-degree nodes, compared to low-degree ones, since high-degree nodes will generally be involved in more paths

To alleviate this, LHN proposes an improved metric by considering the ratio between the actual number of observed paths and *the number of expected paths between two nodes*

$\frac{A^i}{\mathbb{E}[A^i]} $

i.e. the number of paths between two nodes is normalized based on our expectations of how many paths we expect under a random model

To compute the expectation $\mathbb{E}[A^i] $, we rely on what is called the `configuration model`, which assumes that we draw a random graph with the same set of degrees as our given graph. Under this assumption, we can analytically compute that:

$\mathbb{E}[A[u,v]] = \frac{d_u d_v}{2m} $

where we have used $m = | E | $ to deonte the total number of edges in the graph. this states taht under a random configuration model, the likelihood of an edge is simply proportional to the product of the two node degrees

This can be seen by noting that there are $d_u $ edges leaving u, and each of these edges has a $\frac{d_v}{2m} $ chance of ending at v.

For $\mathbb{E}[A^2[v_1, v_2]] = \frac{d_{v1}d_{v2}}{(2m)^2} \sum_{u \in V} (d_u - 1)d_u $

This follows from the fact that path of length 2 could pass through any intermediate vertex u, and the likelihood of such a path is proportional to the likelihood that an edge leaving v1 hits u multiplied by the probab that an edge leaving u hits v2

The analytical computation of expected node path counts under random configuration model becomes intractable as we go beyind path lengths of three. So to obtain the expection $E[A^i]$ for longer paths (i.e., i > 2), LHN relies on the fact the largest eigenvalue can be used to approximate the growth in the number of paths

In particular, if we define $\bar{p}_i \in \mathbb{R}^{|V|} $ as the vector counting the number of length-i paths between node u and all other nodes, then we have that for large i

$A \bar{p}_{i} = \lambda_{i \bar{p}_{i-1}} $

since $\bar{p}_{i} $ will eventually converge to the dominant eigenvector of the graph.

This implies that the number of paths between two nodes grows by a factor of $\lambda_{1} $ at each iteration, where $\lambda_{i} $ is the largest eigenvalue of A

Based on this approximation for large i as well as the exact solution for i = 1 we obtain:

$\mathbb{E}[A^i[u,v]] = \frac{d_u d_v \lambda^{i-1}}{2m} $

Finally, putting all together we obtain the *normalized version* of the `Katz index`, aka the `LNH index`

$S_{\operatorname{LNH}}[u,v] = I[u,v] + \frac{2m}{d_u d_v} \sum_{i=0}^{\infty} \beta^{i}\lambda_{i}^{1 - i} A^{i}[u,b] $

where I VxV identity matrix

Unlike the `Katz index`, the `LNH index` accounts for the *expected* number of paths between nodes and only gives a high similarity measure if two nodes occur on more paths than we expect

using Theorem 1, the solution to the matrix series (after ignoring diagonal terms) can be written as:

$S_{\operatorname{LNH}} = 2\alpha m \lambda_1 D^{-1}(I - \frac{\beta}{\lambda_{1}} A)^{-1} D^{-1} $

where D is a matrix with node degrees on the diagonal

#### Not Garbage LHN Notes

Improve on LHN by considering the ratio between the actual # of observed paths, compared to the number of **expected** paths between two nodes

$\frac{A^i}{\mathbb{E}[A^i]} $ Where $\mathbb{E} $ is the expected paths

This expresses a normalization based on the expected # of paths under a random model

$\mathbb{E}[A^i] $ depends on the `configuration model`. The `configuration model` assumes:

- we draw a random graph
- has the same set of degrees as the given graph, so

$\mathbb{E}[A[u,v]] = \frac{d_u d_v}{2m} $, where $m = |\mathcal{E}| $ is the number of edges in the graph

In laymans terms, the expected likelihood is proportional to the product of the two node degrees

So, standing on $u$, there are $d_u$ edges exiting. Each of these edges has a $\frac{d_v}{2m} $ chance of ending on $v$

This is fine for $A^1$ and $A^2$ but becomes computationally intractable at for >= 3

To work aorund this, the largest eigenvalue can be used to approximate the growth in the number of paths.

Here, we exploit the `power method` to identify a vector. In regular power method, this estimates the eigenvector. But here, this vector is the expected number of paths (I think)

$\bar{p}_{i+1} = A\bar(\bar{p}_i) $

##### Power Method Notes

Iterative method to estimate the eigenvector + eigenvalue numerically

$\bar{x}_{k+1} = \frac{A \bar{x}_{k}}{|A \bar{x}_{k} |}$

Then, $\bar{x}_{k} \rightarrow \bar{v}_{max} $

And, $\lambda_{max} \approx \bar{x}_{k}^{T} A \bar{x}_{k} $

Aka, $A\bar{v}_{i} = \lambda_i \bar{v}_{i} $

Back to the LHN notation, we are given:

$A \bar{p}_{i} = \lambda_{1}\bar{p}_{i-1} $

Im pretty sure this is just a notation mismatch, and its structurally the same. Here, $\bar{p}_{i} $ is the vector counting the number of length-i paths between node u and all other nodes. The above equation holds for large values of i (long paths), because $\bar{p}_i$ will converge to the dominant eigenvector of the graph

Knowing this, we obtain an updated expectation:

$\mathbb{E}[A^{i}[u, v]] = \frac{d_u d_v \lambda^{i-1}}{2m} $

- here, the $\lambda^{i-1} $ basically means that because we start at i=1
  - at the first iteration, we have $\lambda^{0} = 1 $
  - which, for i=1 we just get the $E[A^1[u, v]] = \frac{d_u d_v \bar{1}}{2m} $

Then to reconcile this with the Katz index, take the 1/E... to get the non closed form equation

$S_{LNH}[u,v] = I[u,v] + \frac{2m}{d_u d_v}\sum_{i=0}^{\infty} \beta^{i}\lambda_{1}^{1-i}A^{i}[u,v] $

Here, we include the leading I[u,v] because even though we begin at i=0, i=0 does not produce an I matrix. therefore, we need to add that I to have the series form required for inversion.

Also note that the $\sum_{i=0} $ begins at 0, instead of 1.

Then the next big jump is:

$(D^{-1}X D^{-1})_{uv} = \frac{D_{uv}}{d_u d_v} $, for my derivations, just took the entire series term and hid behind X

Additionally, that leading identity doesnt add anything cause it only has values on diagonal for self paths or whatever. The resulting equation only represents off-diagonal entries (which are the interesting ones). Also the book explicitly says "after ignoring diagonal terms" (which originally I thought had to do with D, but its about the identity lol)

$S_LNH = 2 \alpha m \lambda_{1} D^{-1}(I - \frac{\beta}{\lambda_{1}} A)^{-1} D^{-1} $

#### Random Walk Methods (start of page 21)

Here, random walks are considered instead of exact counts of paths over the graph

Stochastic matrix: $P = AD^{-1} $, assuming D is degrees on diagonal

Then, compute:

$\bar{q}_u = cP\bar{q}_{u} + (1 - c)e_u $

- $e_u $ is one-hot indicator vector for node u
- $\bar{q}_{u}[v] $ gives stationary probability that random walk starting at node $u$ visits node $v$
- $c$ determines probability that random walk restarts at node $u$ at each timestep
  - *without* this, random walk probabilities converge to a normalized variant of the eigenvector centrality
  - *with* it, we obtain measure of importance specific to node $u$
    - this is because random walks are continually "teleported" back to that node

Solution given by:

$\bar{q}_u = (1 - c)(I - cP)^{-1}e_u $

and we can define a node-node random walk similarity measure as

$S_{RW}[u,v] = \bar{q}_{u}[v] + \bar{q}_{v}[u] $

i.e., the similarity between a pair of nodes is proportional to how likely we are to reach each node from a random walk starting from the other node

#### Not Ass notes

$\bar{q}_u[v]$ is the stationary probability that a random walk starting at u is at node v, so $\bar{q}_u$ contains probabilities for all nodes

- here, $\bar{q}_u[u] $ is the likelihood of being on node u and staying stationary, including both:
  - explicit "teleports" back to u
  - returns via graph structures (cycles, neighbors)

And, $S_{RW}[u,v] = \bar{q}_{u}[v] + \bar{q}_{v}[u] $

effectively answers: "how easily can I reach u from v, and v from u" with the following characteristics:

- balancing graph asymmetry
- mitigating degree gias
- balancing directionality
- makes similarity symmetric
- aligns w/ kernel-style similarity measures

## 2.3 Graph Laplacians and Spectral Methods

This section:

- addresses learning to cluster the nodes in a graph
- motivates the task of learning low-dimensional embeddings of nodes
- introduces important matrices that can be used to represent graphs
- introduces the foundations of `spectral graph theory`

### 2.3.1 Graph Laplacians

There is no loss of information in an adjacency matrix.

`Laplacians` are formed by various transformations of the adjacency matrix

#### Unnormalized Laplacian (most basic)

$L = D - A$, where D is degree matrix, with the following properties:

- It is symmetric: $L^T = L$
- positive semi-definite PSD: $\bar{x}^{T}L\bar{x} \geq 0, \forall x \in \mathbb{R}^{|\mathcal{V|}} $
- The vector identity holds $\forall x \in \mathbb{R}^{|\mathcal{V|}} $

$\bar{x}^{T} L x = \frac{1}{2} \sum_{u \in V}\sum_{v \in V} A[u,v](x[u] - x[v])^{2} $

$\bar{x}^{T} L x = \sum_{(u, v)\in \mathbb{E}} (x[u] - x[v])^{2} $

- $L $ has $|V| $ non-negative eigenvalues:
  - (eigenvalue for each node, which are non-negative)
  - $0 = \lambda_{|V|} \leq \lambda_{|V| - 1} \leq ... \lambda_{1} $

#### Normalized Laplacians

1. Symmetric normalized Laplacian:

$L_{sym} = D^{-\frac{1}{2}}LD^{-\frac{1}{2}} $

2. Random walk Laplacian:

$L_{RW} = D^{-1}L $

Both have similar properties as the unnormalized Laplacian, but agebraic properties differ by small constants due to the normalization

Theorem 2 (below):

- holds but with eigenvectors for the 0 eigenvalue scaled by $D^{\frac{1}{2}} $ for $L_{sym} $
- holds exactly for $L_{RW} $

#### Theorem 2 (blue section)

The Laplacian sumamrizes many important properties of a graph.

Theorem 2: The `geometric multiplicity` of the 0 eigenvalue of the Laplacian **L** corredpons to the number of connected components in the graph

- GM = the number of linearly independent eigenvectors for that eigenvalue
  - dimension of the eigenspace for the eigenvalue
  - GM <= AM, and GM=AM required for diagonalization
- `0 eigenvalue`
  - There exists a non-zero vector x such that $Lx = 0 $
    - aka the `null space`
    - The dimension of Null(L) equals the number of connected components of the graph
  - Meaning: there are directions in the vector space that L completely anihilates
    - no stretch, shrink, or rotating
  - when 0, means that every node has the same value of all its neighors
    - aka perfect smoothness
    - correspond to directions that diffusion cannot act on
  - in context of contagion dynamics:
    - zero laplacian eigenvalues correspond to opinion modes that are invariant under diffusion, reflecting disconnected componments where opinions evolve independently - structural independence

## 2.3.2 Graph Cuts and Clustering

1/14/2026 LTS - hitting pause here... I am loving this book but that proof for Theorem 2 I do not have my foundational Linear Algebra up to this level. Going to revisit LADR and sync back up with this in a bit.
