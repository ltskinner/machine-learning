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

## 5.3 Propositionalizing Heterogeneous Information Networks

### 5.3.1 TEHmINe Propositionalization of Text-Enriched Networks

- combines elements from text mining and network analysis as basis for constructing feature vectors describing both the position and internal properties of network nodes
- three steps:
  - network decomposition
  - feature construction
  - data fusion

#### 5.3.1.1 Heterogeneous Network Decomposition

- focuses on network structure
- orig hetin (heterogeneous information network) is decomposed into a set of homogeneous networks
- each homogeneous network is constructed from a circular walk in the original network schema
  - if a sequence of node types $t_1, t_2, ..., t_n $ forms a circular walk in the network schema (meaning $t_1 = t_n $)
    - then two nodes a and b are connected in the decomposed network
  - if there exists a walk $v_1, v_2, ..., v_m $ such that $v_1 = a, v_m = b $ and each node $v_j  $ in the walk is of the same type

goal: construct homogeneous networks with nodes of a single type

- construct relationships between nodes of a given type
- instead of original directed edges, construct undirected edges
  - each edge corresponds to one circular path in the network schema
  - label edges by counting the number of paths in the original network that correspond to the newly constructed undirected edge

This step is not necessarily fully automatic, as different meta-paths can be considered for each hetero entwork. Usually, meta paths of heterogeneous networks have real-world meaning, therefore expert judgement should be used to assess which meta-paths should be defined

#### 5.3.1.2 Feature Vector Construction

Second step: two feature vector types

- BoW vectors
  - constructed from text docs enriching the individual nodes
  - built w std text preprocessing (tokenizaiton, stop words, lemma, n-gram)
  - euclidean norm nodmalized
- P-PR vectors
  - running Personalized PageRank (P-PR) for each node of individual homo network
  - see process below

P-PR construction

- for given node v of homo network
  - P-PR vector of node v ($P-PR_v $) has (for each other node w of the network) a likelihood of reaching w with a random walk starting from v
  - defined as the stationary distribution of the position of a random walker, which either
    - selects one outgoing node (from v)
    - or jumps back to starting position
  - probability p of continuing the walk is a param of the alg (default 0.85)
- resulting vectors normalized by Euclidean to be compatible with BoW vectors

Many ways to compute P-PR vectors, below is iterative version

- first step, weight of node v is 1, other nodes weight 0 to construct r^0, the initial estimate of P-PR vector
- at each step, weights are spread along the connections of network by:
  - $r^(k + 1) = p \cdot (A^T \cdot r^{k}) + (1 - p)\cdot r^(0) $
    - where r^(k) contains weights of P-PR vector after k iterations 
    - and A is the networks adjacency matrix, normalized so elements in each row sum to 1
- if all elements in given row of the coincidence matrix are zero (i.e. if a node has no outgoing connections)
  - then all values in row are set to $\frac{1}{n} $, where n is number of vertices
    - (this simulates behavior of walker jumping from a node with no outgoing connections to any other node in the network)

#### 5.3.1.3 Data Fusion

Step 3

- For each network node v, the result of running both BoW and P-PR vector construciotn is a set of vectors $\{v_0, v_1, ..., v_n  \} $
  - where v_0 is the BoW vector for text in node v
  - and for each $i (1 \leq i \leq n) $
    - where n is number of network decomps
    - v_i is the P-PR vector of node v in the ith homo network
- in final step, vectors are concat to create a concatenated vector v_c for node v:
  - $v_c = \sqrt{\alpha_0}v_0 \bigoplus \sqrt{\alpha_1}v_1 \bigoplus ... \bigoplus \sqrt{\alpha_n}v_n $
  - constructed by using positive weights $\alpha_{0}, \alpha_{1}, ..., \alpha_{n} $ (which sum to 1)
  - $\bigoplus $ represents the concatenation of two vectors
- values of weights $\alpha_{i} $ are determined by optimizing the weights for a given task (such as node classification)
- if multiple kernel learning (MKL) optimization is used, which views the feature vectors as linear kernels, each vectors v_i correspond to a linear mapping $\bar{v_{i}} \mapsto x \cdot v_i $, and the final concatenated vector v_c for node v represents the linear mapping:
  - $[x_0, x_1, ..., x_n ]  \mapsto \alpha_{0} x_{0} \cdot v_{0} + \alpha_{1} x_{1} \cdot v_{1} + ... + \alpha_{n} x_{n} \cdot v_{n} $
- another possibility is to fine-tune weights using a general purpose optimization algorithm, such as differential evolution

### 5.3.2 HINMINE Heterogeneous Networks Decomposition

Designed for propositionalization of hetero networks that do not include texts enriching the individual network nodes

In TEHmINe:

- weight of link between two base nodes is equal to the number of intermediate nodes that are linked to both the base nodes
  - formally, the weight of alink between nodes v and u is:
    - $w(v, u) = \sum_{m \in M} 1  $ (m is linked to v and u)
    - whre M is the set of all intermediate nodes

A weakness of TEHmINe approach is that it treats all the nodes qually, which may not be appropriate from the information content point of view

- ex. if two papers share an author who only co-authored a small number of papers
  - it is more likely that these two papers are similar
  - than if the two papers thare an author that co-authored tens or even hundreds of papers
  - the first pair of papers should therefore be connected by a stronger weight than the second
- similarly, if papers are labeled by the research field, two papers sharing an author publishing in only one research field are more likely to be similar than if they share an author who has co-authored papers in several fields of research.
  - again, the first pair of papers should be connected by an edge with a larger weight

HINMINE resolves above weakness

- in the network decomp step (going from hetero to simpler homo)
  - HINMINE leverages weighting schemes used in text analysis for word weighting in BoW vector construction to weighting nodes in network analysis

Consequently, the underlying idea of HINMINE is to treat the paths between network nodes as if they were words in text documents

- forms BoW representations by treating paths (with one intermediate node) between pairs of nodes of interet as individual words
- samples paths and counts their occurences
- counts are weighted using an appropriate weighting heuristic (below) ensuring that more specific paths are assigned higher weights
- so, adapted and exteded in a similar way as TF (term frequency) weights are in text minim
  - as a result, we replace count 1 with weight w(t)
    - where w is a weighting function which, e.g., penalizes terms t appearing in many documents (like TF-IDF weight function which penalizes term t appearing in docs from different classes)

Based on this, calculating w(v,u) in network setting:

$w(v, u) = \sum_{m \in M} w(m) $  m is linked to v and u

By using different weighting functions w, more informative weights can be used in the network decomp step

- analogous to TF-IDF, weighting functions w should decrease the importance of links to/from highly connected intermediate nodes
- term weighting schemes can be adapted to set weights to intermediate nodes in hetero networks (such as authors in this example)
- this can be done in a way that the weights of a link between two nodes is the sum of weights of all the intermediate nodes they share
- and, if we construct homo network in which nodes are connected if they share a connection to a node of type T in the original heteo network, then the weight of the link is:

$w(v, u) = \sum_{m \in T} w(m) $ where $(m, v) \in E $ and $(m, u) \in E $

Several ways to compute:
- TF, TF0IDF, $\chi^2 
- IG, GR, Delta-IDF, RF, Okapi BM25$

Empirical results of HINMINE show the choice of heuristics impacts the performance of classifiers

The best choice of weighting heuristic depends on the structure of the network

## 5.4 Ontology Transformations

Approaches which leverage domain-curated background knowledge

Semantic Data Mining (SDM)

### 5.4.1 Ontologies and Semantic Data Mining

Ontology:

- DAGs
- formed of concepts, relations
  - (subject, predicate, object) triplets

`taxonomy` = simplest ontologies (hmm)

- hierarchically ordered
  - general nodes up top
  - specific leaf nodes
  - relationships are "is-a" or "part-of"

To find patterns in data annotated with ontologies, *semantic data mining* take as input a set of class labeled instances, and the background knowledge encoded in the form of ontologies

- frequent goal: find descriptions of target class instances as a set of rules of the form:
  - $\operatorname{TargetClass} \leftarrow \operatorname{Explanation} $
  - where Explanation is a logical conjunction of the terms from the ontology
  - roots in:
    - symbolic rule learning
    - subgroup discovery
    - enrichment analysis

SDM toolkit:

- SDM-SEGS
  - extension of domain-specific algorithm SEGS for semantic subgroup discovery in gene expression data
- SDM-Aleph
  - populat inductive logic program Aleph

Hedwig system takes best of both

- search mechanism to exploit hierarchical nature of ontologies
- uses background knowledge in RDF triplets
- output is set of descriptive patterns in the form of rules
  - rule conditions are conjunctions of domain ontology terms
- seeks to discover explanations that best describe and cover as many target instances (and as few non-target instances) as possible

### 5.4.2 NetSDM Ontology Reduction Methodology

SDM algs perform well on relatively small datasets. Even within these, the algorithms search in a very large space of patterns

NetSDM constrains the search space to only most relevant concepts - useful for data and ontology reduction, as well as data preprocessing, before using an alg for network propositionalization or embeddings

Four main steps:

- 1. Convert examples and background knowledge to network, where background knowledge terms represent nodes
- 2. Estimate the significance of background knowledge terms using a node scoring function
- 3. Reduce the background knowledge by removing less significant terms, keeping only a proportion c of the top-ranking terms
- 4. Apply a semantic data mining alg to the original dataset and the reduced background knowledge

Scoring function used in the onltology shrinkage should:

- 1. evaluate the significance of terms based on the data
- 2. be efficiently computed

#### 5.4.2.1 Converting Ontology and Examples into Network Format

Direct conversion:

- convert each relation between two ontology terms into an edge

Hypergraph conversion:

- every triplet of ontology-based background knowledge (along with background knowledge terms) forms an additional node in the network with three connections:
  - one with subject
  - one with object
  - one with predicate
- results in larger information network, but less information is loss

#### 5.4.2.2 Term Significance Calculation

After converting background knowledge and input data intoa network, score significance of background

Two approaches:

- P-PR
- node2vec

#### 5.4.2.3 Network Node Removal

Low-scored nodes are removed. Two variants of removal

- Naive node removal
  - delete both node and any edges
  - can cause resulting network to decompose into several disjoint components
- Transitive node removal
  - many edges are transitive
    - i.e. $a R b \land b R c \implies a R c $
  - when removing node b, insert a bridging/stitch relation between a and c

## 5.5 Embedding Knowledge Graphs

In knowledge graphs, edges correspond to relations between entities (nodes) and the graphs present subject-predicate-object triplets

The learning algorithms on KGs solve problems like:

- triplet completion
- relation extraction
- entity resolution

KG embedding algs are highly scalable for large, semantics-rich graphs

- (h, r, t) = (head, relation, tail)
- embedding methods optimize the total *plausability* of the input set of triplets
  - plausability of signle triplet is $f_{r}(h, t) $

Translational distance models

- exploit distance-based scoring functions
- assess plausability of a fact with the distance between the two entities, usually after a translation carried out by the relation
  - TransE
    - models relationships by interpreting them as translations operating on the low-dimensional embeddings of the tntities
    - cost fn to minimize:
      - $f_{r}(h, t) = \|h + r - t\|^{2} $

Non-deterministic KG embeddings

- take into account the uncertainty of observing a given triplet
  - KG2E
    - models triplets with multivariate Gaussians
    - models individual entities, and relations, as vectors
    - assume h, r, t are normally distributed
      - mean vectors: $\mu_h, \mu_r, \mu_t \in R^d $
      - covariance matrices: $\Sigma_h, \Sigma_r, \Sigma_t \in R^{d \times d} $
    - uses Kullback-Liebler divergence to directly compare distributions

Semantic matching models

- exploit similarity-based scoring funcitons
- measure plausability of facts by matching latent semantics of entities and relations embodied in their vector space representations
  - RESCAL

Matching using Neural Networks

- train NNs and extract weights as embeddings
  - Semantic Matching Energy (SME)
    - projects entities and their relations to their corresponding vector embeddings
    - representaiton is combined with relations head and tail entities to obtain:
      - $g_1(h, r) = W_{1}^{(1)} \cdot h + W_{1}^{(2)} \cdot r + b_1 $
      - $g_2(t, r) = W_{2}^{(1)} \cdot h + W_{2}^{(2)} \cdot r + b_1 $
        - Ws are R^dxd dimensional weight matrices, and bs are bias vectors
    - output layer:
      - $f_r(h, t) = g_1(h, r)^T \cdot g_2(t, r) $

Future research directions:

- **Hyperbolic geometry** is used to better capture latent hierarchies, commonly present in real-world graphs
  - POG BRO (not new but love to see it)
- KG embedding based on large, multi-topic data collections
  - Linked Data (LD) which standardize and fuse data from different resources
- RDF2vec exploits information in LD and transform it into a learning-suitable format
- other trends explore how symbolic, logical structures could be used during embedding construction
  - KALE
