# Chapter 6 - Graph Representation Learning

Graph structure, which can represent objects and their relationships, is ubiquitous in big data including natural languages. Besides original text as a sequence of word tokens, massive additional information in NLP is in the graph structure, such as syntactic relations between words in a sentence, hyperlink relations between documents, and semantic relations between entities. Hence, it is critical for NLP to encode these graph data with graph representation learning. Graph representation learning, also known as **network embedding**, has been extensively studied in AI and data mining. In this chapter, we introduce a variety of graph representation learning methods that embed graph data into vectors with shallow or deep neural models. After that, we introduce how graph representation learning helps NLP tasks.

## 6.1 Introduction

Typical non-euclidian data structure

Multiple granularities:

- 1. Word level
  - variety of syntactic/semantic relations
  - words as nodes
  - syntactic/semantic relations as edges
- 2. Document level
  - like wikipedia
- 3. Knowledge level
  - entities as nodes
  - relationships as edges

Graph representation learning aims to learn the low-dimensional representations of nodes or the entire graph. The geometric relationship in the low-dimensional semantic space should effectively reflect the structural information of the original graph, such as the global topological structure or the local graph connection

## 6.2 Symbolic Graph Representation

- nodes V - set
- edges E - set
- graph G = (V, E)

When processing graph data in a computer, we usually represent the connections in a graph in an adjacency matrix

Adjacency matrix $A \in \mathbb{R}^{|\mathbb{V}|\times|\mathbb{V}|} $. If there is any edge between node v and node u, i.e. $(v, u) \in \mathbb{E} $, we have the corresponding element $A_{vu} = A_{uv} = 1; $ otherwise $A_{vu} = A_{uv} = 0 $

For a directed graph, A_uv = 1 indicates there is an edge from node v to node u

For a weighted graph, we can store the weight sof the edges instead of binary values in the adjacency matrix A

In the era of statistical learning, such sumbolic representations are widely used in graph-based NLP such as TextRank

Ajdacency matrices are sparse, and requires VxV storage space, which is usually unacceptable when V gets large

## 6.3 Shallow Node Representation Learning

To address above issues, shallow node representation learning methods propose to map nodes to low-dimensional vectors. Formally, the goal is to learn a d-dimensional vector representation v \in \mathbb{R}^{d} for every node v \in V in the graph

### 6.3.1 Spectral Clustering

Typically computes the first k eigenvectors (or singular vectors) of an affinity matrix, such as adjacency or Laplacian matrix of the graph

#### Locally Linear Embedding (LLE)

Assumes the representaitons of a node and its neighbors lie in a locally linear patch of the maniforls. A nodes representation can be approximated by a linear combination of the representation of its neighbors. LLE uses the linear combination of neighbors to reconstruct the center node

$\mathcal{L}(W, V) = \sum_{v \in V}||v - \sum_{u \in V}W_{vu}u ||^2 $

Where $V \in \mathbb{R}^{|V| \times d} $ is the representation matrix containing all node representations v and Wvu is the learnable contribution coefficient of node u to v. LLE enforces Wvu = 0 if v and u are not connected, i.e. (v, u) \notin E. Further, the summation of a row matrix W is set to 1, i.e. \sum_{u\inV} Wvu = 1.

The equation is solved by alternatively optimizing weight matrix W and representation V. The optimizatino over W can be solved as a least-squares problem. the optimization over V leads to the following optimization problem:

$\mathcal{L}(W, V) = \sum_{v \in V}||v - \sum_{u \in V}W_{vu}u ||^2 $

$s.t. \sum_{v \in V} v = 0, $ $|V|^{-1} \sum_{v \in V} v^\top v = I_{d} $

where Id denotes the dxd identity matrix. The conditions of the 2nd equation ensure the uniqueness of the solution. The first condition enforces the center of all node representations to a zero point, and the second condition guarantees different coordinates have the same scale, i.e. equal contribution to the reconstruction error

The optimization problem can be formulated as the computation of the eigenvectors of matrix $(I_{|V|} - W\top)(I_{|V|} - W) $ which is an easily soveable eigenvalue problem

#### Laplacian Eigenmap

Simply follows the idea that the representations of two connected nodes should be close. The "closeness" is measured by the sequare of the Euclidian distance. We use $S \in \mathbb{R}^{|V| \times |V|} $ to denote the diagonal degree matrix shwere Dvv is the degree of node v. By defining the Laplacian matrix L as the difference between D and adjacency matrix A, we have L = D - A. Minimize the following obj:

$\mathcal{L}(V) = \sum_{\{v, u|(v, u) \in \mathcal{E}\}} ||v - u||^{2} $

$s.t. V^\top DV = I_{d} $

The cost function is the summation of the square loss of all connected node pairs, and the condition prevents the trivial all-zero solution caused by arbitrary scale. Can be refomulated in matrix form as

$V^{*} = arg min_{V^\top DV = I_{d}} tr(V^\top LV) $

where tr(.) is the matrix trace function. The optimal solution V* is the eigenvectors corresponding to d smallest nonzero eigenvalues of L. Note that the Laplacian Eigenmap alg can be easily generalized to the weighted graph

A significant limitatino of both LLE and Laplacian Eigenmap is that they have a symmetric cost function, leading to both algorithms not being applied to direct graphs

#### Directed Graph Embedding (DGE)

Generalizes Laplacian Eigenmap for both directed and undirected graphs based on a predefined transition matrix. For example, we can define a transition probability matrix $P \in \mathbb{R}^{|V| \times |V|} $ where Pvu denotes the probability that node v walks to u. The transition matrix defines a Markovian random walk thought the graph. We denote the stationary value of node v as $\pi_{v} where \sum_{v} \pi_{v} = 1$ The stationary distribution of random walks is commonly used in many ranking algorithms such as PageRank. DGE designs a new cost function that emphasizes those important nodes with higher statinoary values:

$\mathcal{L}(V) = \sum_{v \in V} \pi_{v} \sum_{u \in V} P_{vu} ||v - u ||^{2} $

By denoting M = diag(pi1, pi2, ..., pi_|V|), the cost function can be refomrulated as:

$\mathcal{L}(V) = 2tr(V^\top BV)$

$s.t. V^\top MV = I_{d} $

where

$B = M - (MP - P^\top M)/2 $

This condition is added to remove an arbitrary scaling factor of solutions. Similar to Laplacian Eigenmap, the optimization can also be solved as a generalized eigenvector problem

#### Latent Social Dimensions

Introduce modularity into the cost function instead of minimizing the distance between node representations in previous works. **Modularity is a measurement that characterizes how far the graph is away from a uniform random graph**. Given G = (V,E), we assume that nodes V are divided into n nonoverlapping communities. By 
uniform random graph" we mean nodes connect to each other based on a uniform distribution given their degrees. Then, the expected number of edges between v and u is $deg(v) (deg(u))/(2|\mathcal{E}|) $. Then, the modularity Q of a graph is defined as

$Q = 1/(2|\mathcal{E}) \sum_{v,u} (A_{vu} - (deg(v)deg(u))/(2|\mathcal{E}|)) \delta(v, u) $

where \delta(v,u) = 1 if v and u belong to the same community and delta(v, u) = 0 otherwise. Larger modularity indicates that the subgraphs inside communities are denser, which follows the intuition that a community is a dense well-connected cluster. Then, the problem is to find a partition that maximises the modularity Q

However, a hard clustering on modularity maximization is proved to be NP-hard. Therefore, they relax the problem to a soft case. Let $d \in \mathcal{Z}_{+}^{|V|} $ denotes the degrees of all nodes and $S \in \{0, 1\}^{|V| \times n} $ denotes the community indicator matrix where S_vc indicates node v belongs to community c and S_vc = 0 otherwise. Then, we define modularity matrix B as

$B = A - (dd^\top / 2|\mathcal{E}) $,

and modularity Q can be refomulated as

$Q = (1/2|\mathcal{E}|) tr(S^\top BS) $

by relaxing S to a coninuous matrix, it has been proved that the optimal solution of S is the top-n eigenvectors of modularity matrix B. Then, the eigenvectors can be used as node representations

to conclude, spectral clustering-based methods often define a cost function that is linear or quadratic to the node representations. Then, the problems can be reformulated as a matrix form, and then solved by calculating the eigenvectors of the matrix. However, the computation of eigenvectors for large-scale matrices is both time- and space-consuming, limiting these methods from being applied in real-world scenarios

### 6.3.2 Shallow Neural Networks

With the success of word2vec, many words resort to shallow neural networks for node representation learning. Typically, each node is assigned a vector of trainable parameters as its representation, and the parameters are trained by optimizing a certain objective via gradient descent

#### DeepWalk

novel approach that introduces neural network techniques into graph representation learning for the first time. DeepWalk provides a faster way to learn low-dimensional node representations. The basic idea of DeepWalk is to adapt the well known word representation learning algorithm word2vec by regarding nodes as words and random walks as sentences

Formally, given G = (V, E), DeepWalk uses node v to predict its neighboring nodes in short random walk sequences, $u_{-w}, ..., u_{-1}, u_{1}, ..., u_{w} $ where w is the window size of skip-gram, which can be formulated as

$min \sum_{j=-1, j != 0} -log P(u_{j}|v) $

where it assumes the prediction probabilities of each node are independent and the overall loss integrates the losses of all nodes in every random walk

DeepWalk assigns each node v with two representations: node representation $v \in \mathbb{R}^{d} $ and context representation $v^{c} \in \mathbb{R}^d $. Then, each probability P(u|v) is formulated as a Softmax function over all nodes

$P(u|v) = (\exp(v^\top u^{c}))/(\sum_{Pu' \in V} \exp(v^\top u'^{c}))$

#### LINE

can handle large-scale graphs with arbitrary types:

- (un)directed
- weighted

To characterize the interaction between nodes, LINE models the first-order proximity and second-order proximity of the graph

The modeling of first-order proximity, i.e., observed links, is the modeling of the adjacency matrix. As the adjacency matrix is usually too sparse, the modeling of second-order proximity, i.e. nodes with shared neighbors, can serve as complementary information to enrich the adjacency matrix and make it denser

Formally, first-order proximity between node v and u is defined as the edge weight A_vu in the adjacency matrix. If nodes v and u are not connected, first order proximity is 0

Second order procimity is defined as the similarity between neighbors. Let the row of node v in the adjacency matrix A(v,:) denote the first order proximity between node v and other nodes. Then, the second order proximity between v and u is defined as the similarity between A(v,:) and A(u,:). If they have no shared neighbors, the second-order proximity is 0

To approximate the first order proximity $\hat{P}_{1} (v,u) = (A_{vu})/(\sum_{(v', u') \in \mathcal{E}} A_{v', u'}) $, the joint probability between v and u is defined as

$P_{1}(v, u) = 1/(1 + \exp(-v^\top u)) $

To approximate second order proximity $\hat{P}_{2} (u|v) = (A_{vu})/(\sum_{u'} A_{vu'}) $, the probability that node u appears in v's context P2(u|v) is defined in the same form. In this way, the second order relationship between node representations is bridged by the context representations of their shared neighbors

For parameter learning, the distances between Phat_1(v, u) and P1(v, u) as well as Phat_2(u|v) and P2(u|v) are minimized. In specific, LINE learns node representatinos for first-order and second-order prozimities individually, and then concatenates them as output embeddings. Although LINE can efficiently capture both first-order and second-order local topological information, it can be easily extended to capture higher-order global topological information

#### Connection with Matrix Factorization

We proce that DeepWalk algorithm with the skip gram model is actually factoing a matrix M where each entry Mij is the logarithm of the average probability that noise vi randomly walks to node vj in fixed steps. Intuitively, the matrix M is much denser than the adjacency or Laplaciam matrices, and therefore can help representations encode more structural information. In pracice, the entry value Mij is estimated by random walk sampling, LINE is also proved to be equivalent to matrix factorization, where matrix is defined by the first- and second-order proximities

To summarize, DeepWalk and LINE introduce shallow NNs into graph rep learning. These methods can outperform conventional graph representation learning methods such as spectral clustering-based algorithms, adn are also efficient for large-scale graphs

### Matrix Factorization

Inspired by the connection between DeepWalk and matrix factorization, many research works turn matrix factorization. Note that the eigenvector decomopsition in spectral clustering methods can also be seen as a special case of matrix factorization.

#### GraRep

Directly follows the proof of matrix factorization form of deepwalk. DeepWalk is actually factorizing a matrix M where M = log((A + A^2 + ... + A^K)/(K)). From matrix factorizatino form, DeepWalk has considered the higher-order topological information of the graph jointly. In contrast, GraRep proposes to regard different k-step information separately in graph representatino learning, and can be divided into three steps:

- Calculate k-step transition probability matrix A^k for each k = 1, 2, ..., K
- Obtain each k-step node reps
- Concatenate all k-step node representations as the final node representation

For the second step to obtain the k-step node representaiton, GraRep directly uses a typical matrix decomposition technique, i.e., SDV on A^k. Not efficient especially when k becomes large

#### TADW

The first attributed network embedding algorithm where node features are also available for learning node representations, text-associated DeepWalk (TADW) further generalizes the matrix factorization framework to use text

The main idea of TADW is to factorize node affinity matrix $M \in \mathcc{R}^{|V| \times |V|} $ into the product of three matrixes

does some stuff with TF-IDF

Shallow graph representation learning methods have shown outstanding ability in modeling various kinds of graphs, such as large-scale or heterogeneous graphs. However, still have drawbacks:

- model capacity is usually limited, and leads to suboptimal performance in complex scenarios
- representations of different nodes share no parameters, which makes the number of parameters grow linearly with the number of nodes, leading to computational inefficiency
- shallow graph representation methods are typically transductive, and thus cannot generalize to new nodes in a dynamic graph without retraining

## 6.4 Deep Node Representation Learning

Most deep node representatino learning methods assume node features are available, and stack multiple neural network layers for representaiton learning. Initial feature vector of node v as x_v and the hidden representation of node v at the kth layer as $h_{v}^{k} $

#### 6.4.1 Autoencoder-Based Methods

Structural deep network embedding (SDNE). the main body is a deep autoencoder whose input and output vectors are the initial node feautre x_v and reconstructed feature \hat{x}_{v}. The algorithm takes the representations from the intermediate layer as node embeddings and encourages the embeddings of connected nodes to be similar

Formally, a deep autoencoder first compresses the input node feature into a low-dimensional intermediate vector and then tries to reconstruct the original input from the low-dimensional intermediate vector. hence, the intermediate hidden representation can capture information of the input node since we can recover the input features from them

optimizatino of autoencoder is to minimize the difference between input vector x_v and output vector (reconstructed \hat{x}_{v})

$\mathcal{L}_{1} = \sum_{v \in V} ||\hat{x}_{v} - x_{v}||^{2} $

to encode structure information, SDNE simply requires that the representaitnos of connected nodes shouls be close to each other. Thus, the loss function is

$\mathcal{L}_{2} = \sum_{(v,u) \in E} ||h_{v}^{K} - h_{u}^{K}||^{2} $

Finally, the overall loss functino is $\mathcal{L} = \mathcal{L}_{1} + \alpha \mathcal{L}_{2} $, where \alpha is a harmonic hyperparameter. After the training process, h_V^K is taken as the representation of node v and used for downstream tasks

Experiments show that SDNE can effectively reconstruct the input graph and achieve better results in several downstream tasks. However, the deep nn (deep autoencoder) is isolated with graph structure during the feed-forward computatino, which neglects high-order information interaction among nodes.

### 6.4.2 Graph Convolutional Networks

Aim to generalize convolutional operation from CNNs to the graph domains

The success of CNNs comes from its local connection and multi-layer architectures, which may benefit graph modeling as well

- 1. Graphs are also locally connected
- 2. multilayer architectures can help capture the hierarchical patterns in the graph

However, CNNs can only operate on regular euclidian data like text (1D sequence) and images (2D grid) and cannot be directly transferred to the graph structure. In this subsection, we introduce how GCNs extend the convolutional operation to deal with the non-Euclidian graph data

Mainstream GCNs usually adopt semi-supervised settings for training, while previous graph embedding methods are mostly unsupervised or self-supervised. Here we only introduce the encoder architectures of GCNs and omit their loss functions which depend on downstream tasks. In specific, typical GCNs can be divided into

- spectral approaches
- spatial (nonspectral) approaches

#### Spectral Methods

From the signal processing perspective, the convolutional operation:

- first transforms a signal to the spectral domain
- then modifies the signal with a filter
- finally projects the signal back to the original domain

spectral GCNs follow the same process and define the convolution operator in the spectral domain of graph signals

Formally, d-dimensional input representations $X \in \mathbb{R}^{|V| \times d} $ of a graph can be seen as d graph signals, then spectral GCNs are fomulated as

$H = \mathcal{F}^{-1}(\mathcal{F}(g) \odot \mathcal{F}(X)) $

Where H is the representaiton of all nodes, g is the filter in the spatial domain, F(.) and F-1(.) indicate the graph Fourier transform (GFT) and inverse, respectively, which can be defined as

$\mathcal{F}(X) = U^\top X, \mathcal{F}^-1 (X) = UX $

where U is the eigenvector matrix of the normalized graph Laplacian $L = I_{|V|} - D^{-1/2}AD^_{-1/2} $. A is adjacency matrix and D is degree matrix

In practice, we can use a learnable diagonal matrix g_{\theta} to approximate the spectral graph filter F(g) and the graph convolutional operation can be refomrulated as

$H = U g_{\theta} U^\top X $

Intuitively, initial graph signals X are transformed int ospectral domain by multiplying U^\top. Then filter g_\theta performs the convolution, and U projects graph signals back to their original space

Some limitations:

- filter g_theta is not directly related to the graph structure
- the kernel size of the graph filter grows with the number of nodes in the graph, which may cause inefficiency and overfitting issues
- the calculation of U relies on computationally inefficient matrix factorization

##### ChebNet

Approximates Ug\thetaU6\top by Chebyshev polynomials T_k(.) up to Kth order, which involve information within K-hop neighborhood. In this way, ChebNet does not need to compute matrix U, and the number of learnable parameters is related to K instead of |V|

##### GCN

GCN is a first order approximation of ChebNet. GCN revelas that ChebNet may suffer from the overfitting problem when handling graphs with very wide node degree distributions. Hence, GCN limits the maximum order of Cheb polynomials to K = 1, and the equation is simplified to the following form with tro trainable scalars $ \theta_{0}' $ and $\theta_{1}' $

$\theta \in \mathbb{R}^{K} $ is a weight vector indicating Cheb coefficients

$H = \theta_{0}' X + \theta_{1}'(L - I_{|V|} X) $

$  = \theta_{0}' - \theta_{1}' D^{-1/2}AD^{-1/2} X $

GCN futher reduces the number of parameters to address overfitting by setting $\theta_{0}' = -\theta_{1}' = \theta $

To summarize, spectral GCNs can effectively capture complex global patterns in graphs with spectral graph filters. However, the learned filters of the spectral methods usually depend on the Laplacian eigenbasis of the graph structure. This leads to the problem that a spectral-based GCN model learned on a specific graph cannot be directly transferred to another graph

#### Spatial Methods

Spatial GCNs define graph convolutional operation by directly aggregating information on spatially close neighbors, which is also known as the message-passing process. The representation h_v^k of node v at kth layer can be seen as a function aggregating the representations of v and its neighbors at (k-1)-th layer

$h_{v}^{k} = f(\{h_{v}^{k-1}\} \cup \{h_{u}^{k-1} | \forall u \in \mathcal{N}_{v} \}) $

where $\mathcal{N}_{v}$ is the neighbor set of node v

The key challenge of spatial CGNs is how to define the convolutional operation to satisfy the nodes with different degrees while maintaining the local invariance of CNNs

##### Neural FPs

propose to use different weight matrixes for nodes with different-sized neighborhoods

Neural FPs require learning weight matrixes for all node degrees in the graph. Hence, when applied to large-scale graphs with diverse node degrees, it cannot capture the invariant information among different node degrees, and needs more parameters as node degrees get larger

##### GraphSAGE

transfers GCNs to handle the inductive setting, where the representations of new nodes should be computed without retraining. Instead of utilizing the full set of neighbors, GraphSAGE learns graph representations by uniformly sampling a fixed-size neighbor set from each nodes local neighborhood

$h_{\mathcal{N_{v}}}^{k} = Aggregate({h_{u}^{k-1} | \forall u \in \mathcal{N}_v }) $

$h_{v}^{k} = \sigma(W6{k} concat(h_{v}^{k-1}; h_{\mathcal{N_{v}}}^{k} )) $

where $\mathcal{N_{v}}$ is the sampled neighbor set of node v and the aggregator functions Aggregate(.) usually utilize the following three types:

- Mean aggregator
  - by utilizing a mean-pooling aggregator, can be viewed as the inductive version of the original transducive GCN framework
  - $h_{v}^{k} = \sigma(W^{k} \cdot Mean(\{h_{v}^{k-1}  \} \cup \{h_{u}^{k-1} | \forall u \in \mathcal{N_{u}}  \})) $
- Max-pooling aggregator
  - first feeds each neighbors hidden representation into a fully connected layer and then utilizes a max-pooling operation to the obtained representaitons of the nodes neighbors
  - $h_{\mathcal{N_{u}}}^k = Max(\{\sigma(W^{k} h_{u}^{k-1} + b^{k}) | \forall u \in \mathcal{N_{u}}  \}) $
- LSTM aggregator
  - stronger expressive capability
  - since LSTMs process sequentially, GraphSage randomly permutes node v's neighbors to adapt LSTMs

GNS have shown superior abilities over autoencoder based methods. Even now, we can still get sota performance by equipping vanilla GCNs with proper training strategies or knowledge distillation methods

### 6.4.3 Graph Attention Networks

#### GAT

proposes to adopt the self-attention mechanisms for the information aggregation of GNNs

$h_{v}^{k} = \sigma (\sum_{u \in \mathcal{N_{v} \cup \{v\}}} \alpha_{vu}^k W^{k} h_{u}^{k-1} ) $

where \sigma (.) is a nonlinear function and \alpha_{vu}^{k} is the attention coefficient of node pair (v,u) at the kth layer, which is normalized over nmode v's neighbors:

$\alpha_{vu}^{k} = (\exp(LeakyReLU(a^\top concat(W^{k} h_{v}^{k-1}; W^{k} h_{u}^{k-1})))) / (\sum_{m \in \mathcal{N_{u}} \cup \{v \}} \exp(LeakyReLU (a^\top concat( W^{k} h_{u}^{k-1}; W^{k} h_{m}^{k-1})))) $

Where W^k is the weight matrix of a shared linear transformation applied to every node, a is a learnable weight vector and LeakyReLU(.) is a nonlinear function

Moreover, GAT utilizes the multi-head attention mechanism to further aggregate different types of information. Specifically, GAT concatenates (or averages) the output representaitons of M independent attention heads:

$h_{v}^{k} = \|_{m=1}^{M} \sigma (\sum_{u \in \mathcal(N_{u}) \cup \{v \}} \alpha_{vu}^{k,m} W_{m}^{k} h_{u}^{k-1} ) $

where $\alpha_{vu}^{k,m} $ is the attention coefficient from the mth attention head at the kth layer, W_m^k is the transform matrix from the mth attention head, and $\|$ is the concatenation operation

by incorporating the attention mechanism into the information aggregating phase of spatial GCNs, GATs can assign proper weights to different neighbors and offer better interpretability. The ensemble of multiple attention heads further increases the model capacity and brings performance gains over GCNs
