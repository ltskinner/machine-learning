# Chapter 1. Introduction

## 1.1 What is a graph

- if undirected graph, the adjacency matrix A will be symmetric

### 1.1.1 Multi-relational graphs

- different types of edges
- edge notation:
  - $(u, \tau, v) \in E $ where edge type is $\tau $
  - define one adjacency matrix per edge type $A_{\tau} $
- entire `multi-relational graph` summarized as an `adjacency tensor`
  - $A \in \mathbb{R}^{\|V\| \times \|R\| \times \|V\| } $
    - where R is the set of relations

Two types of multi-relational graphs:

- `heterogeneous graphs`
  - nodes also have types
    - can partition nodes into disjoint sets
      - $V = V_1 \cup V_2 \cup ... \cup ... V_{k} $ where $V_i \cap V_j = \emptyset, \forall i \neq j $
  - edges in hetero geaphs generally satisfy constraints according to the node types
    - the most common constraint is that certain edges only connect nodes of certain types, i.e.
      - $(u, \tau_{i}, v) \in E \implies u \in V_j, v \in V_k $
  - `multipartitle graphs`
    - special case of hetero graphs
    - edges can only connect nodes that have different types, i.e.
      - $(u, \tau_{i}, v) \in E \implies u \in V_j, v \in V_k \land j \neq k $
- `multiplex graphs`
  - assume grpah can be decomposed in a set of *k layers*
  - evey node assumed to belong to every layer
    - each layer corresponds to a unique relation - which represent the *intra-layer* edge type for that layer
    - assume that *inter-layer* edge types can exist, which connect the same node across layers
  - examples:
    - multiplex transportation network
      - node represent city
      - each layer is different mode of transportation (air, train, road)
      - intra-layer edges would represent cities that are connected by different modes of transportation
      - inter-layer edges represent the possibility of switching modes of transportation within a particular city

### 1.1.2 Feature Information

*attribute* = *feature* information associated with a graph (e.g. profile picture associated with a user in a social network)

most often these are node-level attributes, represented in a single real-valued matrix $X = \in R^{\|V\| \times m } $, where we assume the ordering of the nodes is consisstent with the ordering in the adjacency matrix

- in hetero graphs, assume that each different type of node has its own distinct type of attributes
- in rare cases, also consider graphs that have real-valued edge features in addition to discrete edge types, and even some cases where we associate real-valued features with entire graphs

Graph or network? network is an instantiation of a graph

## 1.2 Machine Learning on Graphs

supervised and unsupervised

### 1.2.1 Node Classification

e.g. classifying users as bots 0_0

- goal: predict label $y_u $ for all nodes $u \in V $, which can be:
  - type, category, attribute
- based on training set of nodes which have true labels $V_{train} \subset V $
  - assumes have labeled information for a small set subset of nodes for the entire graph
    - also instances of node classification that involve many labeled nodes and/or that require generalization across disconnected graphs

Some key differences from standard supervised classification:

- most important diff:
  - nodes in graph are **NOT** *independent and identically distributed (i.i.d)*
    - typical assumption for supervised ML
    - assume that each data point is statistically independent from all the other data points
      - otherwise, may need to model the dependencies between inputs
    - also assumed that identically distributed (balanced)
      - for node class, we are modeling an interconnected set of nodes

Interestingly, key insight behind many of the most successful node classification approaches is to explicitly leverage the connections between nodes

- popular idea: exploit `homophily`
  - the tendency for nodes to share attributes with their neighbors in the graph
- based on this, build models to assign similar labels to neighboring nodes in graph
  - other concepts like
    - `structural equivalence`
      - nodes in similar local neighborhood structures will have similar labels
    - `heterophily`
      - presumes that nodes will be preferentially connected to nodes with different labels

When we build node classification models we want to exploit these concepts and model the relationships between nodes, rather than simply treating nodes as independent data point

Also, many researchers often refer to it as *semi-supervied* learning because of the atypical nature of node classification

- this is bc when training node class models, we usually have access to the full graph, which includes all the unlabeled (test) nodes
- here, we can use informatoin about the test nodes to improve the modesl during training
  - which differs from traditional supervised where unlabeled data points are completely unobserved during training

lol "Machine learning tasks on graphs do not easily fit our standard categories"

### 1.2.2 Relation Prediction

What about when we are missing relationship information?

Other names:

- link prediction
- graph completion
- relational inference
- relation prediction

Std setup - set of nodes $V$ and incomplete set of edges between nodes $E_{train} \subset E $

- goal: use partial information to infer missing edges $E \setminus E_{train} $
- complexity of the task is highly dependent on the type of graph
  - simple graphs, simple heuristics work well
  - complex multi-relational graphs require complex reasoning and inference strategies

Typically requires inductive biases that are specific to the graph domain, breaking traditional ML un/supervised categories

### 1.2.3 Clustering and community detection

Both node classification and relation prediction require inferring *missing* informatoin

`community detection` is the graph analogue of unsupervised clustering

Looking at google scholar, make a `collaboration graph` that connects two researchers if they have co-authored a paper together

We expect grpahs to exhibit `community` structure, where nodes are much more likely to form edges with nodes that belong to the same community - this is the underlying intuition

`community detection` goal: infer latent community structures given only the input graph $G = (V, E) $

### 1.2.4 Graph classification, regression, and clustering

example: build classification model to detect whether a computer program is malicious by analyzing a graph-based representation of its syntax and data flow

- given multiple different graphs
- goal: make independent predictions specific to each graph

in `graph clustering`: learn an unsupervised measure of similarity between pairs of graphs

The challenge is how to define useful features that take into account the relational structure within each data point (each graph basically)
