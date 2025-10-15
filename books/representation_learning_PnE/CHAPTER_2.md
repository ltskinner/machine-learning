# Chapter 2: Machine Learning Background

## 2.1 Machine Learning

- Traditionally, concerned with discovering models, patterns, and other regularities in data

Supervised learning:

- samples are labeled by a discrete class label
- or, by numeric prediction value

Unsupervised learning:

- cluster or discover descriptive patterns that hold for the dataset

Semi-supervised learning

- some samples are labeled by class labels
- some sample have missing values of the class attribute

Misc:

- association rule learning
  - aim to find interesting relationships between variables, using different measures of interestingness
- subgroup discovery algorithms
- generative learning
  - tries to learn underlying distribution of data
  - and generate new instances from it
- reinforcement learning
  - learn actions in env that maximize cumulative reward

Key for this book:

- Binary class learning
  - concept learning or binary classification
  - class label is binary
  - each sample labeled with one of two discrete class values
- Multiclass learning
  - class attribute has more than two discrete class values
- Multi-label learning
  - several class labels
  - relaxes mutual-exclusiveness of class
  - hierarchical multi-label learning
- Multi-target learning
  - in contrast with standard supervised learning, tries to learn several related target attributes simultaneously with a singl emodel
  - goal is to achieve transfer of knowledge between tasks
  - exploits commonalities and differences across tasks to reduce overfitting and improve generalization

### 2.1.1 Attributes and Features

- most feature types only involve a single attribute and can thus be directly mapped to propositoinal features
- `relational features` differ because they relate the values of two (or more) different attributes to each other

### 2.1.2 Machine Learning Approaches

early approaches able to handle only data in single table format

- perceptrons
- decision tree learners
  - ID3
  - CART
- Rule learners
  - AQ
  - INDUCE

Grouped into two categories: `symbolic` and `statistical` approaches

- Symbolic approaches
  - inductive learning of symbolic descriptions, such as:
    - rules
    - decision trees
    - logical representations
- Statistical approaches
  - k-nn
  - instance learning
  - NNs
  - SMVs (is this really statistical? I guess)

### 2.1.3 Decision and Regression Tree Learning

Decision Tree

- classification model whose structure consistes of a number of *nodes* and *arcs*
  - simply:
    - node is an attribute
    - outoing arc correspond to values of the attribute
  - leaf nodes labeled by a prediction
  - constructed in top down manner
  - nodes/attributes selected by *purity* of class distribution
    - (the degree to which a node contains instances of a single class)
    - attributes with greatest utility are selected for tree expansion
    - expansion stops when number of instances in node is low, or all instances are labeled with the same class value
  - "recursive partitioning"
    - number of examples in each successive node steadily decreases
    - statistical reliability of chosen attributes decreases with increasing depth of tree
    - overly complex models may be generated, but not generalize well - overfitting

### 2.1.4 Rule Learning

Classification rule learning

- initially focused on learning predictive models consisting of set of classification rules of form:
  - TargetClass $ \longleftarrow $ Explanation
  - where explanation is logical conjunction of features (attribute values) characterizing the given class
  - `association rule learning`

Subgroup discovery

- aim to find interesting sets of patterns as sets of rules that best describe the target class
- similar to symbolic rule learning, also builds classification rules
  - however, each rule shall describe an interesting subgroup of target class instances
  - so, key difference is in rule selection and model evaluation criteria

### 2.1.5 Kernel Methods

Representation space given by raw data is often insufficient for successful learning of predictive models

To transform, a class of kernel methods requires only a similarity function (a `kernel function`) over pairs of data points in the raw representation

- Let $\phi(x) $ be a transformation of instance x into the feature space
- then, the kernel function on instances $x_1$ and $x_2 $ is defined as:

$k(x_1, x_2) = \phi(x_1)^T \cdot \phi(x_2) $

- here, both $\phi(x_1)^T $ and $\phi(x_2) $ are both feature vectors
- the simplest kernel is a **linear kernel** defined as:
  - $\phi(x) = x $
  - in which case, $k(x_1, x_2) = x_1^T \cdot x_2 $ is the dot product

Another family of populat kernels are the `Radial Basis Functions` which use the negative distance between arguments as the magnitude of similarity. Typically defined as:

$k(x_1, x_2) = f(-\|x_1 - x_2 \|^2) $

The most frequent choice is the Gaussian kernel, defined as:

$k(x_1, x_2) = \exp(-\gamma\|x_1 - \|x_2\|^2) $

Where $\|x_1 - \|x_2\| $ is the Eucliden distance, and $\gamma $ is a parameter

Kernel fns allow learning in a **high dimensional, implicit feature space without computing the explicit representation of the data in that space**

- instead, kernel fns only compute the inner products between the instances in the feature space
- this implicit transformation is often computationally cheaper than the explicit computation of the representation
- the idea of the kernel trick is to:
  - formulate the learning alg in such a way that it uses *only* the similarities between objects formulated as the inner product (i.e. dot product or scalar product)
  - an algebraic operation that takes two equal-length sequences of numbers and returns a single number computed as the sum of th eproducts of the corresponding entries of the two sequences of numbers
  - >> this is pretty epic, I think we should be able to use this super abstractly

Uses:

- SVMs abuse this
- PCA
- gaussian processes
- ridge regression

### 2.1.6 Ensembe Methods

Two flavors:

- homogeneous predictors
  - typically use multiple tree-based models (such as decision or regression trees)
  - can be further split into:
    - averaging methods
      - Bagging, random forests
    - boosting approaches
      - sequentially adds new ensemble members to reduce errors of previous members
- heterogeneous predictors
  - various successful predictors (nns, rfs, SVMs)

"averaging" = "voting", where voting is special case of averaging for discrete class attributes

- heterogenous ensemble methods can use voting
  - stacking is more successful
  - captures the output of several learners
  - then, a meta model is trained on dataset of outputs from all original learners
- to have good ensembles, need classifiers are diverse, yet accurate

boosting algorithms:

- original AdaBoost
  - constructs series of base learners by weighting their training set of examples according to correctness of prediction
    - correctly predicted examples have weights decreased
    - incorrect predictions increase weights of such instances
    - this way, subsequent base learners receive effectively different training sets
      - and, gradually come to focus on most problemmatic instances
    - usually, tree based models such as "decision stumps" are used as base learners
- XGBoost
  - uses a regularized obj fn and subsampling of attributes, which helps to control overfitting
  - the stated optimization problem is solved with gradient descent
  - parallelizable and highly scalable
- Random Forests
  - construct series of decision tree-based learners
    - each base learner receives a different training set of n instances (drawn with replacement)
    - in each node, the splitting attribute is selected from a randomly chosen sample of attributes
    - proven to not overfit and are less sensitive to noisy data (compared to original boosting)
  - b/c training sets of individual trees are constructed by `bootstrap replication`:
    - there is 1/e = 1/2.718 = 36.8% of instances not taking part in construction of tree
    - these are called "out-of-bag" instances
      - useful for internal estimates of error, attribute evaluation, and outlier detection
- gcForest
  - builds a deep forest with cascade structure, which enables representation learning by forests
  - the cascade levels are determined automatically based on the dataset
  - is competitive to DNNs on many problems, but require fewer data and hyperparameters

### 2.1.7 Deep Neural Networks

activation fns:

- sigmoid
- relu
- softplus
- exponential linear unit (ELU)

For non-linear activation fns, shown that NNs can predict any continuous function (possibly using exponentially many eurons)

CNNs:

- mathematical convolution
- nxn filters, w various pooling methods
- adv of CNNs is that filters are learned from the data, so no need to manually engineer them based on knowledge of which patterns might be important

## 2.2 Text Mining

- deals with construction of models and patterns from text resources
  - categorization, clustering, taxonomy construction, sentiment analysis
  - information retrival, information extraction, knowledge management
- document features
  - words
  - terms
  - concepts (most semantically loaded)
- preprocessing pipelines
  - tokenization
  - stop-word removal
  - lemmatization
  - pert of speech tagging
  - dependency parsing
  - syntactic parsing

## 2.3 Relational Learning

Standard ML and data mining focus on models or patterns from single data table. Relational learning require use of specific learners that can handle multi-tabular data

- propositional data representation, where each instance is represented with a fixed-lengh tuple, is natural for many ML problems
  - where there is no natural propositional encoding, some problems require a relatoinal representation

Oh interesting - so like this is like joins with foreign keys, where:

- (personID) ^ (companyID) -> (employeeId)
  - using conjunctions and whatnot

Some approaches:

- Relational Data Mining (RDM)
  - discovered patterns/models are expressed in relational formalisms
- Inductive Logic Programming (ILP)
  - discovered patterns/models are expressed in first order logic

FOIL (Quinlan 1990) was early relatoinal learning algorithm

ILP community

- developed algorithms for learning clausal logic
  - Progol
  - Aleph (A learning engine for proposing hypotheses)
    - >> see Aleph (Srinivasan 2007)

### Propositionalization

In p18s, learning is performed in two steps:

- 1. Constructing complex relational features, transforming the relational representation into a propositional single-table format
- 2. Applying a propositional learner on the transformed single-table representation

Basically, construct a bunch of complex feature relations, *then* bring forward variety of queries q1, ..., qn, and complete the table to True or False values

Note: the propositoinal representations (a single table format) impose the constraint that: each training example is represented by a single fixed-length tuple; the transformation into propositoinal representation can be done with many ML or data mining tasks in mind (such as classification, association discovery, clustering, etc)

However, p18s, which changes the representation for relational learning into a single-table format, **cannot always be done without the loss of information**

P18s is powerful method when problem on hand is *individual-centered*

- involving one-to-one or one-to-many relationships
  - these have clear notion of an individual
  - learning occurs only at the level of (sets of) individual instances
    - rather than the (network of) relationships between instances (which occur in many-to-many)

For some relational problems, there may not exist an elegant propositoinal encoding.

- e.g. a citation network cannot be represented in a prop format without the loss of information since each author can have any number of co-authors and papers
  - here, the problem is naturally represented using multiple relations

## 2.4 Network Analysis

For non-tabular data, especially when instances are interconnected to a varying (non-fixed) number of other instances. Representing each connection from an instance as a separate col would result in a different number of columns for each row. Alternatively, if we encoded connections of an instance with a single column, columns would have to contain composite data structures of varying length (such as lists)

Alternative is to represent as a network

Network analysis grounded in:

- mathematical graph theory
- social science

Information networks: directed connections encode flow of information between nodes

### 2.4.1 Selected Homogeneous Network Analysis Tasks

Network node classification

- given a network and class labels for some of the network entities
  - predict the class labels for the rest of the tntities in the network
    - known as `label propagation`

Link Prediction

- focuses on unknown connections between the entities
- assume some edges not known
- some approaches
  - assign score s(u,v) to each pair of vertices u and v, which models probabilitiy of the connected vertices
  - calculate score as product of vertex degrees
  - number of common neighbors of two vertices

Community detection

- community summarized as:
  - a group of network nodes, with dense links within the group and sparse links between the rest of the groups

Network node ranking

- objective of ranking in information networks is to assess the relevance of a given object:
  - globally (concerning the whole graph)
  - locally (relative to some object in graph)
- PageRank is one example
  - random walk style - each edge has some probability of it
    - the PageRank of the vertex is the expected proportion of time the walker spends in the vertex
  - score propagation

### 2.4.2 Selected Heterogeneous Network Analysis Tasks

Authority ranking

- rank vertices of a bipartite network

Ranking based clustering

- joins ranking and clustering
  - RankClus
    - for bipartite information networks
  - NetClus
    - networks with star schema
  - both of these
    - cluster entities into certain types
    - then rank entities within clusters

Classification through label propagation

- find probability distribution of node being labeled with a positive label
- GNETMINE
  - idea of knowledge propagation through heterogeneous informatoin network to find probability estimates for labels of the unlabeled data

Ranking based classification

- builds on GNETMINE:
  - RankClass relies on within-class ranking functions to achieve better classification results

Relational link prediction

- expanding ideas of link prediction from homogeneous information networks
  - link prediction for each pair of object types in the network
  - where the score is higher if the two objects are likely to be linked

Semantic link association prediction

- Semantic Link Assocaiation Prediction (SLAP) (bro lmao)
  - statistical model
  - measures associations between elements of a heterogeneous network

### 2.4.3 Semantic Data Mining

Domain specific knowledge (background knowledge) usually structured as taxonomies (or some more elaborate knowledge structures curated by human experts)

Ongologies are directed acyclic graphs, formed of concepts and their relations, encoded as subject-predicate-object (s, p, o) triplets

`Semantic Data Mining (SDM)` is when learning is performed on the knowledge structure directly

- goal:
  - find descriptions of target class instances as a set of rules of the form TargetClass $\Longleftarrow $ Explanation
    - where the explanation is a logical conjunction of terms from the ontology
    - this has roots in symbolic rule learning, subgroup discovery, and enrichment analysis

Semantics-aware learning:

- entity resolutoin
- heterogeneous network embedding
- author profiling
- recommendation systems
- ontology learning

### 2.4.4 Network Representation Learning

Process of:

- transforming a network
  - topology
  - node and edge labels
- into a vector representation format

In transformed vector representation:

- rows correspond to individual network nodes
- columns correspond to automatically constructed features used for describing properties of nodes
- (rarely) rows correspond to edges and columns to their properties

Knowledge Graphs babyy

## 2.5 Evaluation

- k-fold cross-validation

### 2.5.1 Classifier Evaluation Measures

- tp, fp, fn, tn
- acc = (tp + tn) / all
- error = 1 - acc
- precision
- recall
- f-measure

### 2.5.2 Rule Evaluation Measures

For data mining, including association rule learning and subgroup discovery, assess rule quality by computing `coverage` and `support` of each individual rule R

- Coverage(R) = |TP| + |FP|
  - total number of instances covered by the rule
- Support(R) = (|TP| + |FP|) / |all|

Both coverage and support assume binary class and return values in [0,1] range, where larger values mean better rules

- Lift(R) = Support(R) / $\hat{p} $
  - frequently used measure
  - ratio between support and expected support
  - $\hat{p} = \frac{\|P\|}{\|E\|} $
    - corresponding to the support of the empty rule that classifies all examples as positive
  - range is [0, \inf], where larger values indicate better rules

To eval the whole rule set, treat rule set as any other classifier, using eval measures such as class accuracy

## 2.6 Data Mining and Selected Data Mining Platforms

### 2.6.1 Data Mining

- ML for supervised learning
- data mining for pattern mining

Notable approaches

- association rule learning
- `mining closed itemsets` and `maximal frequent itemsets` for unlabeled data
  - `itemset` is **closed** if none of its immediate supersets has the same support as the itemset
  - `itemset` is **maximally frequent** if none of its immediate supersets is frequent
  - can be used to analyze labeled data
    - adapt closed sets mining to task of discriminating different classes by contrasting covering properties on the positive and negative samples
- subgroup discovery is cool
  - aims to find interesting patterns as sets of individual rules that best describe the target class
- contrast set mining
  - learns patterns that differentiate one group of instances from another
- emerging pattern mining algorithms
  - construct itemsets that significantly differ from one class to the other
  - I feel like there might be something here...

### 2.6.2 Selected Data Mining Platforms
