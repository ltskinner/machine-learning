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
      - Gagging, random forests
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

Basically, construct a bunch of complex feature relations, *then* bring forward variety of queries q1, ..., 1n, and complete the table to True or False values

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
