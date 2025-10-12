# Chapter 1: Introduction to Representation Learning

- core data structure is tabular, which technically is vector

## 1.1 Motivation

- Most best performing ML algs (SVMs and DNNs):
  - Assume numeric data and outperform symbolic approaches in terms of:
    - predictive performance
    - efficiency
    - scalability
- most ml approaches are numeric
  - however humans percieve and describe real world problems mostly in symbolic terms
  - need to transform discrete data into form suitable for numeric learning algs
    - need to maintain similarities between objects to be preserved and expressed as distances in the transformed numeric space
- `embeddings`
  - contemporary transformation approaches
- `propositionalization`
  - symbolic approach to transforming relational instances into feature vectors
- P&E:
  - represent different types of data transformations
  - viewed as two sides of the same coin
  - unifying element:
    - transform input data into a tabular format
    - express relations between objects in the original space as distances (and directions) in the target vector space

## 1.2 Representaition Learning in Knowledge Discovery

replace manual feature learning

### 1.2.1 Machine Learning and Knowledge Discovery

- machine learning:
  - mainly concerned with training classifiers and predictive models from labeled data
- data mining:
  - concerned with extracting interesting patterns from large data stores of mainly unlabeled data
- data science:
  - tries to unify statistic, ML, and data mining methods to understand real-life phenomena through data analysis
- Knowledge Discovery in Databases (KDD) Steps
  - Data Selection
    - developing understanding of application domain, prior knowledge, end user goals, selecting data, variables
  - **Data preprocessing**
    - removing noise or outliers
  - **Data transformation** (focus of this)
    - automated feature engineering (feature extraction or construction)
  - ML or Data Mining
    - applying algorithms
  - Evaluation

Focus:

- Data Preprocessing
  - Data cleaning, instance selection, normalization, missing values, out of range values
  - manual, automated, semi-automated
- Data Transformation
  - Automated representation learning step
  - feature construction for compact data representation

### 1.2.2 Automated Data Transformation

Definition 1.1 (Data Transformation)

Data transformation refers to the automated representation learning step of the KDD process that automatically transforms the input data and the background knowledge into a uniform tabular representation, where each row represents a data instance, and each column represents one of the automatically constructed features in a multidimensional feature space

- data
  - is considered by the learner as the target data from which the learner should learn a model
    - (e.g. classifier in the case of class labeled data)
  - or a set of describptive patterns
    - (e.g. a set of association rules in the case of unlabeled data)
- background knowledge
  - any additional knowledge used by the learner in model or pattern construction from the target data
  - simplest forms:
    - define hierarchies of features (attribute values)
      - such as color "green" being more general than "light green" or "dark green"
  - more complex:
    - any other declarative prior domain knowledge
      - knowledge encoded in:
        - relational databases
        - knowledge graphs
        - domain-specific taxonomies
        - ontologies

## 1.3 Data Transformations and Information Representation Levels

### 1.3.1 Information Representation Levels

- levels of cognitive representations (Gardenfors 2000):
  - neural / sensory
    - sub-conceptual connectionist level
    - informatoin is represented by activation patterns in densely connected networks of primitive units
    - enables concepts to be learned from the observed data by modifying the connection weights between units
  - spatial / conceptual
    - informatoin is represented by points or regions in a conceptual space built upon some dimensions that represent
      - geometrical, topological, orginal properties (of the observed objects)
    - similarity between concepts is represented in terms of distances between points or regions in multidimensional space
    - concepts are learned by modeling the similarity of the observed objects
  - symbolic / language
    - info represented by language of symbols (words)
    - the meaning is internal to the representation itself
      - i.e. symbols have meaning only in terms of other symbols
    - at the same time, their semantics is grounded at the spatial level
    - concepts are learned by symbolic generalization rules

#### Transformations at each level

- `symbolic transformations` aka `propositoinalization`
  - transformations into a symbolic representation space
- `numeric transformations` aka `embeddings`
  - transformations into a spatatial representation space

### 1.3.2 Propositionalization: Learning Symbolic Vector Representations

In symbolic learning, the result of a ML or data mining algorithm is a predictive model of a set of patterns described in a symbolic representation language, resulting in symbolic human-understadable models and patterns

- classification rule learning
- decision tree learning
- association rule learning
- learning logical representations
  - relational learning
  - inductive logic programming

Definition 1.2 (Propositionalization)

- Given:
  - input data of given data type and format, and heterogenous background knowledge of various data types and formats
- Find:
  - a tabular representaiton of the data enriched with the background knowledge, where each row represents a single data instance, and each column represents a feature in a d-dimensional dymbolic feature space $F^d $

The distinguishing property of data transformation via propositionalization (compared to embedding), results in construction of **interpretable symbolic features**

### 1.3.3 Embeddings: Learning Numeric Vector Representations

Statistical ML and pattern-recognition:

- NNs
- SVMs
- random forests
- boosting
- stacking

Some approaches are both symbolic and statistical

- select decision tree learning
- rule learning
- ensemble techniques
  - boosting
  - bagging

Definition 1.3 (Embeddings)

- Given:
  - input data of a given data type and format, and heterogenous background knowledge of various data types and formats
- Find:
  - a tabular representation of the data enriched with the background knowledge, where each row represents a single data instance, and each column represents one of the dimensions in the d-dimensional numeric vector space $R^d $

Clarification:

- `embeddings` mostly refer to dense numeric representations involving **relatively few non-interpretable numeric features**
  - some exceptoins:
    - one-hot-encoding
    - bag-of-words
      - (both these are sparse and encode symbolic features, yet can be treated either as propositionalized vectors or as embeddings, depending on the context of their use)

## 1.4 Evaluation of Propositionalization and Embeddings

Evaluate success of representaiton learning in terms of:

- performance
- interpretability

### 1.4.1 Performance Evaluation

#### Extrinsic evaluation

(of both propositionalization and embeddings)

- quality of transformation is evaluated on downstream learning tasks, such as classifier construction
  - uhh like the better the model using the xforms, the better the features

#### Intrinsic evaluation

- eval on some accessible and computaitonally manageable task
- most common:
  - assessment of whether the similarities of the input entities (training examples) **are preserved** in terms of the similarities of the transformed representations
  - >> seems like we should be using a variety of eval metrics, not just cossim
- no guarantee that good performance on an intrinsic tasks correlates well with performance on a downstream task

### 1.4.2 Interpretability

p18n are mostly used with symbolic learners, whose results are interpretable

## 1.5 Survey of Automated Data Transformation Methods

- P18s methods
  - get tabular data from relational databases
  - do not perform dimensionality reduction
- Graph traversal methods
  - p18s via random walk graph traversal
    - representing nodes via their neighborhoods and communities
- Matrix factorization methods
  - when data is not explicitly presented in the form of relations, but the relations between objects are implicit (given by a similarity matrix) the objects can be encoded in a numeric form using matrix factorization
  - Latent Semantic Analysis
- Neural network-based methods
  - Text embeddings
    - word, sentence, doc vectors
  - Graph and heterogenous information network embeddings
    - use convolutoin filters
  - Entity embeddings
    - these embeddings can use any similarity function between entities to form a prediction task for a NN
    - pairs of entities are used as a training set for the NN, which forecasts whether two entities are similar or dissimilar
    - The weights of the trained network are then used in the embeddings
