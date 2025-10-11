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
