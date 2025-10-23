# Chapter 4: Propositionalization of Relational Data

Relational learning addresses the task of learning models or patterns from relational data

This chapter complements ILP, by showing how to convert a database into a single-table representation

## 4.1 Relational Learning

Issue with `propositoinal learners` is they require single table. This poses problem for data which is naturally represented in multiple relational tables

*Relational learning* problems cannot be directly represented with a tabular representation without a loss of information.

Learning from relatoinal data can be approached in two main ways:

- Relational learning and inductive logic programming
  - Relational Learning
  - Inductive Logic Programming
  - Relational Data Mining
  - Statistical Relational Learning
  - -> these all learn a relational model or set of relational patterns directly from relational data
  - **algorithm focused**
- Propositionalization
  - transforma relational representation into a propositional single-table representation by constructing *complex features*
    - and then use a propositional learner on the transformed table
  - **data preprocessing focused**

## 4.2 Relational Data Representation

- Explicit
  - Marks the condition (IF) part of the rule
  - Conjunction (AND) of features in the condition of the rule,
  - conclusion (THEN) part
  - ex: `IF Shape == triangle \land Color = red \land Size = big THEN Class = positive`
- Formal
  - Formal propositional logic (with $\implies \land \lor $)
  - ex: `Class = positive \leftarrow Shape = triangle \land Color = red \land Size = Big`
- Logical
  - Most frequently used in ILP
  - mostly used here
  - ex: `positive(X) :-shape(X, triangle), color(X, red), size(X, big)`

### 4.2.1 Michalski's East-West trains challenge dataset

### 4.2.3 Example Using Relational Database Representation

## 4.3 Propositionalization

Workload of finding good relational features is performed by the propositionalization algorithm

Work of finding a good model (by combining the features) is offloaded to traditional ML algorithm

In p18n:

- use entity-relationship diagrams to define types of objects in the domain, where each entity corresponds to a distinct type
- data model constitutes a *language bias* that can be used to restrict the hypothesis space (i.e. space of possible models) to guide search for good models
- in most problems, only individuals and their parts exist as entities
  - meaning the entity-relationship model has a tree structure
- transformation of relational data into single table is good **only** when the problem at hand is *individual-centered*

### 4.3.1 Relational Features

Individual features are described by *literals* or *conjunctions of literals*

- literal: clength(C, short)
- conjunction: clength(C, short), not croff(C, no_roof)

Definition 4.1 (Relational Features)

A relational feature is a *minimal set* of *literals* such that no local (existential) variable occurs both inside and outside the set of literals

### 4.3.2 Automated Construction of Relational Features by RSD

The actual goal of propositionalization is to automatically generate several relevant relational features about an individual that the learner can use to construct a model

`automated feature construction` = `constructive induction`

queries will evaluate as true or false on original data when constructing a transformed data table

RSD:

- There is exactly one free variable that plays the role of the global variable in rules
  - This variable corresponds to the individuals identifier in the individual-centered representation (author, train, etc)
- Each predicate introduces a new (existentially quantified) local variable, and uses either the global variable or one of the local variables introduced by a predicate in one of the preceding literals
  - e.g. the predicate hasCar(T, C) introduces a new local existential variable C for a car of train T
- Properties do not introduce new variables
  - e.g. lshape(L, triangular) stands for triangular shape of load L, provided that load variable L has been already instantiated
- All cariables are 'consumed'
  - i.e. used either by a predicate of a property

Relational feature construction can be restricted by parameters that define:

- the maximum number of literals consituting a feature
- number of local variables
- number of occurrences of individual predicates

### 4.3.3 Automated DAta Transformation and Learning

Once a set of relational features has been automatically constructed, these features start acting as database queries that return true or false values for a given individual

Learning from a transformed data table:

- the learner exploits the feature values to construct a classification model
- if a decision tree learner is used, each node in the tree then contains a feature that has two branches (true and false)
  - to classify unseen individuals, the classifier evaluates the features found in the decision tree nodes

## 4.4 Selected Propositionalization Approaches

In p18n, relational feature construction followed by feature evaluation and table construction is the most common approach to data transformation

- LINUS was a pioneer
  - restricted to the generation of features that do not allow recursion and existential local variables
    - aka the target relation could not be many-to-many and self-referencing
      - the second limitation meant queries could not contain joins (conjunctions of literals)
- aggregation approaches
  - popular alternative to relational feature construction
  - an `aggregate` can be viewed as a function that maps a set of records in a relational database to a single value, which can be done by adding a constructed attribute to one or several tables in the relational db
  - becomes powerful if it involves more tables in the relational db

p18n approaches

- Relaggs
  - "relational aggregation"
  - takes input relational database schema as a basis for declarative bias, using optimization techniques typically used in relational databases (indexes)
  - employs aggregation functions to summarize non-target relations concerning the individuals in the target table
- Stochastic p18n
  - employs search strategy similar to random mutation hill-climbing:
    - alg iterates over generations of individuals which are added and then removed with a probability proportional to the fitness of individuals, where the fitness function used is based on the Minimum Description Length (MDL) principle
- 1BC
  - uses propositional naive bayes classifier
  - first generates set of first-order conditions, then uses them as attributes in the naive bayes classifier
  - transformation is done dynamically, as opposed to standard propositionalization (which is done statically as part of preprocessing)
  - extended by 1BC2, which allows distributions over sets, tuples, and multisets
- Tertius
  - top down rule discovery system, incorporating first-order clausal logic
  - main idea is that no particular prediction target is specified beforehand
  - so, can be seen as an ILP system that learns rules in an unsupervised manner
  - incorporates 1BC
- RSD
  - two main stepa:
    - propositionalization step
    - (optional) subgroup discovery step
  - efficiently produces an exhaustive list of first-order features that comply with user-defined mode constraints
  - satisfy the **connectivity requirement**
    - no feature can be decomposed into a conjunction of two or more features
- HiFi
  - constructs first-order features with hierarchical structure
  - transforms in polynomial time of the maximum feature length
  - resulting features are the shortest in their semantic equivalance class
  - much faster than RSD for longer features
- RelF
  - constructs a set of tree-like relational features by combining smaller conjunctive blocks
  - scales better than other sota p18n algs
- Cardinalization
  - designed to handle not only categorical but also numerical attributes in p18n
  - handles a threshold on numeric attribute values and threshold on number of object simultaneously satisfying the attributes condition
  - can be seen as an implicit form of discretization
- CARAF
  - approaches problem of large relational features search space by aggregating base features into complex compounds, similarly to Relaggs
  - Relaggs tackles overfitting by restricting itself to relatively simple aggregates
  - CARAF incorporates more complex aggregates into a random forest, which ameliorates the overfitting effect
- Aleph
  - ILP toolkit with many modes of functionality
    - learning of theories
    - feature construction
    - incremental learning
  - uses mode declarations to define syntactic basis
- Wordification
  - p18n method inspired by text mining that can be viewed as a transformation of a relational db into a corpus of text docs
  - efficient when used on large relatoinal datasets
  - potential for use in text minimg

## 4.5 Wordification

Other p18n methods construct complex features, wordification generates much simpler features with a goal to achieve greater scalability

### 4.5.1 Outline of Wordification

- input relational db, consisting of one main table and several related tables (with one-to-one or one-to-many)
- each row of main table is the individual

Feature vector construction:

- in first step, new features are constructed as a combination of table name, name of attribute, and its discrete (or discretized value)
  - tableName_attributeName_attributeValue
- these are called *word-items*

For each instance of the main table data:

- one-to-many:
  - get one itemset of word-items in the output vector
- for one-to-many
  - get several itemsets in the output vector

Concatenation and normalization:

- In second step, all constructed itemsets of word-items are concatenated into a joint feature d_i
- output of wordification is a corpus D of 'text documents', consisteing of BoW vectors di constructed for each instance
- in every di, individual word-items are weighted with TF-IDF weights

To overcome loss of information, n-grams of word-items, constructed as conjunction of several word-items, can also be considered

### 4.5.2 Wordification Algorithm

## 4.6 Deep Relational Machines

Problem that propositionalized datasets, represented as sparse matrices, are a challenge for DNNs, which are effective for learning from numeric data by constructing intermediate knowledge concepts improving the semantics of baseline input representations

Sparse matrices resulting from propositionalization are not suitable inputs to DNNs

DRM from Lodhi used feature selection with information-theoretic measures, such as information gain, as measures to address the challenge:

Challenges at intersection of deep learning and relational learning:

- DRMs demonstrated that deep learning on propositionalized relational structures in a sensible approach to relational learning
- Their input is comprised of logical conjuncts, offering the opportunity to obtain human-understandable explanations
- Ideas from the area of repl have only been recently explored in relational context, indicating there are possible improvements both in terms of execution speed, as well as more informative feature construction at the symbolic level
