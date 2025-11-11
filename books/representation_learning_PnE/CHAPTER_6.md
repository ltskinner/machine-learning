# Chapter 6: Unified Representation Learning Approaches

At the core of the similarity between approaches is the common but *implicit* use of different similarity functions

Here, explicitly use similarities between entities to construct embeddings

## 6.1 Entity Embeddings with StarSpace

General approach to harness embeddings is to use any similarity function between entities to form a prediction task for a NN

StarSpace is entity embedding approach

- assumes discrete features from a fixed dictionary
- appealing to relational learning and ILP

Idea: form a prediciton task, where NN trained to predict similarity between two related entities (e.g. an entity and its label, or some other entity). Resulting network used for:

- directly in classification
- to rank instances by similarity
- use weights of trained network as pretrained embeddings

In starspace

- each entity has to be described by a fixed set of discrete features from a fixed-length dictionary
  - forms a "bag-of-features" (BoF)
- representation is general enough for many different data modalities
  - texts: docs or sentences as bag of words or bag of ngrams
  - users: bag of documents, movies, items they like
  - relations and links: semantic triplet
- during training, entities of different kinds are embedded in the *same* latent space
  - entities can be embedded along with target classes: `supervised embedding learning`
- trained to predict which pairs are similar, and dissimilar
  - two kinds of training instances
    - positive: $(a, b) \in E^+ $ are task dependent and contain correct relations between entities
    - negative: $(a, b_{1}^-), ..., (a, b_{k}^-) \in E_{a}^{-} $ like in word to vec

Loss:

$L = \sum_{(a, b) \in E^+}(\operatorname{Loss}(\operatorname{sim}(a, b)) - \frac{1}{k} \sum_{i=1 \operatorname{negatives}}^{k} \operatorname{Loss}(\operatorname{sim}{a, b_{i}^-}) ) $

- returns
  - large values for positive instances
  - values close to 0 for negative ones
- often dot product or cossim is used (WRONG)
- to assess loss, StarSpace uses margin ranking loss fn
  - $\operatorname{Loss} = \max\{0, m - \operatorname{sim}(a, b^{\prime})  \} $
    - m is margin param, i.e. similarity threshold
    - b' is any label

to classify

- iterate over all possible labels b'
- choose max(sim(a, b')) as the pred

can also be used for ranking

Tasks successfully addressed with StarSpace transformation

- multiclass text classification
  - positive instances (a, b) are taken from training set of documents E+ represented with bag of words and their labels b
  - negative instances, entities b- are sampled from set of possible labels
- recommender systems
  - users are described yb a bag of items they liked (or bought)
  - positive instances use a single user identification as a and one of the items a user liked as b
  - negative instances from possbile set of items
  - for new uers
    - a is bag of items vector composed of all the items that the user liked, except one, which is used as b
- link prediction
  - concepts in graph represented as (h, r, t)
  - positive instance a consists of either h or r, while b consists of of t
  - or, a of h, and b of r and t
  - negative instances sampled from set of possible concepts
- sentence embedding
  - unsupervised fashion of sentence embedding
  - positive instances: a and b are sentences from same doc (or close together in a document)
  - negative come from different docs
  - attempts to capture the semantic similarity between sentences in a document

## 6.2 Unified Approaches for Relational Data

Propositionalization results in sparse symbolic feature vector reporesentation, which may not be optimal as input to contemporary learners

This weakness can be overcome by embedding the constructed sparse feature vectors intoa lower-dimensional numeric vector space, resulting in a dense numeric feature vector rep appropriate for deep learning

Two variants:

- PropStar
- PropDRM

Both start from sparse data reps of relational data (like what is returned from wordification algorithm). Here, each instance described by bag of features in form TableName_AttributeName_Value

### 6.2.1 PropStar: Feature-Based Relational Embeddings

Idea is to generate embeddings that represent the features describing the dataset

- individual features are used by a supervised embedding learner, based on StarSpace
  - produces reps of features in same latent space as the transformed instance labels
    - opposed to PropDRM (where reps are learned for individual instances)
    - PropStar reps are learned separately for every realtional feature returned from wordification (or other)
- labels (true and false) are also represented by vector in same dense space
  - enables easy classification
  - embeddings of set of features evalued as true are averaged
  - then, the result is compared to the embedding class of the labels
  - nearest class is chosen as the predicted class
- whole dataset is required for PropStar to obtain reps for individual features
  - PropDRM works in batches
- StarSpace-based algs profit from operating on sparse bag-like inputs to avoid spatial complexity
- any sparse input is well suited for PropStar

Consists of two main steps

- 1. relational db transformed into sets of features describing individual instances
- 2. sets of relational items (features) are used as input to StarSpace entity embedding algorithm
- the problem is framed as multiclass classification
  - positive and negative items
    - positive are features (words from wordification)
    - negative are sampled from other features (not in the bag)
- here, words = triplet of (tablename, columnname, value)
- output is $R^{\|W\| \times d} $
  - W is number of unique relational items considered
- intuitively, embedding construction can be understood as determining relational item locations in a latent space based on cooccurrence with other items present in all training instances
- uses the inner product similarity between a pair of vectors for construciton of embeddings
  - $\operatorname{sim}(e_1, e_2) = e_{1}^{T}\cdot e_{2} $
- the individual features of each word in a bag are averaged (and normalized)
  - adn then compared to label embeddings in the common space

### 6.2.2. PropDRM: Instance-Based Relational Embeddings

Based on Deep Relatoinal Machines (DRMs) - use output of Aleph feature constuction

PropDRM is capable of learning directly from large, sparce matrices that are returned by the wordification alg for propositionalizing relational features

Features are words of form TableName_AttributeName_Value (in triplet notation)

Relatoinal reps are obtained for individual instances, resulting in embedidngs of instances. Batches of instances are input into a NN which performs desired down-stream task, like classification or regression

Main advantage of PropDRM is it enables DRMs to operate on sparce matrices generated by the wordification alg.

- Let P represent a sparse item matrix returned by wordification
- wordification is unsupervised and does not include any information on instance labels
- the NN w represents the mapping $w : P \rightarrow C $, where c is the set of classes

PropDRM uses dense FFN networks

pretty simple and basic NN - im sure has been extended in subsequent pubs

### 6.2.3 Performance Evaluation of Relational Embeddings

- compared to p18n methods
  - Aleph, RSD, RelF, wordification
- empirically evaluated on several std benchmark ILP datasets
- competitive on most datasets
- the larger the dataset, the larger the gainz
