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
