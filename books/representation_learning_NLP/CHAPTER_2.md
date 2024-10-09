# Chapter 2 - Word Representation Learning

Words are the building blocks of phrases, sentences, and documents. Word representation is thus critical for NLP. In this chapter, we introduce the approaches for word representation learning to show the paradigm shoft from symbolic representation to distributed representation. We also describe the valuable efforts in making word representations more informative and interpretable. Finally, we present applications of word representation learning to NLP and interdisciplinary fields, including psychology, social sciences, history, and linguistics.

## 2.1 Introduction

language has been described as 'the infinite use of infinite means'

we can investigate word representations from two aspects:

- `form` how knowledge is organized
- `source` where knowledge is from

one hot embedding, symbolism, is not optimal for computers due to high dimensionality and sparsity issues: computers need large storage for these high-dimensional representations, and computation is less meaningful because most entried of the representations are zero

in low-dimensional, real-valued dense vectors, each dimension in isolation is meaningless because semantics is distributed over all dimensions of the vector

given a word, we can find its hypernyms, synonyms, hyponyms, and antonyms from a human-organized linguistic knowledge base (like WordNet) to represent word semantics

## 2.2 Symbolic Word Representation

### 2.2.1 One-Hot Word Representation

For any two words, their one-hot vectors are orthogonal to each other. That is, the cosine similarity between cat and dog is the same as the similarity between cat and sun, which are both zeros

there is no internal semantic structure in the one-hot representation. To incorporate semantics in the representation, we will present two methods with different sources of semantics: linguistic KB and natural corpus

### 2.2.2 Linguistic KB-based Word Representation

`rationalism` regards the introspective reasoning process as the source of knowledge. Therefore, researchers construct a complex word-to-word network by reflecting on the relationship between words. Ex., human linguists manually annotate the synonyms and hypernyms of each word.

`hypernyms` are words whose meaning includes a group of other words, which are instances of the former. Word u is a hyponym of word v if and only if word v is a hypernym of word u.

it is clear that this representation has limited expressive power, where the similarity of two words without common hypernyms and hyponyms is 0. it would be better to directly adopt the original graph form, where the similarity between the two words can be derived using metrics on the graph. For synonym networks,  we can calculate the distance between two words on the network as their semantic similarity (i.e. the shortest path length between the two words). Hierarchical information can be utilized to better measure the similarity for hypernym-hyponym networks.

The information content (IC) approach is proposed to calculate the similarity based on the assumption that the lower the frequency of the closest hypernym of two words is, the closer the two words are

### 2.2.3 Corpus-based Word Representation

The process of constructing a linguistic KB is labor-intensive. In contrast, it is much easier to collect a corpus. This motivation is also supported by *empiricism*, which emphasizes knowledge from naturally produced data

## 2.3 Distributed Word Representation

"distributed representation" is completely different from and othogonal to the "distributional representation" (induced by "distributional hypothesis"). Distributed representation describes the form of a representation, while distributional hypothesis (representation) describes the source of semantics

### 2.3.1 Preliminary: Interpreting the Representation

Although each dimension is uninterpretable in distributed representation, we still want ways to interpret the meaning conveyed by the representation approximately. Two basic computational methods to understand:

- similarity
  - euclidian distance
  - euclidian similarity
  - cossine similarity
- dimension reduction
  - distributed representations still exist in manifolds higher than 3 dimension. To visualize them, we need to reduct the dimension of the vector to 2 or 3
    - PCA does this
      - principal component analysis
      - transforms vectors into a set of new coordinates using orthogonal linear transformations
      - in the new coordinate system, an axis is pointed in the direction which explains the datas most variance while being orthogonal to all other axes
      - the later constructed axes explain less variance and therefore are less important to fir the data
      - highly adopted in high-dimensinoal data vis, **PCA is unable to visualize representation that form non-linear manifolds**
        - t-SNE can solve this problem
    - SVD-based PCA
      - singular value decomposition
    - Latent Semantic Analysis (LSA)

### 2.3.2 Matrix Factorization-based Word Representation

Distributed representations can be transformed from symbolic representations by matrix factorization or NNs

- LSA - latent semantic analysis
  - uses SVD
- PLSA - probablistic LSA
- LDA - latent dirichlet allocation

PSLA and LDA are moer often used in document retrieval

LDA can be seen as a bridge between distributed representations and symbolic representations, because the sparsity and interpretability make LDA essentially a form of symbolic representation

The information source (i.e. counting matric M) of matrix factorization-based methods is still based on the bag-of-words hypothesis. These methods lose the word order information in the documents, so their expressiveness capability remains limited

### 2.3.3 Word2vec and GLoVe

- word2vec
  - adopts the distributional hypothesis, but does not take a count-based approach
  - directly uses gradient descent to optimize the representaiton of a word toward its neighbors representations
  - two sepcifications
    - continuous bag of words (CBOW)
      - predicts a center word based on multiple context words
      - window size is a hyperparemter
      - two typical approximation methods:
        - hierarchical softmax
          - build hiererchical classes for all words and to estimate the probability of a word by estimating the conditional probability of its corresponding hierarchical classes
          - then, the probability of a word can be obtained by multiplying the probabilities of all nodes on the path from the root node to the corresponding leaf
          - tree of hierarchical classes is generated according to the word frequencies, which is called the Huffman tree
        - negative sampling
          - more straightforward
          - directly samples k words as negative samples according to the word frequency
          - computes the softmax over k+1 words
    - skip-gram
      - predicts multiple context words based on the center word
- GloVe
  - word2vec and matrix factorization ahve complementary advantages and disadvantages
    - learning efficiency and scalaiblity - word2vec is superior bc uses online (or batch learning) approach and can learn over a large corpa
    - considering the preciseness of distribution modeling, the matrix factorization-based methods can exploit global co-occurrence information by building a global co-occurence matrix
    - word2vec is wondow based
  - glove is proposed to combine the advantages of word2vec and matrix factorization-based methods
  - to learn global count statistics, GloVe firstly builds a co-occurence matrix M over entire corpus but does not directly factorize it
    - instead, it takes each entry in the co-occurence matrix M and optimizes the following target
  - GloVe uses weighted squared loss to optimize the representation of words based on the elements in the gloval co-occurence matrix
    - compared to word2vec, it captures global statistics
    - compared with matrix factorization
      - reasonably reduces the weights of the most frequent words at the level of matrix entries
      - reduces the noise caused by non-discriminative word pairs by implicitly optimizing the ratio of co-occurrence frequencies
      - enables fitting on large corpus by iterative optimization
    - since the number of nonzero elements of the occurence matrix is much smaller than abs(V)^2, the efficency of GloVe is ensured in practice

Word2Vec as Implicity Matrix Factorization:

- SVD achieves a significantly better objective value when the embedding size is smaller than 500 dims and the number of negative samples is 1
- with more negative samples and higher embedding dimensions, SGNS gets a better objective value
- for downstream tasks, under several conditions, SVD achieves slightly better performance on word analogy and word similarity
- in contrast, skip-gram with negative sampling achieves better performance by 2% on syntactical analogy

### 2.3.4 Contextualized Word Representations

In natural language, the semantic meaning of an individual word can usually be different wrt its context:

- willows lined the *bank* of the stream
- a *bank* account

Traditional word word embeddings (CBOW, skip-gram, GloVe, etc) cannot well understand the different nuances

ELMo was proposed, whose word representaiotn is a function of the whole input. Rather than having a look-up table fo word embeddings, ELMo converts words into low-dimensional vectors on-the-fly by feeding the word and its context into a deep neural network. ELMo utilizes a bidirectional language model to conduct word representation, and a forward LM models the probability of the sequence yb predicting the probability of each word according to the historical context. This is a LSTM. the backward LM is similar, but the difference is that it reverses the input word sequence and predicts each word according to future context

When used in a downstream task, ELMo combines all layer representations of the bidirectional LM into a single vector as the *contextualized word representation*.

## 2.4 Advanced Topics

Essential features of a **good** word representation

- Informative Word Representation
  - A key point where representaiton learning differs from traditional prediction tasks is that when we construct representatinos, we do not know what information is needed for downstream tasks
  - Therefore, we need to compress as much information as possible into the representation to facilitate various downstream tasks
- Interpretable Word Representation
  - For distributed representations, a single dimension is not responsible for explaining the factors of semantic change, and the semantics is entangled in multiple dimensions. As a result, distributed representations are difficult to interpret
  - a good distributed representation should "disentangle the factors of variation"

### 2.4.1 Informative Word Representation

To make representations informative, we can learn word representations from universal training data. Another key direction for being informative is incorporating as much additinoal information into the representaitno as possible. From small to large information granularity, we can utilize character, morphological, syntactic, document, and knowledge abse information

- Multilingual Word Representation
- Character-enhanced Word Representation
  - ike chinese and japanese. Position-based and cluster-based methods are porposed to address the issue that characters ar highly abiguous
- Morphology-Enhanced Word Representation
  - many languages, like english, have rich morphological information and plenty of rare words
  - most word rep models ignore right morphology information
  - there is a limitation bc a words affixes can help infer a words meaning
  - when facing rare words without enough context, the reps tend to be innacurate
- Syntax-Enhanced Word Representatino
  - during training, the model optimizes the probability of dependency-based context rather than neighboring contexts
- Document-Enhanced Word Representation
  - `topical word embedding (TWE)` introduces topic information generated by LDA to help distinguish different meanings of a word
    - each word w is assigned a unique topic, and each topic has a topic embedding
  - `TopicVec` further improves the TWE model. TWE simply combines the LDA with word embeddings and lacks statistical foundations
    - TopicVec encodes words and topics in the same semantic space - can learn coherent topics from only one doc, and does not require numberous documents
- Knowledge-Enhanced Word Representation
  - introduce relational objectives into the CBOW model
    - with the objective, the embeddings can predict their contexts and words with relations
  - can also retrofit, which introduces a post-processing step that can introduce knowledge bases into word representation learning
    - attempts to find a knowledgeable embedding space which is close to W_hat but considers the relations in the KB

### 2.4.2 Interpretable Word Representation

Distributed word representation achieves ground-breaking performance on numerous tasks, is less interpretable than traditional symbolic representation

We can improve interpretability from three directions:

- Increase interpretability of vector among its neughbors
  - since a word has multiple meanings, the vectors of different meanings should locate in different neighborhoods
- Increase interpretability of each dimension in the representations
- Increase interpretability of the embedding space
  - introduce more spatial properties

More

- Disambiguated Word Representation
  - using only one vector to represent a word is problemmatic
  - antithetical to having multiple meanings
  - someone did some clustering
    - clustering is offline, and number of clusters is fixed
    - difficult for a model to select an appropriate amouunt of meaning for different words, to adapt to new sense new words or new data, and align the senses with prototypes
- Nonnegative and Sparse Word Representations
  - Nonnegative and sparse embeddings (NNSE)
    - each dimension indicates a unique concept
- Non-Euclidian Word Representation
  - two special embedding spaces
    - gaussian distribution space
      - the mean mu of the Gaussian distribution is similar to traditional word embeddings and the variance SIGMA becomes the uncertainty of the word meaning
    - hyperbolic space
      - are spaces with constant negative curvature
      - the volume of the hyperbolic space grows exponentially with radius
        - this property makes it suitable for encoding tree structure, and as such, is suitable for encoding herarchical structures
    - (both enjoy hierarchical spatial properties that are understandable by humans)

## 2.5 Applications

### 2.5.1 NLP

Word representations helpful for:

- word similarity
- word analogy
- ontology construction
- sentiment analysis

Simple word vectors gradually ceased to be used alone:

- high level (e.g. sentence level) semantic units require combinations between words, and simple arithmetic operations between word representations are not sufficient to model high-level semantic models
- most word representation models do not consider word order and cannot model utterance probabilities, much less generate language

### 2.5.2 Cognitive Psychology

### 2.5.3 History and Social Science
