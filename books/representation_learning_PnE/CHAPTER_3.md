# Chapter 3: Text Embeddings

## 3.1 Background Technologies

- Transfer learning
- Language Models

(note - not sure how these are "background" lol)

### 3.1.1 Transfer Learning

Idea: solve a selected problem and transfer the gained knowledge to solve other related problems. Reuse of information from previously learned tasks

- Model reuse
  - A model developed for one task, reused as the starting point for a second task
  - good for image and text models
  - greatly reduce resource required to develop NNs on final problem
- Multi-task learning
  - train models on several related tasks together
  - related tasks share representation
  - reduces likelihood of overfitting, as related tasks serve as regularization mechanism enforcing better generalization

### 3.1.2 Language Models

- traditional LMs on n-grams
  - probabilistically, follows Bayes rule or Markovian approaches
- masked language models for NNs

## 3.2 Word Cooccurrence-Based Embeddings

- need to transform words, phrases, sentences, documents into vectors of real numbers
  - one-hot and bag-of-words encoding were very high dimensionality and sparse
  - modern approaches are lower dimensionality
    - rely on "the distributional hypothesis" of Harris 1954, which states:
      - words occuring in similar contexts tend to have similar meanings
- goal is exploit word cooccurrence to develop vectors that reflect:
  - semantic similarities
  - semantic dissimilarities
  - (distance/similarity in embedding space)

Two types:

- Sparse word embeddings
  - one hot
  - bag of words (BoW)
  - term frequency
- Dense word embeddings
  - Latent Semantic Analysis (LSA)
    - reduce dim of word cooccurrence matrix with SVD
  - Probabilistic LSA
  - Latent Dirichlet Analysis (LDA)
    - topic based

### 3.2.1 Sparse Word Cooccurrence-Based Embeddings

- `One-hot` word encoding
  - one vector for each word with one index active for that word
- `Bag-of-words`
  - sentences represented by activating all index for each word in single vector
  - for documents, the overlap of active words becomes problemmatic

### 3.2.2 Weighting Schemes

For diff tasks, the individual features (words or terms) are not equally important. Four most popular weighting schemes:

- Binary
  - feature weight is 1 if term is present in doc
- Term occurrence
  - count of occurences (often better than simple binary value)
- Term frequency
  - divide occurence by sum of vector weights
- Term Frequency-Inverse Document Frequency (TF-IDF)
  - weight depends on document as well as corpus where doc belongs
  - weight increases w/ number of occurrences in document (TF)
  - inverse of its frequency in document corpus
    - high frequency in all corpus docs indicates a commonly appearing word

Other approaches:

- Okapi BM25
  - eval how well query of several terms matches a given document
- $\Chi^2$
  - defined for docs with assigned class labels
  - attempts to correct a drawback of TF scheme (which is not addressed by TF-IDF) by considering the class value of processed documents
  - the scheme penalizes terms that appear in documents of all calsses, and favors terms specific to individual classes
- Information Gain (IG)
  - uses class labels to improve term weights
  - applies information-theoretic approach by measuring amount of info about one random variable (the class of a document) gained by knowledge of another random variable (the appearance of a given term)
- Gain Ratio (GR)
  - similar to information gain, but normalized by the total entropy of the class labels
- Delta-IDF
  - and Relevance Frequency
  - seek to merge ideas of TF-IDF and class-based schemes by penalizing both
    - common terms
    - terms that are not class-informative

### 3.2.3 Similarity Measures

- Dot product
  - sum of products of feature values of vectors
- Cosine similarity
  - cos of angle between two non-zero vectors (equals the inner product of the same vectors normalized to both have length of 1)

A well-tested combination for comparing the similarity of documents is TF-IDF with cosine sim

### 3.2.4 Sparse Matrix Representation of Texts

- BoW
  - *term-document matrix*
    - one row represents one word (term)
    - columns represent docs
    - n words in vocab
    - m documents

### 3.2.5 Dense Term-Matrix Based Word Embeddings

Matrix factorization-based dense word embeddings

Based on idea from linear algebra to compress the information encoded in a sparse matrix

- SVD
- LSA
  - SVD to a term-document matrix

The problem with matrix factorization based embeddings: frequent words have too high of impact

One approach weights entries of the term-term matrix with positive point-wise mutual information

### 3.2.6 Dense Topic-Based Embeddings

LSA can also be viewed from perspective of topic modeling, which assumes that documents are composed of some underlying topic

The eigenvectors produced by SVD can be seen as latent concepts, as they characterize the data by relating words in similar contexts

Unfortunately, the coordinates in the new space do not have an obvious interpretation

- Probabilistic Latent Semantic Analysis
  - docs viewed as mixture of various topics
  - instead of SVD, uses statistical latent class model on a word-document matrix using `tempered expectation maximization`
  - can see sets of words as latent concepts, and documents are probabilistic mixtures of these topics
- Latent Dirichlet Allocation
  - shortcoming of PLSA is it has to estimate probabilities from topic-segmented training data
  - takes similar latent-variable approach, but assumes topic dist has sparse Dirichlet prior
  - LDA supports the intuition that each doc covers only a small set of topics, and in that topic, only a small set of words appear frequently
  - however, topics that LDA returns are not semantically strongly defined (as they rely on the likelihood of term cooccurrence)

## 3.3 Neural Word Embeddings

Gradually became method of choice for text embeddings

Common procedure: train NN on one or more semantic text classification tasks, and then take the weights of the trained NN as representation for each unit of text (unit, n-gram, sentence, document)

The labels required for training such a classifier originate from huge corpa of texts. Typically they reflect word occurrence

Representation learning can be extended with other related tasks, such as prediction if two sentences are sequential (sentence entailment)

- positive instances are obtained from text in corpus
- negative instances obtained with negative sampling (sampling from instances that are unlikely to be related (...??? probabilistically or non-corpa texts?))

### 3.3.1 Word2vec Embeddings

- uses single layer FFN to predict word cooccurrence
- weights of hidden layer used as embedding

Consists of two related methods:

- Continuous Bag-of-Words (CBOW)
- skip-gram
  - slight advantage over CBOW

Words and their contexts (one word at a time) appearing in the training corpus constitute the training instances of the classification problem.

- For window size of d = 2
- "Tina is watching her linear algebra lecture with new glasses"
- for "linear", some positive instances are:
  - (linear, watching)
  - (linear, her)
  - (linear, algebra)
  - (linear, lecture)
- training:
  - input pair[0]
  - output pair[1]
  - loss: average log probability

$\frac{1}{T} \sum_{t=1}^{T} \sum_{-d \leq j \leq d, j \neq 0} \log{p}(w_t + j | w_t ) $

To make efficient, actual implementation uses several approximation tricks

- biggest problem: estimation of p(w|wt)
  - this normally requires computation of dot product between wt and all other words in vocab
  - can be solved with negative sampling
    - replaces log p(w|wt) terms with resuts of logistic regression classifiers trained to distinguish between similar and dissimilar words

### 3.3.2 GloVe Embeddings

LSA and other statistical methods take entire term-document or term-term matrix into account. This workds **poorly** on word analogy task, indicating a sub-optimal vector space structure

GloVe exploits statistics stored in term-term matrix

- `GloVe` = Global Vectors
  - presents training of word embeddings as an optimization problem
    - stochastic optimization on word cooccurrence, using a context window of 10 words before and after given word, but decreasing impact of more distant words
    - instead o fusing count of cooccurrence of word w_i with word w_j within a given context (or its entropy-based transformation)
      - GloVe operates on ratio of cooccurrence of words w_i and w_j, relatively to a context word w_t
      - P_it is probability of seeing word w_i and w_t together
        - computed by dividing number of times w_i and w_t appear together (c_it) by total number of times t_i appeared in corpus (c_i)
      - P_jt has similar notation
      - on ratio of Pit/Pjt
        - large when context words wt related to wi but not wj
        - small when wt related to wj but not wi
        - close to 1 when wt is related to both wi or wj (or unrelated to both)

Loss fn

$= \sum_{i=1}^{\|V\|} \sum_{j=1}^{\|V\|} f(C_{ij})(\hat{w}_{i}^{T}\hat{w}_{j} + b_i + b_j - \log{c_{ij}})^2 $

- Cij is cooccurrence count of word vectors $\hat{w}_{j}, \hat{w}_{j} \in R^k $
- $f(C_{ij}) $ is weighting function that assigns relatively lower weights to rare and higher weight to frequent cooccurrences
- bi and bj are bias terms

GloVe is comparable and in some cases better than populat word2vec

Shortcomings: they both produce a single vector, disregard different meanings (plant = grow or machinery)
