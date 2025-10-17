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

GloVe is comparable and in some cases better than popular word2vec

Shortcomings: they both produce a single vector, disregard different meanings (plant = grow or machinery)

### 3.3.3 Contextual Word Embeddings

Problem with word3vec: failure to express polysemous words

- the final vector is placed somewhere in the weighted middle of all words meanings
  - thus, rare meanings are poorly expressed

Idea of contextual word embeddings: generate a different vector for each context a word appears in (context is typically defined sentence wise and solves problem of polysemy to a large extent)

NN training takes pairs of neighboring words and uses one as an input and the other as the prediction.

Language modeling:

- frequent approach is to concatenate weights from several layers into a vector
  - note: this follows step of encoding inputs via word2vec before doing classification or w/e
- often more convenient ot use the whole pretrained LM as a starting point (transfer learning)

Main representatives of contextual embeddings:

#### ELMo (Embeddings from Language Models)

Bro the first of an era this was such a sick time to be doing this stuff

- actual embeddings are bidirectional LSTM
  - higher level layers capture context-dependent aspects of the input
  - lower level layers capture aspects of the syntax
- treats input on character level
- builds words from subword tokens (groups of characters)
  - trained one sentence as a time as input

Outperformed previous pretrained word embeddings (word2vec and GloVe) on many tasks

- If **explicit vectors are required**, ELMos compact three-layer architecture may be preferable to BERTs deeper architecture

#### ULMFiT (Universal Language Model Fine-Tuning)

- Aims to transfer knowledge captured in a source task to one or more target tasks
  - language modeling is the source task
- LSTM w dropout
- 3 phases of training:
  - general domain LM training
  - target task LM fine-tuning
  - target task classifier fine-tuning
    - in each fine-tuning, gradually start reactivating different layers of the network to allow adaptation to target domain
    - basically diff learning rate for each layer
      - reccomended to geometrically decreasing

#### BERT (Bidirectional Encoder Representaitons from Transformers)

The GOAT dude bless

- masked language models, inspired by gap-filling tests
- another task: if two sentences appear in a sequence

#### UniLM (Universal Language Model)

- extends learning tasks of BERT
  - masked language modeling
  - predicting if two sentences are consecuitive
  - >> sequence to sequence prediction task
- left to right masked word prediction
- right to left masked word prediction
- seq-t0-seq predicts masked word in second (target) sequence based on words in first (source) sequence

To predict tokens in the second segment, UniLM learns to encode the first segment. In this way, the gap-filling task corresponds to sequence-to-sequence LM and is composed of two parts, encoder and decoder. The encoder part is bidirectional, and the decoder is unidirectional (left-to-right) *(is this fully correct?)*

Similar to other encoder-decoder models, this can be exploited for text generation tasks such as summarization

## 3.4 Sentence and Document Embeddings

Like word embeddings, embeddings can be constructed for longer units of texts such as sentences, paragraphs, and documents

The longer the text, the more challenging to capture the semantics, which must be encapsulated in the embedded vectors, and is much more diverse and complex than the semantics of words

- Deep averaging network
  - simplest approach
  - average word embeddings of all words
  - one apprach (deep averaging network):
    - first averages embeddings of words and bigrams
    - then, uses them as input to ffn dnn to produce sentence embeddings
- doc2vec
  - extension of word2vec
  - behave as if a doc has another floating word-like vector (named doc-vector) contributing to all training predictions
  - `paragraph vector-distributed memory (PV-DM)` is analogous to the word2vec CBOW variant
   - doc-vectors are obtained by training a nn on the `synthetic task` (bro love this term) of predicting a center word based on an average of both context word-vectors and the documents doc-vector
  - `paragraph vector-distributed bag of words (PV-DBOW)` is analogous to word2vec skip-gram variant
    - doc vecs trained on synthetic task of predicting a target word just from the documents doc-vector
- Transformer-based universal sentence encoder
  - skip-thought tasks
    - unsupervised learning from arbitrary running text (based on an input sequence)
    - task is to predic the previous and next sentence around a given sentence
  - conversational input-response task
    - assures inclusion of parsed conversation data
      - uses pairs of email messages and their responses

## 3.5 Cross-Lingual Embeddings

Has been observed that dense word embedding spaces exhibit similar structures across languages

Assuming embedding vectors in matrix, can use a transformation that approx aligns vector spaces of two languages L1 and L2:

- $W\cdot L_2 \approx L_1 $

Alignment transformations:

- supervised
  - obtained from dictionary
- semi-supervised
  - small seeding dictionary to construct initial mappings that more words can be derived from
- unsupervised
  - initial matching words are obtained by frequence of words in two languages or a learning task
    - referred to as "dictionary induction"

## 3.6 Intrinsic Evaluation of Text Embeddings

### Extrinsic Evaluation

- most common
- quality of embedding evaluated in some downstream learning task, such as classification

### Intrinsic Evaluation

relates produced embeddings to similarity of entities in the original space

- similarity in original space can be expressed with "loss function" used in the data transformatoin task
- in some cases, can reconstruct original entities from transformed ones
  - here *data reconstruction error* is the eval metric
    - think: autoencoders
- compare to human-annotated gold-standard dataset

#### Word Analogy

See Mikolov 2013 for "word analogy task for intrinsic evaluation of word embeddings"

Definition 3.1 (Word Analogy)

The word analogy task is defined as follows. For a given relationship a:b, the task is to find a term y for a given term x so that the relationship between x and y best resembles the given relationship a:b in the pair x:y

There are two main groups of analogy categories: semantic and syntactic

- Semantic relationships
  - Consider capital of country, i.e. city a is capital of country b
  - if given word pair is Helsinki:Finland
  - and given Stockholm -> need to answer with Sweden
- Syntactic relationship
  - each category refers to a grammatical feature, e.g. 'adjective a is the comparative form of adjective b'
  - the two words in any given pair have a common stem (or even the same lamma)
    - ex: longer:long

In vector space, the analogy task is transformed into vector arithmetic.

- search for nearest neighbors, i.e. compute distance between vectors
  - search for word y which would give the closest result in the expression:
    - d(vec(Stockholm), vec(y))
  - In Mikolov, the analogies are already pre specified, so one does not need to search for closest result
    - just check if prespecified word is indeed the closest
    - many listed analogies will be matched if the relations from the dataset are correctly expressed in the embedding space.
      - therefore, the eval metric, use classification accuracy of nearest neighbor classifier
        - where query point is given as vec(x) - vec(a) + vec(b)

15 categories: 5 semantic and 10 syntactic/morphological

- capital cities to countries
- family member relations
- non-capital city
- species

etc theres a bunch (all of 15 lazy boi), theyre pretty neat fr

## 3.7 Implementation and Reuse
