# Chapter 1 - Representation Learning and NLP

Natural language processing (NLP) aims to build linguistic-specific programs for amchines to understand and use human languages. Conventional NLP methods heavily rely on feature engineering to constitute semantic representations of text, requiring careful design and considerable expertise. Meanwhile, representation learning aims to automatically build informative representations of raw data for further application and achieves significant success in recent years. This chapter presents a brief introduction to representation learning, including its motivation, history, intellectual origins, and recent advances in both machine learning and NLP.

## 1.1 Motivation

3 components of ML:

Machine Learning = Representation + Objective + Optimization

- first, transform raw data into internal representations (such as feature vectors)
- representation methods determine what and how valuable information can be extracted from raw data
- the more information extracted, the better the performance of classification

Representation learning aims to learn informative representations of objects from raw data automatically. Deep learning is a typical approach, which has two distinguishing features:

- Distributed Representation:
  - DL represent each object with a low-dimensional and real-valued vector
  - representation form named `distributed representation` of `embedding`
  - compared to conventional symbolic representation, distributed rep is more compact and smooth by mapping data in the low-dimensional and continuous space, making it more robust to sparsity (sparsity which is ubiquitous and inevitable due to power-law distribution in large-scale data)
- Deep Architecture:
  - allows capture of of informative features and complicated patterns of objects from raw data

## 1.2 Why Representation Learning is Important for NLP

### 1.2.1 Multiple Granularities

NLP is concerned about multiple levels of language items, including but not limited to:

- characters
- senses
- words
- phrases
- sentences
- paragraphs
- documents

Representation learning is able to represent the semantic meanings of all language items in a unified semantic space

### 1.2.2 Multiple Knowledge

Types of knowledge:

- Linguistics (like parse trees)
  - capture subject and object of sentence
- Commonsense
  - "A play is a work of drama, consisting of dialogue between characters"
  - We *know* that most of shakespears plays consist of character dialogues
- Facts
  - Hamlet is written by Shakespeare
  - so, we can infer that Hamlet is an English play
- Expertise

Knowledge should be provided as much as possible to make machines more intelligent

Difficult for symbolic text and knowledge to work together due to diferse representation forms, which are usually remedied by additional engineering efforts such as entity linking **and suffered from error propogation**

Representation learning, in contrast, can easily incorporate multiple types of structured knowledge into NLP systems by encoding both sequential text and structured knowledge into unified embedding forms

### 1.2.3 Multiple Tasks

Can perform multiple tasks on the same sentence:

- Part of Speech (POS) tagging
  - classify words into nouns, verbs, adj, etc
- Dependency parsing
  - a language grammer to build syntactic relations between language items in a sentence
  - important for statistical NLP
- NER
- Entity Linking
  - further links named entity mentions into corresponding entities in external knowledge graphs
  - entity resolution
- Relation extraction
  - core task of information extraction to acquire structured knowledge from unstructured text and complete large-scale knowledge graphs
- Question answering
- Machine translation

Building a unified and learnable representation of an input for multiple tasks will be more efficient and robust: on the one hand, a better and unified text representation will help to promote all NLP tasks, and on the other hand, taking advantage of more learning signals from multitask learning may contribute to building better semantic representations of NL. Hence, representation learning can benefit from multitask learning and further promote the performance of multiple tasks

### 1.2.4 Multiple Domains

## 1.3 Development of Representation Learning for NLP

### 1.3.1 Symbolic Representation and Statistical Learning

- `one-hot encoding`
  - Words are minimum units in NLP
  - Easiest way to represent is `one-hot vector`
  - Contain no semantic information
- `bag of words (BOW) models`
  - Builds into document representaiton, i.e. `bag of words (BOW) models`
  - BOW models regard a doc as a bag of its words, neglecting the order of words
  - Represented as vocabulary sized vector
  - These are straightforward and work great in applications like:
    - spam filtering
    - text classification and clustering
    - information retrieval (here, use cosine dist as semantic similarity)
- `n-gram models`
  - if we want to predict next word, look back at some previous words (n - 1 words)
  - allow us to count and estimate reasonable probability of a word
- `distributional hypothesis`
  - language items sharing similar distributions of context have similar meanings
  - "a word is characterized by the company it keeps"
- `local representation` - `symbolic representation`
  - each entry in n-gram or BOW or one-hot explicitly matches one language item
- `IBM model`
  - symbolic based
- `LDA`
  - symbolic based

### 1.3.2 Distributed Representation and Deep Learning

- `distributed representation`
  - represents an object by a pattern of activation distributed over multiple entries
  - i.e. a low-dimensional and real-valued dense vector, and each computing entry can be involved in representing multiple objects
  - proved to be more efficient because it usually has low dimensions
  - also prevents from the sparsity issue that is inevitable for the symbolic representations due to the power low distribution in large-scale data
- `neural probabilistic language model (NPLM)`
  - one of the pioneering approaches
  - a LM predicts the conditional probability of the next word, given those previous words in a sentence
  - n-gram models can be regarded as simple language models based on symbolic representation
  - inspired:
    - `word2vec`
    - `GloVe`
    - `fastText`

### 1.3.3 Going Deeper and Larger with Pre-training on Big Data

- `ELMo`
- `BERT`
  - started the `pre-training-fine-tuning` pipeline
  - trained though language model objectives, naming them pre-trained language models (PLM)

## 1.4 Intellectual Origins of Distributed Representation

### 1.4.1 Representation debates in Cognitive Neuroscience

Most well known origin from the classical book of parallel distributed processing (PDP), which raises two opposite representation schemes:

- `local representation`
  - Given a network of simple computing elements and some entities to be represented, the most straightforward scheme is to use one computing element for each Entity. This is called a local representation
  - straightforward bc a simple mirror of knowledge structure, with each object and concept corresopnding to distinct neurons
  - high-level knowledge can be organized into symbolic systems, such as:
    - concept hierarchy
    - propositional networks
    - semantic networks
    - schemas
    - frames
    - scripts
- `distributes representatio`n
  - Each Entity is represented by a pattern of activity deistributed over many computing elements, and each computing element is involved in representing many different entities
  - better representation capacity
  - automatic generalization to novel entities
  - learnability to changing environments
- `grandmother cell hypothesis`
  - assumes a hypothetical neuron can encide and respond to a specific and complicated entity such as someone grandmother - like a one-hot representation

It is debatable whether individual neurons encode high-level concepts or objects, distributed representation seems to be a general solution for information processing at different levels, ranging from visual stimulus to high-level concepts

### 1.4.2 Knowledge Representation in AI

An essential branch of philosophy is the theory of knowledge, also known as epistimology. Epistomology studies the nature, origin, and organization of human knowledge and concerns the problems like where knowledge is from and how knowledge is organized

- `how knowledge is organized`
  - philosophers have developed many tools and methods, typically in the *symbolic form*, to describe human knowledge
  - formal logic, has played an essential role in CS
  - two main approaches:
    - `symbolism`
      - aims to develop formal symbolic systems to organize knowledge of machine intelligence
      - semantic web
      - large-scale knowledge graphs
    - `connectionism`
      - rooted in parallel distributed processing
      - deep learning of 2010s regarded as great success of this approach
- `where knowledge is from`
  - two basic views:
    - `rationalism`
      - regards reason as the chief source of human knowledge and regards intellectual and deductive as truth criteria
      - facts and rules, directly designed or collected by human experts
    - `empiricism`
      - generally appreciates the role of sensory experience in the generation of knowledge
      - knowledge representation can be regarded as the epistemology for machines in AI

### 1.4.3 Feature Engineering in Machine Learning

Aims to build feature vectors of instances for machine learning. Can be regarded as the representation learning of instances in the era of statistical learning. Feature engineering provides another intellectual origin of distributed representation i.e. dimensionality reduction of data by mapping from a high-dimensional space into a low-dimensional space, usually with the term "embedding"

Feature engineering can be divided into:

- `feature selection`
  - select most informative features and remove redundant and irrelevant ones from large amounts of candidates to represent instances such as words and documents
  - this approach is expected to improve representation efficiency and remedy the curse of dimensionality
- `feature extraction`
  - aims to build a novel feature space from raw data, with each dimension of the feature space either interpretable or not
  - latent topic models and dimensionality reduction can be regarded as representative approaches for feature extraction. latent topic models represent each document and word as a distribution over latent topics, which can be regarded as `interpretable space`
    - pLSA and LDA
  - dimensionality reduction methods learn to map objects into a low dimensional and `uninterpretable space`
    - PCA and SVD (singlular value decomposition)
  - `embedding`:
    - refers to either the projection process (such as the algorithm locally linear embedding) or the corresponding low-dimensional representation of objects
    - in recent in recent years, `distributed representation` and `embedding` are used mutually to refer to each other
      - the difference is that the model architecture most of the dim reduciton methods in statistical learning is usually shallow and straightforward and the algorithm is also restricted to specific data forms such as matric decomposition
        - tasks are good at organizing data, and good for reccomender systems
      - deep learning is capable of modeling complex interactions and capturing sophisticated semantic compositions ubiquitous in human languages

### 1.4.4 Linguistics

Human languages are regarded as the epitome of human intelligence, and linguistics aims to study the nature of human languages. Since human languages are regarded as one of the most complicated symbolic systems, linguistics typically follows the symbolism approach

- `structuralism`
  - influential theory
  - prespectives:
    - 1) a *symbol* (or sign) in human languages is composed of the *signified* (i.e., a concept in mind) and the *signifier* (i.e., a word token, or its sound or image)
    - 2) For a symbol, :the bond between the signified and the signifier is arbitrary
      - there is no *intrinsic* relationship between the concept of "sister" and the sound of the word "sister"
      - words in different languages may refer to the same concept
    - 3) hence, a symbol can only get its meaning from its relationship with other symbols

## 1.5 Representation Learning Approaches in NLP

### 1.5.1 Feature Engineering

Semantic representations for NLP in the early stage often come from statistics instead of learning with optimization. Feature engineering is a typical approach to representation learning in statistical learning and can be divided into feature selection and feature extraction

In a long period before the era of distributed representation, researchers devoted lots of effort to manually designing, selecting and weighting useful linguistic features and incorporating them as inputs of NLP models. The feature engineering pipeline heavily relies on human experts of specific tasks and domains, is thus time-consuming and labor intensive and cannot generalize well across objects, tasks, and domains

### 1.5.2 Supervised Representation Learning

Distributed representations can emerge from the optimization of neural networks under supervised learning. Relies heavily on gold standard labels, of which there are few

### 1.5.2 Self-supervised Representation Learning

In many cases, do not have human labeled data for supervised learning. We need to find "labels" intrinsically from large-scale unlabeled data to acquire the training objective necessary for neural networks.

- Langauge modeling
  - is a typical self-supervised objective because it does not require human annotations
- autoencoder
  - another type
  - has a reduction (encoding) phase
  - reconstruction (decoding) phase
  - training obj is the reconstruction loss, taking the original data as the gold standard
  - meaningful information will be encoded and kept in latent representations, and noisy or useless signals will be discarded

The most advanced `pre-trained language models` combines the advantages of self-supervised learning and supervised learning. In the `pre-training-fine-tuning` pipeline, pre-training can be regarded as self-superviesd learning from large-scale unlabeled corpa, and fine-tuning is supervised learning with labeleld task-specific data. Self-supervised learning can effectively learn from almost infinite large-scale text corpa

Other learning approaches:

- adversarial training
- contrastive learning
- few-shot learning
- meta-learning
- continual learning
- reinforcement learning

## 1.6 How to Apply Representaiton Learning to NLP

Four typical approaches:

- input augmentation
- architecture reformulation
- objective regularization
- parameter transfer

### 1.6.1 Input Augmentation

Basic ideas is to learn semantic representations of objects in advance. Then, object representations can be augmented as some parts of the input in downstream models

- word embeddings learned with language modeling from large-scale corpa and then used as input initialization for downstream NLP models
  - during learning process of downstream models, we can either keep word embeddings fixed or tune them in addition to new parameters. There is no answer to which strategy is better. In practice, should be determined by empirical comparison
- we can also introduce external knowledge related to input to augment inputs of downstream models
  - world knowledge
  - linguistic and commonsense knowledge
  - domain knowledge
  - (these representations can be learned based on KGs or symbolic rules)

### 1.6.2 Architecture Reformation

Can use objects (such as entity knowledge) and their distributed represntations to restructure the architecture of NNs for downstream tasks

### 1.6.3 Objective Regularization

Can also apply object representations to regularize downstream model learning. Singe multiple language items mapped into unified semantic space, can formalize various learning objs to regularize model learning.

Ex say we train a neural language model from large-scale corpa and learn entity representatinos from a world knowledge graph. We can add a new learning objective of entity linking as regularization by minimizing the loss of predicting the mentioned entity in a sentence to the corresponding entity in the KG

### 1.6.4 Parameter Transfer

The semantic composition and representation capabilities of language items such as sentences and documents lie in the weights within neural networks. We can directly transfer these pre-trained model parameters to downstream tasks in an end-to=end fashion

## 1.7 Advantages of Distributed Representation Learning

- `Unified Representation Space`
  - the unified scheme and space can facilitate knowledge transfer across
    - multiple language items
    - multiple human knowledge
    - multiple NLP tasks
    - multiple application domains
  - significantly improve the effectiveness and robustness of NLP performance
- `Learnable Representation`
  - Embeddings in distributed representation can be learned as a part of model parameters in supervised of self-supervised ways
- `End-to-End Learning`
  - Feature engineering in the symbolic representation scheme usually consists of multiple learning components, such as feature selection and term weighting
  - These components are conducted step-by-step in a pipeline, which cannot be well optimized according to the uultimate goal of the given task
  - In contrast, distributed representaiotn learning supports end-to-end learning via backprop across NN hierarchies
