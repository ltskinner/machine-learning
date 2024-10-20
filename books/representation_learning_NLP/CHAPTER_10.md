# Chapter 10 - Sememe-Based Lexical Knowledge Representation Learning

Linguistic and commonsense knowledge bases describe knowledge in formal and structural languages. Such knowledge can be easily leveraged in modelrn NLP systems. In this chapter, we introduce on typical kind of linguistic knowledge (sememe knowledge) and a sememe knowledge base named HowNet. In linguistics, **sememes are defined as the minimum indivisible units of meaning**. We first briefly introduce the basic concepts of sememe and HowNet. Next, we introduce how to model the sememe knowledge using NNs. Taking a step further, we introduce sememe-guided knowledge applications, including incorporating sememe knowledge into compositionality modelin, language modeling, and recurrent NNs. Finally, we discuss sememe knowledge acquisition for automatically constructing sememe knowledge bases and representative real-world applications of HowNet.

## 10.1 Introduction

Words are typically the smallest unit but words can be further decomposed. *Teacher* comprises the meanings of *education*, *occupation*, and *teach*. Linguists content that the maning of all the words comprise a closed set of sememes, and the semantic meanings of these sememes are orthogonal to each other. Compared with words, sememes are fairly implicit, and the full set of sememems is difficult to define, not to mention how to decide which sememes a word has

Basically, lots of effort to construct, and done through formal linguistic study. HowNet is the most representative sememe-based linguistic KB. Besides being a linguistic KB, HowNet also contains commonsense knowledge and can be used to unveil the relationships among concepts

## 10.2 Linguistic and Commonsense Knowledge Bases

### 10.2.1 WordNet and ConceptNet

The most representative KBs aiming at organizing

- linguistic knowledge and
- commonsense knowledge

#### WordNet

Large lexical databased that can also be viewed as a KB containing multi-relational data. Based on meanings WordNet groups English nouns, verbs, adjectives and adverbs into synsets (i.e. sets of coginitive synonyms), which represent unique concepts.

- Words in the same synset are linked with the `synonymy` relation, which indicates the words share similar meanings and could be replaced by each other in some context
- the `hypernympy/hyponymy` linkes a general synset and a specific synset which indicates that the specific synset is a subclass of a general one
- the `antonymy` describes the relation among adjectives with opposing meanings

#### ConceptNet

Commonsense knowledge (generic facts about social and physical environments)

ConceptNet describes the conceptual relations among words. Nodes in ConceptNet are represented as free-text descriptions, and edges stand for symmetric relations like SimilarTo or asymmentric relations like MadeOf

ConceptNet nodes can be easily linked with nodes in other KBs such as WordNet. Convenient to integrate commonsense knowledge and linguistic knowledge for researchers

### 10.2.2 HowNet

HowNet treats sememes as the smallest linguistic objects and additionally focuses on the relation between sememes and words

#### Construction of HowNet

three components

- 1. Sememe set construction
  - the sememe set is determined by analyzing, merging, and sifting Chinese characters and words (ayo)
  - all sememes can be categorized into 7 types, including
    - part
    - attribute
    - attribute value
    - thing
    - space
    - time
    - event
- 2. Sememe-sense-word structure definition
  - considering the polysemy, Hownet annotates different sets of sememes for different senses of a word
  - every dense is defined as the root of a "sememe tree"
- 3. Annotation process
  - HowNet is constructed by manual annotation of human experts

#### Uniqueness of HowNet

similarities to other KBs

- structured with relational semantic networks
- based on the form of natural language
- constructed with extensive human labeling

HowNet wosn unique characteristics that differ from WordNet and ConceptNet in:

- construction principles
- design philosophy
- foci

##### Comparison Between HowNet and WordNet

- Basic unit and design philosophy
  - sememes are smalles unit instead of word
  - WordNet is like a thesaurus
  - WordNet is differential by nature: instead of explicitly expressing the meaning of a word, WordNet **differentiates** word senses by placing them into different sysnets and further assigning them to different positions in its ontology
  - HowNet, in converse, is **constructive** i.e. exploiting sememes from a taxonomy to represent the meaning of each word sense. It is based on the hypothesis that all concepts can be reduced to relevant sememes
- Construction principle
  - WordNet is organized according to semantic relations among word meanings. Since word meanings can be represented by synsets, semantic relations can be treated as pointers among synsets. WordNet taxonomy designed NOT to capture common causality or function, but to show the relations among existing lexemes
  - HowNet is to form a networked knowledge system of the relations among concepts and the relation between attributes and concepts
  - HowNet is consrtucted using top-down induction: the ultimate sememe set is established by observing and analyzing all the possible basic sememes. After that, human experts evaluate whether every concept can be composed of the subsets of the sememe set
- Application scope
  - WordNet was intitially designed as a thesaurus, and evolved into a self-contained machine-readable dictionary of semantics
  - HowNet is established towards building a computer-oriented semantic network
  - WordNet handles lots of languages, HowNet is just english and chinese

##### Comparison Between HowNet and ConceptNet

- Coverage of commonsense knowledge
  - commonsense knowledge in ConceptNet is relatively explicit
  - HowNet is purely lexical, so commonsense knowledge is more implicit
  - HowNet actually covers more diverse facets of the world facts than ConceptNet
- Focu and construction principle
  - ConceptNet focuses on everyday episodic concepts and the semantic relatinos among compond concepts, which are organized hierarchically
  - Hownet focuses on rock-bottom linguistic and conceptual knowledge of the human lnaguage, and its annotation requires basic understanding of sememe hierarhcy
  - HowNet is constructed solely by human handcrafting of linguistic experts, whereas ConceptNet construction involves the general public without much background knowledge. as such, HowNet has higher quality annotations than ConceptNet
- Application scope
  - ConceptNet annotaitons may be ambiguous
  - HowNet can be more easily incorporated into modern neural networks since HowNet overcomes the problem of word ambiguity

#### OpenHowNet

free open source sememe KB, which comprises the core data of HowNet

### 10.2.3 HowNet and DeepLearning

plenty of pubs built on HowNet

#### Advantages of HowNet

characteristics:

- 1. interna of NL understanding, sememe knowledge is closer to the characteristics of natural language. The sememe annotation breaks the lexical barrier and offers an in-depth understanding of the rich semantic information behind the vocabulary
- 2. Sememe knowledge turns out to be a natural fit for deep learning techniques. By accurately depicting semantic information through a unified sememem labeling system, the meaning of each sememe is clear and fixed and thus can be naturally incorporated into the deep learning model as informative labels/tags of words
- 3. Sememe knowledge can mitigate poor model performance in low-resource scenarios

#### How to Incorporate Sememe Knowledge

- 1. Knowledge augmentation
  - adding sememe knowledge into input or designing special naural modules that can be inserted into the original network
  - in this way, can incororate explicitly without changing the neural architectures
  - we can learn sememe embeddings and directly leverage them to enrich the semantic information of word embeddings
- 2. Knowledge reformulation
  - change the original word-based model structures into sememe-based ones
  - such introduction could properly guide neural models to produce inner hidden representatinos with rich semantics in a more linguistically informative way
- 3. Knowledge regularization
  - Design a new training objective fn based on sememe knowledge or use knowledge as extra predictive targets.
  - for instance, we can first extract linguistic information (e.g. the overlap of annotated sememes of different words) from HowNet and then treat it as auxiliary regularization supervision. This approach does not require modifying the specific model architecture but only introduces an additional training objective to regularize the original optimization trajectory

## 10.3 Sememe Knowledge Representation

We can represent them using techniques similar to word representation learning (WRL)

### 10.3.1 Sememe-Encoded Word Representation

SE-WRL assumes each word sense is composed of sememes and conducts word sense disambiguation according to the contexts. In this way, we could learn representations of sememe, senses, and words simultaneously. SE-WRL proposes an attention-based method to choose an approximate word sense according to contexts automatically

For a word w, we denote $S^{(w)}  $ as its sense set. $S^{(w)} = \{s_{1}^{(w)}, ..., s_{|S^{(w)}|}^{(w)}   \}  $ may contain multiple senses. For each sense $s_{i}^{(w)}$, we denote $X_{i}^{(w)} = x_{1}^{(s_{i})}, ..., x_{|X_{i}^{(w)} |}^{(s_{i})}  $ as the sememe set for this sense, with $x_{i}^{(s_{i})}$ being the embedding for the corresponding sememe $x_{1}^{(s_{i})}$

#### Skip-Gram Model

We minimize the following loss:

$\mathcal{L} = - \sum_{c=l+1}^{n - l} \sum_{-l \leq k \leq l, k \neq 0} \log P (w_{c+k}|w_{c}) $

where l is the size of the lsiding window and $P (w_{c+k}|w_{c})$ stands for the predictive probability of the context word w_{c+k} conditioned on the centered word w_{c}. Denoting V as the vocabulary, the probability is formalized as follows:

$P(w_{c+k}|w_{c}) = (\exp (w_{c+k} . w_{c}))/(\sum_{w_{s} \in V} \exp(w_{s}.w_{c}) )   $

#### Simple Sememe Aggregation Model

Build on skip-gram model. Consideres the sememes in all sense of a word w and learns the word embedding w by averaging the embeddings of all its sememes

$w = (1/m)  \sum_{s_{i}^{(w)} \in S^{(w)}}  \sum_{s_{j}^{(s_{i})} \in X_{i}^{(w)}}  x_{j}^{(s_{i})}  $

where m stands for the total number of the sememes of word w. SSA assumes that the word meaning is composed of smaller semantic units

#### Sememe attention over Context Model (SAC)

SSA modifies the word embedding to incorporate sememe knowledge. Nevertheless, each word in SSA is still bound to an individual representation, whicha cnanot deal with polysemy in different contexts. Intuitively, we should have distinct embeddinfs for a word given contexts. To implement this, we leverage the word sense annotation in Hownet and propose the sememe attention over context model (SAC). SAC leverages the attention mechanism to select a proper word sense for a word based on its context. More specifically, SAC conducts word sense disambiguation based on contexts to represent the word

More specifically, SAC utilizes the original embedding of the word w and uses sememe ebeddings to represent context word w_{c}. The word embedding is then employed to choose the proper senses to represent the context word. The context word embedding w_{c} can be formalized as follows

$w_{c} = \sum_{j=1}^{|S^{(w_{c})}  |} ATT (s_{j}^{(w_{c})}) s_{j}^{(w_{c})}  $

where $s_{j}^{(w_{c})}$ is the j-th sense embedidng of w_c and $ATT (s_{j}^{(w_{c})})$ denotes the attention score of the j-th sense word w. The attention score is calculated as:

$ATT (s_{j}^{(w_{c})}) = (\exp (w . \hat{s_{j}^{(w_{c})}})  )/(\sum_{k-1}^{|S^{w_{c}}  |}  \exp( w . \hat{s_{k}}^{(w_{c})} )   )    $

Note $\hat{s_{k}}^{(w_{c})}$ is different from $s_{k}^{(w_{c})}$ and is obtained with the average of sememe embeddings (in this way, we could incorporate the sememe knowledge)

$\hat{s_{k}}^{(w_{c})} = (1/|X_{j}^{(w_{c})}  |)  \sum_{k=1}^{|X_{j}^{(w_{c})}  |} x_{k}^{s_{j}} $

the attention technique is based on the assumption that if a context word sense embedding is more relevant to w, then this sense should contribute more to the context word embeddings. Based on the attention mechanism, we represent the context word as a weighted summation of sense embeddings

#### Sememe Attention over Target Model (SAT)

SAc selects proper senses and sememes for context words. Intuitively, we could use similar methods ot choose the proper senses for the target word by considering the context words as attention - this is SAT

Conversely, SAT learns sememe embeddings for target words and original word embeddings for context words. SAT applies context words to compute attention over the senese of w and learn w's embedding. Formally, we have:

$w = \sum_{j=1}^{|S^{(w)}  |}  ATT(s_{j}^{(w)}  )s_{j}^{(w)} $

and we can calculate the context-based attention as follows:

$ATT(s_{j}^{(w)}  ) = (\exp(w_{c}' . \hat{s}_{j}^{(w)})  )/(\sum_{k=1}^{|S^{(w)}  |} \exp()w_{c}' . \hat{s}_{k}^{(w)}   )     $

where the average of sememe embeddings $\hat{s}_{j}^{(w)}$ is also used to learn the embeddings for each sense $s_{j}^{(w)}$. Here, w_{c}' denotes the context embedding, consisting of the embeddings of the contextual words of w_{i}

$w_{c}' = (1/2K) \sum_{k=i-K}^{k=1+K} w_{k}, k \neq i   $

where K denotes the wondow size. SAC merely leverages one target word as attention to choose the context words sense, wehereas SAT resorts to multiple context words as attention to choose the proper senses of target words. Therefore, SAT is better at WSD and results in more accurate and reliable word representations.

In general, all the above methods could successfully incorporate sememe knowledge into word representations and achieve better performance

### 10.3.2 Sememe-Regularized Word Representation

We explore how to incorporate sememe knowledge to improve word representations. We propose two variants:

- relation-based
- embedding-based

#### Sememe Relation-Based Word Representation

Relation-based word representation is a simple and intuitive method, which aims to make words with similar sememe annotations have similar embeddings. First, a synonym list is constructed from HowNet, with words sharing a certain number (e.g. 3) of sememes regarded as synonyms. Next, the word embedddings of synonyms are optimized to be closer. formally, let w_{i} be the original word embedding of w_{i} and \hat{w_{i}} be its adjusted word embedding. Denote Syn(w_{i}) as the synonym set of word w_{i}; the loss function is:

$\mathcal{L}_{sememe} = \sum_{w_{i} \in V} (\alpha_{i} \|w_{i} - \hat{w_{i}}  \|^{2} = \sum_{w_{j} \in Syn(w_{i})} \beta_{ij} \|\hat{w_{i}}  - \hat{w}_{j}  \|^{2}     )   $

where \alpha_{i} and \beta_{ij} balance the contribution of the two loss terms and V denotes the vocabulary

#### Sememe Embedding-Based Word Representation

Despite simplicity of the relation-based method, it cannot take good advantage of the information of HowNet because it disregards the complicated relations among sememes and words, as well as relations among various sememes. Regarding this limitatino, we propose the sememe embedding-based method

Specifically, sememes are represented using distributed embeddings and placed into the same semantic space as word. This method utilizes sememe embeddings as additional regularizers to learn better word embeddings. Both word embeddings and sememe embeddings are jointly learned

Formally, a word-sememe matrix M is built from HowNet, where M_{ij} = 1 indicates that the word w_{i} is annotated with the sememe x_{j}; otherwise M_{ij} = 0. The loss function can be defined by factorizing M as follows:

$\mathcal{L} = \sum_{w_{i} \in V, w_{j} \in X} (w_{i} . x_{j} + b_{i} + b_{j}' - M_{ij}  )^{2}     $

where b_{i} and b_{j}' are the bias terms of w_{i} and x_{j} and X denotes the full sememe set. w_{i} and x_{j} denote the embeddings of the word w_{i} and the sememe x_{j}

In this method, word embeddings and sememe embeddings are learned in a unified semantic space. The information about the relations among words and sememes is implicitly injected into word embeddings. In this way, the word embeddings are expected to be more suitable for sememe prediction.

## 10.4 Sememe-Guided Natural Language Processing

### 10.4.1 Sememe-Guided Semantic Compositionality Modeling

Semantic compositionality (SC) means the semantic meaning of syntactically complicated unit is influenced by the meanings of the combination rule and the units constituents. SC has shown importance in many NLP tasks including lanugage modeling, sentiment analysis, syntactic parsing, etc

To explroe the SC task, we need to represente multiword expressions (MWEs) (embeddings of phrases and componts). Formulate the SC task with a general framework as follows:

$p = f(w_{1}, w_{2}, \mathcal{R}, \mathcal{K})  $

where p denotes the MWE embedding, w_{1} and w_{2} represent the embeddings fo two constituents that belong to the MWE, R is the combination rule, K means the extra knowledge needed for learning the MWE semantics, and f denotes the compositionality function

Most of the exisint methods focus on reforming compositionality function f, ignoring both R and K. Some researchers try to integrate combination rule R to build better SC models. However ,few works consider additional knowledge K, extept that Zhu et al. incorporate task-specific knowledge into an RNN to solve sentence-level SC

We argue that the sememe knowledge conduces to modeling SC and propose a novel sememe-based method to model semantic compositionality. To begin with, we conduct an SC degree (SCD) measurement experiment and observe that the SCD obtained by the sememe formulae is correlated with manually annotated SDCs. Then we present two SC models on sememe knowledge for representing MWEs, which are dubbed semantic compositionality with aggregated sememe (SCAS) and sementic compositionality with mutual sememe attention (SCMSA). We demonstrate that both models achieve superior performance in the MWE similarity computation task and sememe prediction task. In the following, we first introduce semem-based SC degree (SCD) computation formulae and then discuss the sememe-incorporated SC models

#### Sememe-Based SDC Computation Formulae

SC is common phenomenon of MWEs, there exist some MWEs that are not fully semantically compositional. As a matter of fact, distinct MWEs have distinct SCDs. We propose to leverage sememes for SDC measurement. WE assume that a words sememes precisely reflect the meaning of a word. Based on this assumption, 4 SCD computation formluae are proposed. A smaller number means lowe SCD, X_{p} represents the sememe sets of an MWE. X_{w_{1}} and X_{w_{2}} denote the sememe set of MWEs first and second constituent. the four:

- 1. For SCD 0
  - an MWE is entirely non-compositional, with the corresponding SCD being the lowest. The sememes of the MWE are different from those of its constituents. This implies that the constituents of the MWE cannot compose the MWEs meaning
- 2. For SCD 1
  - the sememes of an MWE and its constituents have some overlap. However, the mWE owns unique sememes that are not shared by its constituents
- 3. For SCD 2
  - an MWEs sememe set is a subset of the sememe sets of constituents
  - this implies the constituents meanings cannot accurately infer the meaning of the MWE
- 4. For SCD 3
  - an MWE is entirely semantically compositional and has the highest SCD.
  - the MWEs sememe set is identical to the sememe sets of two constituents.
  - this implies that MWE has the same meaning as the combination of tis contituent meanings

#### SCD Computation Formulae Evaluation

Spearmans correlation coefficient in testing was .74 which is a high correlation and demonstrates the powerful capability of sememes in computing MWEs SCDs

#### Sememe-Incorporated SD Models

- 1. semantic compositionality with aggregated sememe (SCAS)
- 2. semantic compositionality with mutual sememe attention (SCMSA)

we first consider the case when sememe knowledge is incorporated in MWE modeling without combination rules. For an MWE p = {w_{1}, w_{2}}, we represent its embedding as:

$p = f(w_{1}, w_{2}, \mathcal{K})  $

where $ p \in \mathbb{R}^{d}, w_{1} \in \mathbb{R}^{d} $ and $w_{2} \in \mathbb{R}^{d}  $ denote the embeddings of MWE p, word w_{1}, and word w_{2}, d is the embedding dimension, and K denotes the sememe knowledge. Since an MWE is generally not present in the KB, hence we merely have to access the sememes of w_{1} and w_{2}. Denote X as the set of all the sememes $X^{(w)} = \{x_{1}, ..., x_{|X^{(w)}  |}  \} \subset X  $ as the sememe set of w, and $x \in \mathbb{R}^{d}  $ as the sememe x's embedding

1. SCAS conatenates a constituents embedding and its sememes embeddings:

$w_{1}' = \sum_{x_{i} \in X^{(w_{1})}} x_{i}, w_{2}' = \sum_{x_{j} \in X^{(w_{2})}} x_{j}   $

where $w_{1}'$ and $w_{2}'$ dneote the aggregated sememe embedidngs of w_{1} and w_{2}. we calcaulate p as:

$p = \tanh(W_{c} concat(w_{1} + w_{2}; w_{1}' + w_{2}'  ) + b_{c}  )   $

where $b_{c} \in \mathbb{R}^{d} $  denotes a bias term and $W_{c} \in \mathbb{R}^{d \times 2d}  $ denotes a composition matrix

2. SCAS simply adds up all the sememe embeddings of a constitutent. Intuitively, a constituents ememes may own distinct weights when they are composed of other constituents. To this end, SCMA is introduced, which utilizes the attention mechanism to assign weights to sememes (here we take an example to show how to use w_{1} to calculate the attention score for w_{2})

$e_{1} = \tanh (W_{a}w_{1} + b_{a}   )  $

$\alpha_{2,i} = (\exp(x_{i} . e_{1})  )/(\sum_{x_{j} \in X^{(w_{2})}}  \exp (x_{j} . e_{1}) )  $

$w_{2}' = \sum_{x_{j} \in X^{(w_{2})}} \alpha_{2,j} x_{j} $

where $ W_{a} \in \mathbb{R}^{d \times d} $ and $b_{a} \in \mathbb{R}^{d}  $ are tunable parameters. w_{1}' can be calucalted in a similar way. p is obtained the same as above eqn

#### Integrating Combination Rules

We can futher incorporate combination rules to the sememe-incorporated SC models as follows:

$p = f(w_{1}, w_{2}, \mathcal{K}, \mathcal{R}  )  $

MWEs with different combination rules are assigned with totally different composition matrices $W_{r}^{c} \in \mathbb{R}^{d \times 2d}  $, where r \in R_{s} and R_{s} refers to a combination syntax rule set. The combination rules include adjective-noun (Adj-N), noun-noun (NN), verb-noun (V-N), etc. Considering that there exist various combination rules, and some composition matrices are sparse, therefore, the composition matrices may not be well trained. Regardless this issue, we represent a composition matrix W_{c} as the summation of a low-rank matric containing combination rule information and a matrix containing compositionality information:

$W_{c} = U_{1}^{r} U_{2}^{r} + W_{c}^{c}  $

where $U_{1}^{r} \in \mathbb{R}^{d \times d_{r}}, U_{2}^{2} \in \mathbb{R}^{d_{r} \times 2d}, d_{r} \in \mathbb{N}_{+}   $ and $W_{c}^{c} \in \mathbb{R}^{d \times 2d}   $

in experiments, the sememe-incorporated models achieve better performance on the MWE similarity compuation task and sememe prediciton task.

### 10.4.2 Sememe-Guided Language Modeling

Language modeling (LM) targets at measuring the joint probability of a sequence of words.

Traditional language models follow the assumption that words are atomic symbols and thus represent a sequence at the word level. Nevertheless, this does not necessarily hold true.

example: *The US trade deficit last year is initially estimated to be 40 billion ______*

At first glance, people may thing of a *unit* to fill; after deep consideration, they may realize that the blank should be filled with a currency unit. Based on the country, the US, we can know it is an American currency unit. Then we can predict the word *dollars*. The *American*, *currency*, and *unit* which are basic semantic units of the word dollars, are also semememe of the word *dollars*. However, the above process is not explicitly modeleld by traditional word-level language models. Hence, explicitly introducing sememems could conduce to language modeling

It is non-trivial to incorporate discrete sememe knowledge into neural language models, because it does not fit with the continuous representations of neural networks. To address the above issue, we propose a sememe-driven language model (SDLM) to utilize sememe knowledge. When predicting the next word

- 1. SDLM estimates sememems distribution based on the context
- 2. trat those sememes as experts, SDLM employs a sparse expert product to choose the possible senses
- 3. SDLM calculates the word distribution by marginalizing the distribution of senses

SDLM consists of three components:

- sememe predictor
  - considers the contextual information and assigns a weight for every sememem
- sense predictor
  - regard each sememem as an expert and predict the probability over a set of senses
- word predictor
  - calculates the probability of every word

#### Sememe Predictor

A context vector $g \in \mathbb{R}^{d_{1}}  $ is considered in the sememe predictor, and the predictor compoutes a weight for each sememe. Given the context {w_{1}, w_{2}, ..., w_{t-1}}, the probability P(x_{k}|g) whether the next word w_{t} has the sememe x_{k} is calculated by:

$P(x_{k}|g) = Sigmoid(g . v_{k} + b_{k})   $

where $v_{k} \in \mathbb{R}^{d_{1}}, b_{k} \in \mathbb{R} $ are tunable parameters

#### Sense Predictor

Motivated by product of experts (PoE), each sememe is regarded as an expert who only predicts the senses connected with it. Given the sense embedding $s \in \mathbb{R}^{d_{2}}  $ and the context vector $g \in \mathbb{R}^{d_{1}} $, the sense predictor calculates $\phi^{(k)}(g,s)  $, which measn the score of sense s provided by sememe expert x_{k}. A bilinear layer parameterized using a matrix $U_{k} \in \mathbb{R}^{d_{1} \times d_{2}}  $ is chosen to compute $\phi^{(k)}(.,.)  $:

$\phi^{(k)}(g,s) = g^{\top}U_{k}s  $

The probability $P^{(s_{k})}(s|g)  $ of sense s given by expert x_{k} can be formulated as:

$P^{x_{k}}(s|g) = (\exp(q_{k} C_{k,s} \phi^{(k)}(g,s) )  )/(\sum_{s' \in S^{(x_{k})}} \exp (q_{k} C_{k,s'} \phi^{(k)}(g,s'))  )    $

where $C_{k,s}$ is a constant and $S^{(x_{k})}$ denotes the set of senses that contain sememe x_{k}. q_{k} controls the magnitude of the term $C_{k,s'} \phi^{(k)}(g,s')$. Hence, it decides the faltness of the sense distribution output by x_{k}. Lastly, the predictions can be summarized on sense s by leveraging the probability products computed based on related experts. In other words, the sense s's probability is defined as:

$P(s|g) ~ \prod_{x_{k} \in X^{(s)}}  P^{(x_{k})}(s|g)  $

where ~ indicates that P(s|g) is proportional to $\prod_{x_{k} \in X^{(s)}}  P^{(x_{k})}(s|g)$. X^{(s)} denotes the set of sememes of the sense s

#### Word Predictor

In the word predictor, the probability P(w|g) is calculated through adding up probabilities of s:

$P(w|g) = \sum_{s \in S^{(w)}} P(s|g)  $

where $S^{(w)}$ denotes the senses belonging to the word w. SDLM achieves remarkable performance. In-depth case studies further reveal that SDLM coudl improve both the robustness and interpretability of LMs

### 10.4.3 Sememe-Guided Recurrent Neural Networks

aim of enhancing the ability of sequence modeling

previous works have tried to incorporate other linguistic KBs into rNNs. The utilized KBs are generally word-level KBs (e.g. WordNet and ConceptNet). Differently, HowNet utilizes sememes to compositionally explain the meaning of words. Consequently, directly adopting existing algorithms to incorporate sememes into RNNs is hard. Three algorithms proposed

#### Preliminaroes for RNN Architecture

An LSTM comprises a series of cells, each corresponding to a token. At each ste t, the word embedding w_{t} is input to the LSTM to produce the cell state c_{t} and the hidden state h_{t}. Based on the previous cell state c_{t-1} and hidden state h_{t-1}, c_{t} and h_{t} are calculated as follows:

$f_{t} = Sigmoid(W_{f} concat(w_{t}: h_{t-1}) + b_{f})  $

$i_{t} = Sigmoid(W_{l}concat(w_{t}; h_{t-1}) + b_{l}  )  $

$\tilde{c}_{t} = \tanh(W_{c} concat (w_{t}; h_{t-1}) + b_{c})  $

$c_{t} = f_{t} \cdot c_{t-1} + i_{t} \cdot \tilde{c}_{t} $

$o_{t} = Sigmoid(W_{o} concat(w_{t}; h_{t-1}) + b_{o}) $

$h_{t} = o_{t} \cdot \tanh(c_{t})  $

where f_{t}, i_{t} and o_{t} denote the output embeddings of the forget gate, input gate, and output gate. W_{f}, W_{l}, W_{c} and W_{o} are weight matrices and b_{f}, b_{l}, b_{c}, and b_{o} are bias terms

GRU has fewer gates than LSTM and can be viewed as a simplification for LSTM. Given the hidden state h_{t-1} and the input w_{t}, GRU has a reset gate r_{t} and an update gate z_{t} and computes the output h_{t} as:

$z_{t} = Sigmoid(W_{z} concat(w_{t}; h_{t-1}) + b_{z})  $

$r_{t} = Sigmoid(W_{r} concat (w_{t}; h_{t-1}) + b_{r})  $

$\tilde{h}_{t} = \tanh (W_{h} concat(w_{t}; r_{t} \cdot h_{t-1}) + b_{h})  $

$h_{t} = (1 - z_{t})\cdot h_{t-1} + z_{t} \cdot \tilde{h}_{t}  $

where W_{z}, W_{r}, W_{h}, b_{z}, b_{r}, and b_{h} are tunable parameters

#### Simple Concatenation

The first method focuses on the input and directly concatenates the summation of the sememe embeddings and the word embeddings. Specifically, we have:

$\pi_{t} = (1 / (|X^{(w_{t})} | )) \sum_{x \in X^{(w_{t})}}  x $

$\tilde{w}_{t} = concat(w_{t}; \pi_{t})  $

where x is the sememe ebmedding of x and \tilde{w}_{t} denotes the modified word embedding that contains sememe knowledge

#### Sememe Output Gate

Simple concatentation incorporates sememe knowledge in a shallow way and enhances only the word embeddings. To leverage sememe knowledge in a deeper way, we present a second method by adding a sememe output gate o_{t}^{s}. This architecture explicitly models the knowledge flow of sememes. Note that the sememe output gate is designed specifically for LSTM and GRU. This output gate controls the flow of sememe knoweldge in the whole model. Formally, we have

$f_{t} = Sigmoid(W_{f} concat(w_{t}: h_{t-1}; \pi_{t}) + b_{f})  $

$i_{t} = Sigmoid(W_{l}concat(w_{t}; h_{t-1}; \pi_{t}) + b_{l}  )  $

$\tilde{c}_{t} = \tanh(W_{c} concat (w_{t}; h_{t-1}) + b_{c})  $

$c_{t} = f_{t} \cdot c_{t-1} + i_{t} \cdot \tilde{c}_{t} $

$o_{t} = Sigmoid(W_{o} concat(w_{t}; h_{t-1}; \pi_{t}) + b_{o}) $

$o_{t}^{s} = Sigmoid(W_{o^{s}} concat(w_{t}; h_{t-1}; \pi_{t}) + b_{o^{s}}) $

$h_{t} = o_{t} \cdot \tanh(c_{t}) | o_{t}^{s} \cdot \tanh(W_{c} \pi_{t}) $

where W_{o^{s}} and b_{o^{s}} are tunable parameters.

similarly, we can rewerite the formulation of a GRU cell as:

$z_{t} = Sigmoid(W_{z} concat(w_{t}; h_{t-1}; \pi_{t}) + b_{z})  $

$r_{t} = Sigmoid(W_{r} concat (w_{t}; h_{t-1}; \pi_{t}) + b_{r})  $

$o_{t}^{s} = Sigmoid(W_{o} concat(w_{t}; h_{t-1}; \pi_{t}) + b_{o}) $

$\tilde{h}_{t} = \tanh (W_{h} concat(w_{t}; r_{t} \cdot h_{t-1}) + b_{h})  $

$h_{t} = (1 - z_{t})\cdot h_{t-1} + z_{t} \cdot \tilde{h}_{t} + o_{t}^{s} \tanh (\pi_{t})  $

where bias b_{o} is the bias vector, o_{t}^{s} denotes the sememe output gate, and W_{o} is a weight matrix

#### Sememe-RNN Cell

When adding the sememe output gate, despite the fact that sememe knowledge is deeply integrated into the model, the knowledge is still not fully utilized. Taking an above equation as example, h_{t} consists of two components:

- the information in o_{t} \cdot tanh(c_{t}) has been processed by the forget gate
- while the information in o_{t}^{s} \cdot tanh(W_{c} \pi_{t}) is not processed

Thus, these two components are incompatible

To this end, we introduce an additional RNN cell to encode the sememe knowledge. The sememe embedidng is fed into a sememe-LSTM cell. Another forget gate processes the sememe-LSTMs cells cell state. After that, the updated state is added to the original state. Moreover, the hidden state of the sememe-LSTM cell is incorporated in both the input gate and the output gate:

$c_{t}^{s}, h_{t}^{s} = LSTM(\pi_{t})  $

$f_{t} = Sigmoid(W_{f} concat(w_{t}: h_{t-1}) + b_{f})  $

$f_{t}^{s} = Sigmoid(W_{f}^{s} concat(w_{t}: h_{t}^{s}) + b_{f}^{s})  $

$i_{t} = Sigmoid(W_{l}concat(w_{t}; h_{t-1}; h_{t}^{s}) + b_{l}  )  $

$\tilde{c}_{t} = \tanh(W_{c} concat (w_{t}; h_{t-1}; h_{t}^{s}) + b_{c})  $

$o_{t} = Sigmoid(W_{o} concat(w_{t}; h_{t-1}; h_{t}^{s}) + b_{o}) $

$c_{t} = f_{t} \cdot c_{t-1} + f_{t}^{s} \cdot c_{t}^{s} + i_{t} \cdot \tilde{c}_{t} $

$h_{t} = o_{t} \cdot \tanh(c_{t})  $

where f_{t}^{s} denotes the sememe forget gate and c_{t}^{s} and h_{t}^{s} denote the sememe cell state and sememe hidden state

For GRU, the transition equation can be modified as:

$h_{t}^{s} = GRU(\pi_{t})  $

$z_{t} = Sigmoid(W_{z} concat(w_{t}; h_{t-1}; h_{t}^{s}) + b_{z})  $

$r_{t} = Sigmoid(W_{r} concat (w_{t}; h_{t-1}; h_{t}^{s}) + b_{r})  $

$\tilde{h}_{t} = \tanh (W_{h} concat(w_{t}; r_{t} \cdot (h_{t-1} + ; h_{t}^{s})) + b_{h})  $

$h_{t} = (1 - z_{t})\cdot h_{t-1} + z_{t} \cdot \tilde{h}_{t}  $

where h_{t}^{s} denotes the sememe hidden state

sememe-incorporated RNN surpsees the vanilla model

### 10.5 Automatic Sememe Knowledge Acquisition

Manual inspection and updates for sememe annotation are becoming more and more overwhelming. It is also difficult to ensure annotation consistency among experts

Sememe prediciton task is defined to predict the sememes for word senses unannotated in a semene KB

### 10.5.1 Embedding-Based Sememe Prediction

Intuitively, words with similar meanings have overlapping sememes. Therefore, we strive to represent the semantics of sememes and words and model their semantic relations.

Two methods are proposed:

- sememe prediction with word embeddings (SPWE)
  - for a target word, we look for its relevant words in HowNet based on their embeddings
  - after that, we assign these relevant words' sememes to the target word
  - the algorithm is similar to collaborative filtering in recommendation systems
- sememe prediction with (aggregated) sememe embeddings (SPSE/SPASE)
  - we learn sememe embeddings by factorizing the word-sememe matrix extracted from HowNet
  - Hence, the relation between words and sememes can be measured directly using the dot product of their embeddings, and we can assign relevant sememes to an unlabeled word

#### Sememe Prediction with Word Embeddings

Inspired by collaborative filtering in the personalized recommendation, words could be seen as users and sememes can be viewed as products to be recommended. Given an unlabeled word, SPWE recommends sememes according to the words most related words, assuming that similar words should have similar sememes. Formally, the probability $P(x_{j}|w) $ of sememe x_{j} given a word w is defined as:

$P(x_{j}|w) = \sum_{w_{i} \in V} \cos (w, w_{i})M_{ij}c^{r_{i}}  $

M contains the information of sememe annotatino, where M_{ij} = 1 means that the word w_{i} is annotated with the sememe x_{j}. V denotes the vocabulary, and cos(., .) means the cosine similarity. A high probability P(x_{j}|w) means the word w should probably be recommended with sememe x_{j}. A declined confidence factor c^{r_{i}} is set up for w_{i}, and r_{i} deontes the descending rank of cos(w, w_{i}) and c \in (0, 1) denotes a hyperparameter

Simple as it may sound, SPWE only leverages word embeddings for computing the similarities of words. SWPE is demonstrated to have superior performance in sememe prediction. This is because different from the noisy user-item matrix in recommendation systems, HowNet is manually designed by experts, and the word-sememe information can be reliably applied to recommend sememes.

#### Sememe Prediction with Sememe Embeddings

Directly viewing sememes as discrete labels in SWPE could overlook the latent relations among sememes. To consider such latent relations, a sememe prediciton with sememe embeddings (SPSE) model is proposed, which learns both word embeddings and sememe embeddings in a unified semantic space

Inspired by GloVe, optimize sememe embeddings by factorizing the sememe-sememe matrix and the word-sememe matrix. Both matrices can be derivec from the annotation in HowNet. SPSE uses word embeddings pre-trained from an unlabeled corpus and freezes them during matrix factorization. After that, both sememe embeddings and word embeddings are encoded in the same semantic space. Then we could use the dot product between them to predict the sememes

Similar to M, a sememe-sememe matrix C is extracted, where C_{jk} is defined as the point-wise mutual information between sememes x_{k} and x_{k}. By factorizing C, we finally get two different embeddings (x and \bar{X}) for each sememe x. Then we optimize the loss fn to get sememe embeddings:

$\mathcal{L} \sum_{w_{i} \in V, x_{j} \in X}  (w_{i} . (x_{j} | \bar{x}_{j}  ) + b_{i} + b_{j}' - M_{ij}  )^{2} + \lambda \sum_{x_{j}, x_{k} \in X} (x_{j}.\bar{x}_{k} - C_{jk}   )^{2}    $

where b_{i} and b_{j}' are the bias terms. V and X denote the word vocabulary and the full sememe set. The above loss fn consists of two parts, i.e. factorizing M and C. Two parts are balanced by a hyper-parameter \lambda

Condsidering that every word is generally labeled with 2-5 sememes in HowNet, the word-sememe matrix is very sparse, with most of the elements being 0. It is found empirically that, if both "zero elements" and "non-zero elements" are treated in the same way, the performance would degrade. Therefore, we choose distinct factorization strategies for zero and non-zero elements. For the former, the model factorizes them with a small probability (e.g. 0.5%), while for "non-zero elements" the model always chooses to factorize them. Armed with this strategy, the model can pay more attention to those "non-zero elements" (i.e. annotated word-sememe pairs)

#### Sememe Prediction with Aggregated Sememe Embeddings

A simple way to model semantic composability is to represent word embeddings as a weighted summation of all its sememes' embeddings.

Based on this intuition, we propose sememe prediction with aggregated sememe embeddings (SPASE). SPASE is also built upon matrix factorization:

$w_{i} = \sum_{x_{j} \in X^(w_{i})} M_{ij}' x_{j}  $

where $X^(w_{i})$ denotes the sememe set of the word w_{i} and M_{ij}' represents the weight of sememe x_{j} for word w_{i}. To learn sememe embeddings, we can deompose the word embedding matrix V into the dot product of M' and the sememe embedding matrix X, i.e. V = M'X. During training, the pre-trained word embeddings are kept frozen.

The representation capability of SPASE is limited, especially when modeling the complex semantic relation between sememes and words

### 10.5.2 Sememe Prediction with Internal Information

Previous methods do not consider the internal information in words, such as the characters of Chinese words

#### Sememe Prediction with Word-to-Character Filtering

Similar techniques to collaborative filtering. If two words have the same characters at the same  positions, then these two words should be considered to be similar

uhh not sure if this really applies to english

### 10.5.3 Cross-lingual Sememe Prediction

for other languages, propose task of cross-lingual lexical sememe prediciton (CLSP), which aims at automatically predicting lexical sememes for words in other languages

novel model for CLSP to translate sememe-based KGs from a source language to a target language. Our model mainly contains two modules:

- 1. monolingual embedding learning
  - which jointly learns semantic representations of words for the source and the target languages
- 2. cross-lingual word embedding alignment
  - which bridges the gap between the semantic representations of words in two languages

Learning these word embeddings could conduce to CLSP. The overall objective function is:

$\mathcal{L} = \mathcal{L}_{mono} + \mathcal_{cross}  $

Here, the monolingual term L_{mono} is designed to learn monlingual word embeddings for source and target languages. The cross-lingual term L_{cross} aims to align cross-lingual word embeddings in a unified semantic space.

#### Monolingual Word Representation

Learned using monolingual corpa of source and target languages. Since these two corpa are non-parallel, L_{mono} comprises two monolingual sub-modules that are independent of each other:

$\mathcal{L}_{mono} = \mathcal{L}_{mono}^{S} + \mathcal{L}_{mono}^{T}   $

where the superscripts S and T denote source and target languages. To learn monolingual word embeddings, we choose the skip-gram model, which maximizes the predictive probability of context words conditioned on the centered word. Formally, taking the source side for example, given a sequence $\{w_{1}^{S}, ..., w_{n}^{S}   \}    $, we minimize the following loss:

$\mathcal{L}_{mono}^{S} = - \sum_{c=l+1}^{n-l} \sum_{-l \leq k \leq l, k \neq 0}  \log P(w_{c+k}^{S} | w_{c}^{S}  )   $

where l is the size of the sliding window. $P(w_{c+k}^{S} | w_{c}^{S}  )$ stands for the predictive probability of one of the context words conditioned on the centered word w_{c}^{S}. it is formalized as:

$P(w_{c+k}^{S} | w_{c}^{S}  ) = (\exp(w_{c+k}^{S} .w_{c^{S}} )  )/(\sum_{w_{s}^{S} \in V^{S} \exp(w_{s}^{S} . w_{c}^{S}  )}  )   $

in which V^{S} denotes the vocabluary of the source language. L_{mono}^{T} can be formulated in a similar way

#### Cross-Lingual Word Embedding Alignment

Aims to build a unified semantic space for both source and target languages. The cross-lingual word embeddings are aligned with supervision from a seed lexicon. Specifically, L_{cross} includes two parts:

- alignment by seed lexicon
- alignment by matching

$\mathcal{L}_{cross} = \lambda_{x} \mathcal{L}_{seed + \lambda_{m} \mathcal{L}_{match}}  $

where \lambda_{s} and \lambda_{m} are hyper-parameters balancing both terms. The seed lexicon term L_{seed} pulls word embeddings of parallel pairs to be close, which can be achieved as follows:

$\mathcal{L}_{seed} = \sum_{w_{s}^{S}, w_{t}^{T} \in \mathcal{D}}  (w_{s}^{S} - w_{t}^{T})^{2}  $

where D denotes a seed lexicon w_{s}^{S} and w_{t}^{T} indicate the words in the source and target languages in D

L_{match} is designed by assuming that each target word should be matched with a single source word or a special empty word and vice versa. The matching process is defined as follows:

$\mathcal{L}_{match} = \mathcal{L}_{match}^{T2S} + \mathcal{L}_{match}^{S2T}  $

where $\mathcal{L}_{match}^{T2S}$ and $\mathcal{L}_{match}^{S2T}$ denote target to source matching and source to target matching

Details of target to source and source to target matching: a ltent variable $m_{t} \in \{0, 1, ..., |V^{S}|   \} (t = 1, 2, ..., |V^{T} |) $ is first introduced for each target word w_{t}^{T}, where |V^{S}| and |V^{T}| indicate the vocabulary sizes of the source and target languages. Here, m_{t} specifies the indec of the source word that w_{t}^{T} matches and m_{t} = 0 signifies that the empty word is matched. Then we have m = {m_{1}, m_{2}, ..., m_{|V^{T} |} } and can formalize the target-to-source matching term as follows:

$\mathcal{L}_{match}^{T2S} = - \log P(\mathcal{C}^{T}| \mathcal{C}^{S}) = - log \sum_{m} P (\mathcal{C}^{T}, m|\mathcal{C}^{S} )  $

where \mathcal{C}^{T} and \mathcal{C}^{S} denote the target and source corpus. Then we have:

$P (\mathcal{C}^{T}, m|\mathcal{C}^{S} ) = \prod_{w^{T} \in \mathcal{C}^{T}} P (w^{T}, m|\mathcal{C}^{S}) = \prod_{t=1}^{|V^{T} |} P (w_{t}^{T}| w_{m_{t}}^{S})^{c(w_{t}^{T} )}   $

where $w_{m_{t}}^{S}$ is the source word that is matched by w_{t}^{T} and c(w_{t}^{T}) denotes how many times w_{t}^{T} occurs in the target corpus. here, $P(w_{t}^{T}| w_{m_{t}}^{S})$ is calculated similar to the above fn. The original CLSP model contains another loss function that conducts sememe-based word embedding learning. This loss incorporates sememe information into word representations and conduces to better word embeddings for sememe prediction.

#### Sememe Prediction

Based on the assumption that relevant words have similar sememes, we propose to predict sememes for a target word in the target language based on its most similar source words. Using the same word-sememe matrix M in SPWE, the probability of a sememe x_{j} given a target word w^{T} is defined as:

$P(x_{j}|w^{T}) = \sum_{w_{s}^{S} \in V^{S}} \cos (w_{s}^{S}, w^{T} )M_{sj} c^{r_{s}}  $

where w_{s}^{S} and w_{T} are the word embeddings for a source word w_{s}^{S} and the target word w^{T^{s}}. r_{s} denotes the descending rank of the word similarity \cos(w_{s}^{S}, w^{T}) and c means a hyperparameter

### 10.5.4 Connecting to HowNet and BabelNet

BabelNet is a multi-lingual KB that merges wikipedia and representative linguistic KBs (e.g. WordNet). A node in BabelNet is named BabelNet Synset, which contains a different definition and multiple words in different languages that share the same meaning, together with some additional information. The edges in BabelNet stand for relations between synset like *antonym* and *superior*

#### Sememe Prediction for BabelNet Synsets (SPBS)

#### SPBS with Semantic Representation (SPBS-SR)

#### SPBS with Relational Representation (SPBS-RR)

borrows the idea from TransE

## 10.6 Applications

### 10.6.1 Chinese LIWC Lexicon Expansion

Linguistic inquery and word count (LIWC) has been widely used for computational text analysis in social science. LIWC computes the percentage of words in a given thext that fall over 80 linguistic, psychological, and topical categories. Can be used for classification and to examine the udnerlying psychological states of a writer or speaker. Has been widely applied in computational linguistics, demographics, health diagnostics, social relationship, etc

### 10.6.2 Reverse Dictionary

Task of reverse dictionary is defined as the dual task of the normal dictionary: it takes the definitino as input and outputs the target words or phrases that match the semantic meaning of the definition

Some commercial reverse dictionary systems are satisfying in performance, but are closed source. Existing reverse dictionary algorithms face the following problems:

- 1. Human-written inputs differ a lot from word definitions, and models trained on the latter have poor generalization abilities on user inputs
- 2. It is hard to predict those low-frequency target words due to limited training data for them. They may actually appear frequently according to Zipf's law

#### Multi-channel Reverse Dictionary Model (MCRD)

Utilizes POS tag, mopheme, word category, and sememe information of candidate words. MCRD embeds the queried definition into hidden states and computes similarity scores with all the candidates and the query embeddings

#### Basic Framework

embeds the queried definition into representations i.e. Q = {q_{1}, ..., q_{|Q|}}. The model feeds Q into a BiLSTM model and obtains the hidden states as follows:

$h_{i} = concat(\overrightarrow{h_{i}}; \overleftarrow{h_{i}}  )  $

Then the hidden states are passed into a weighted summation module, and we have the definition embedding v:

$v = \sum_{i=1}^{|Q|} \alpha_{i} h_{i} $

$\alpha_{i} = h_{t}^{\top} h_{i}  $

$h_{t} = concat(\overrightarrow{h_{|Q|}}; \overleftarrow{h_{l}}  ) $

Finally, the definition embedding is mapped into the same semantic space as words, and dot products are used to represent word-word confidence score sc_{w, word}:

$v_{word} = W_{word}v + b_{word}  $

$sc_{w, word} = v_{word}^{\top}w $

where W_{word} and b_{word} are trainable weights and w denotes the word embedding

#### Internal Channels: POS Tag Predictor

to return words with POS tags relevant to the input query, we predict the POS tag of the target word. The intuition is that human-written queries can usually be easily mapped into one of the POS tags

Denote the union of the POS tag of all the senses of a word w as P_{w}. we can compute the POS score of the word w with the sentence embedding v:

$sc_{pos} = W_{pos}v + b_{pos}  $

$sc_{w, pos} = \sum_{p \in P_{w}} [sc_{pos}]_{index^{pos}(p)} $

where $index^{pos}(p)$ means the id of POS tag p and operator [x]_{i} denotes the i-th element of x. In this way, candidates with qualified POS tags are assigned a higher score

#### Internal Channels: Word Category Predictor

Semantically related words often belong to distinct categories, despite the fact that they could have similar word embeddings (for instance, "bed" and "sleep"). Word category information can help us eliminate these semantically related but not similar words:

$ sc_{cat,k} = W_{cat,k}v + b_{cat,k} $

$sc_{w,cat} = \sum_{k=1}^{K} [sc_{cat,k}]_{index_{k}^{cat}(w)}  $

where W and b are trainable weights. K denotes the number of the word hierarchy of w and $index_{k}^{cat}(w)$ denotes the id of the k-th word hierarchy

#### Internal Channels: Morphememe Predictor

Similarly, all the words have different morphemes, and each morpheme may share similarities with some words in the word definition. Therefore, we can conduct the morpheme prediction of query Q at the word level

$sc_{mor}^{i} = W_{mor}h_{i} + b_{mor}   $

Where W and b are trainable weights. The final score of whether the qyery Q has the morpheme j can be viewed as the maximum score of all positions:

$[sc_{mor}]_{j} = \max_{i \leq |Q|} [sc_{mor}^{i}]_{j}   $

where the operator $[x]_{j}$ means the j-th element of x. Denote the union of the morphemes of all the senses of a word w as M_{w}, we can then compute the morpheme score of the word w and query Q as follows

$sc_{w,mor} = \sum_{m \in M_{w}} [sc_{mor}]_{index^{mor} (m)}   $

where $index^{mor} (m)$ means the id of the morpheme m

#### Internal Channels: Sememe Predictor

similar to morpheme predictor, we can also use sememe annotations of words and the sememe predicitons of the query at the word level and then compute the sememe score of all the candidate words:

$sc_{sem}^{i} = W_{sem}h_{i} + b_{sem}  $

$[sc_{sem}]_{j}  = \max_{i \leq |Q|} [sc_{sem}^{i}]_{j}  $

$sc_{w,sem} = \sum_{x \in X_{w}} [sc_{sem}]_{index^{scm}(x)} $

where X_{w} is the sememe set of all w's sememes and $index^{scm}(x)$ denotes the id of the sememe x. With all the internal channel scores of candidate words, we can finally get the confidence scores by combining them as follows:

$sc_{w} = \lambda_{word} sc_{w, word} + \sum_{c \in \mathbb{C}} \lambda_{c} sc_{w,c}  $

where \mathbb{C} is the aformentioned channels:

- POS tag
- morpheme
- word category
- schemes

A series of \lambda are assigned to balance different terms. With the sememee annotations as additional knowledge, the model could achieve better performance, even outperforming commercial systems

#### WantWords

open-source online reverse dictionary system based on multi-channel methods

BERT as sentence encoder, and is stable and flexible

## 10.7 Summary and Further Readings

### Further Reading and future Work

sememes appear to be the future

research directions:

- 1. Building Sememe KBs for Other Languages
- 2. Utilizing Structures of Sememe Annotations
  - existing attempts do not fully exploit the structural information of sememes
  - current methods basically just regard them as semantic labels
- 3. Leveraging Sememe Knowledge in Low-Resource Scenarios
  - where there are low-frequency words
