# Chapter 4 - Sentence and Document Representation Learning

Sentence and document are high level linguistic units of natural languages. Representation learning of sentences and documents remains a core and challenging task because many important applications of NLP lie in understanding sentences and documents. This chapter first intriduces symbolic methods to sentence and document representation learning. Then we extensively introduce neural network-based methods for the far-reaching language modeling task, including feed-forward NNs, CNNs, RNNs, and Transformers. Regardless the characteristics of a document consisteing of multiple sentences, we particularly introduce memory-based and hierarchical approaches to document representation learning. Finally, we present representative applications of sentence and document representation, including text classification, sequence labeling, reading comprehension, question answering, information retreival, and seq2seq generation

## 4.1 Introduction

A natural language sentence is a linguistic unit that conveys complete semantic information, which is composed of words and phrases guided by grammatical rules.

## 4.2 Symbolic Sentence Representation

When words and phrses form sentences, they obtain complete semantics.

### 4.2.1 Bag-of-Words Model

V = {w1, w2, ..., w|V|}

one hot representation of word w is 2 = [0, ..., 0, 1, 0, ..., 0]

s(entence) = {w1, w2, ..., wN}

$s = \sum_{i=1}^{N} w_{i}$

the sentence representaiton s is the some of the one-hot representations of N words

Each element in s represents the term frequency (TF) of the corresponding word

Tf alone cannot properly represent a sentence or document since not all the words are equally important. A, an, the usually appear in almost all sentences and reserve little semantics that could represent the sentence or document.

Inverse Document Frequency (IDF) is developed to measure the prior importance of wi in V

$idf_{w_{i}} = log(|D|/df_{w_{i}})$

where |D| is the number of all sentences or docs in the corpus D and df_wi is the document frequency of wi, which is the number of documents that wi appears

The insight behind it is that the more frequently a word appears and the less it appears in other texts, the more it represents the uniqueness of the current text and thus will be assigned more weight. TF-IDF is one of the most popular methods in information retrieval and recommender systems

### 4.2.2 Probabilistic Language Model

A std probabilistic language model defines the probabilitity of a sentence s = {w1, w2, ..., wN} by the chain rule of probability

P(s) = P(w1)P(w2|w1)P(w3|w1, w2)...P(wN|w1, ..., wN-1)

$P(s) = \prod_{i-1}^N P(w_{i}|w_{1},...,w_{N-1})$

The probability of each word is determined by all th epresceeding words. and the probabilities of all the words jointly compute the probability of the sentence

In practice, we set n-1-sized context windows, assuming that the probability of word only depends on {w_i=n+1 ... w_i-1}

After simlifying the language model, how to estimate the conditional probability is cruicial. In practice, a common approach is maximum likelihood estimation (MLE), which is gernerally in the following form

$P(w_{i}|w_{1-n+1},...,w_{i-1}) = (P(w_{i-n+1}, ..., w_{i}))/(P(w_{i-n+1}, ..., w_{i-1}))$

In this equatino, the denominator and numerator can be estimated yb counting the frequencies in the corpus. To avoid the probability of some n-gram sequences from being zero, researchers also adopt several types of smoothing approaches, which assign some of the total probability mass to unseen words or n-grams such as:

- "add-one" smoothing
- Good-Turing discounting
- back-off models

Recent workds on word representation learning are mainly based on the n-gram language model

## 4.3 Neural Language Models

A neural network could also be viewed as an estimator of the language model function. Similar to n-gram probabilistic language models, neural language models are constructed and trained to model a probability distribution of a target word conditioned on previous words

$P(s) = \prod_{i=1}^{N} P(w_{i}|w_{i-n+1},...,w_{i-1})$

where the conditional probabiliyt of selecting word wi can be calculated by multiple kinds of nns and the common choices include the FF-NN, RNN, CNN, etc. The training of neural language models is achieved by optimizing the cross-entropy loss function:

$\mathcal{L} = - \sum_{i=1}^{N} log P(w_{i}|w_{i-n+1},...,w_{i-1})$

### 4.3.1 Feed-Forward Neural Network

To evaluate the conditional probability of the word wi, it first projects an n - 1 context related words to their *word vector representations* [w_i-n+1, ..., w_i-1] and concatenate the representations x = concat(w_i-n+1;...;w_i-1) to feed them into a FFN

h = Mf(W1x+b) + W2x + d

where f(.) is an activatino function, W1, W2 are weighted matrices to transform word vectors into hidden representations, M is a weighted matric for the connections between the hidden layer and the output layer, and b, d are bias terms. Then, the conditional probability of the word wi can be calculated by a softmax function

$P(w_{i}|w_{i-1+1},...,w_{i-1}) = Softmax(h)$

### 4.3.2 Convolutional Neural Network

For the input words {w1, ..., wl} we first obtain their word embedding [w1, ..., wN]. let d denote the dimension of the hidden states. The convolutional layer involves a sliding window within the size of k of the input vectors centered on each word vector by using a kernel matrix Wc, and the hidden representation could be calculated by

h = f(X*Wc + b)

whre * is the convolution operation, f(.) is a nonlinear activatino function (e.g. a sigmoid or tangent function), X E R^lxd is a matrix of word embeddings, Wc E R^kxdxk' (d' kernel size) and b E R^d' are learned parameters. The sliding window prevents the model from seeing the subsequent words so that h does not learn information from future words. For each sliding step, the hidden state of the current word is computed based on the previous k words, and then further fed to an output layer to calculate the probability of the present word. In practice, we can use distinct lengths of sliding windows to form multi-channel operations to learn local information with different scales

### 4.3.3 Recurrent Neural Network

To address the lack of ability to model long-term dependency in the FFN language model, a RNN was proposed. RNNs are difference in the fundamental way that they operate in an internal state space where representations can be sequentially processed. Therefore, the RNN LM can deal with those sentences of arbitrary length. At every time step, its input is the vector of its previous word instead of the concatenation of vectors of its n-1 previous words. The information of all other previous words can be considered by its internal state

Given the input word embeddings x = [w1, w2, ..., wN] at timestep t, the current hidden  state h_t is computed based on the current inptu wt and the hidden state of the last timestep h_t-1, formally:

$h_{t} = f(W concat(w_{t};h_{t-1}) + b)$
$y = Softmax(Mh_{t} + d)$

Where f(.) is a nonlinear activation function, y represents a probability distribution over the given vocabulary, W and M are weighted matrices and b,d are bias terms. As the increase of the lengths of the sequence, a common issue of the RNN language model is the *vanishing gradients probem*. Can be implemented as LSTM and GRU

#### LSTM

LSTM introduces a cell state c_t to represent the current information at timestep t, which is computed from the cell state at the last timestep c_t-1 and the candidate cell state of the curren timestep c_t(~hat)

- c_t(~hat) = tanh(Wc concat (wt;h_t-1) + bc)
- c_t = f_t dot c_t-1 + i_t dot c_t(~hat)
- h_t = o_t dot tanh(c_t)

where dot is the element-wise multiplication operation, Wc and bc are learnable parameters and ft, it, ot, are different gates introduced in LSTM to control the information flow. Specifically, ft is the forgetting gate to determine how much information of the cell state at the last timeste c_t-1 should be forgotten, it is the input gate to control how much information of the candidate cell state at the current timestep should be reserved, and ot is the output gate to control how much information of the current cell state ct should be output to the representaiton ht. And all these gates are computed by the representation of the last timestep ht-1 and the current input wt, formally:

- f_t = Sigmoid(Wf concat(wt;ht-1) + bf)
- i_t = Sigmoid(Wi concat(wt;ht-1) + bi)
- o_t = Sigmoid(Wo concat(wt;ht-1) + bo)

where Wf, Wi, Wo are weight matrices and bf, bi, bo are bias terms in different gates. It is generally believed that LSTM could model longer text than the vanilla RNN model

#### GRU

To simplify LSTM and obtain more efficient algorithms, simple but comparable RNN architecture, uses gating mechanism to handl einformation flow.

Compared to several gates with different functionalities, GRU uses an update gate zt to control the information flow. And a reset gate rt is adopted to control how much information from the last step hidden state ht-1 would flow into the candidate hidden state of the current ste h~, formally

- s_t = Sigmoid(Wz concat(wt;ht-1) + bz)
- r_t = Sigmoid(Wr concat(wt;ht-w) + br)
- h~t = tanh(Wh concat(wt;rt dot ht-1) + bh)
- ht = (1-zt) dot ht-1 + zt dot h~t

where Wz, Wr, Wh are weight matrices and bz, br, bh are bias terms. The update gate z in GRU simultaneously manages the historical and current information. Moreover, the model also omits cell modules c in LSTM and directly uses hidden states h in the computation. GRU has fewer parameters, which brings higher efficiency and could be seen as a simplified version of LSTM. Generally compared to CNNs, RNNs are more suitable for the sequential characteristic of textual data. However, the nature of each steps hidden state is dependent on th previous step also makes RNNs difficult to perform parallel computation and thus slower in training.

### 4.3.4 Transformer

#### Structure

A transformer is a nonrecurrent encoder-decoder architecture with a series of attention-based blocks. For the encoder, there are multiple layers, and each layer is composed of a multi-head attention sumlayer and a position-weise feedforward sublayer. And there is a residual connection and layer normalization of each sublayer. The decoder also comtains multiple layers, and each layer is slightly different from the encoder. First, sublayers of multi-head attention and feed-forward with identical structures with the encoder are adopted. And the input of th emulti-head attention sublayer is from both the encoder and the previous sublayer, which is additionally developed. This sublayer is also a multi-head attention sublayer that performs self-attention ove rthe outputs of the encoder. And the sublayer adopts a masking operation to prevent the decoder from seeing subsequent tokens

#### Attention

There are several attention heads in the multi-head attention sublayer. A *head* represents a scaled dot-product attention structure, which takes the query matrix Q, the key matrix K, and the value matrix V as the inputs and the output is computed by

$ATT(Q,K,V) = softmax((QK^{T})/(\sqrt(d_{k})))V$

where dk is the dimension of the query matrix; note that in LMs, Q, K, and V usually come from the same source, i.e. the input sequences. Sepcifically, they are obtained by the multiplication of the input embedding H and three weight matrices, W^Q, W^K, and W^V, respetively. The dimensions of query, key, and value vectors are dk, dk, and dv respectively. The computation (above) is typically known as the self-attention mechanism

The multi-head attention sublayer linearly projects the input hidden states H several times into the query matric, the key matrix, and the value matrix for h heads. The multi-head attention sublayer could be formulated as follows:

Multihead(H) = [head1, head2, ..., headh]W^O

Where $headi = ATT(QW_{i}^Q, KW_{i}^K, VW_{i}^V)$ and $W_{i}^Q$, $W_{i}^K$, and $W_{i}^V$ are linear projections. $W^{O}$ is also a linear projection for the output

After operating self-attention, the output would be fed into a fully connected position-wise feed-forward sublayer, which contains two linear transformations wih ReLU activation

FFN(x) = W2 max(0, W1x+b1) + b2

#### Input Tokenization

Tokenization is a cruicial step in NLP to process raw input sequences

Mature approaches:

- byte pair encoding (BPE)
  - iteratively replaces two adjacent units with a new unit
  - ensures that common words will remain as a whole and uncommon words are split into multiple subwords
  - applied to RoBERTa, GPT-2
- wordpiece
  - BERT

#### Positional Encoding

Positional encoding indicates the position of each token in an input sequence. The self-attention mechanism of Transformers does not involve positional information. Thus, the model needs to represent positional information of the input sequence additionally. Transformers do not use integers to represent positions because th value range varies with the input length. For example, positional values may become very large if the model process a long text, which will restrain the generalizatio over texts of different lengths

specifically, each position is encoded to a particular vector with the same dimension d of the hidden states to represent the positional information. For the kth token, let pk be the positional vector, the ith element of the positional encoding pik

$p_{i}^{k} = sin(k/(10,000^{2j/d}))$ if i = 2j

$p_{i}^{k} = cos(k/(10,000^{2j/d}))$ if i = 2j+1

In this way, for each positional encoding vector, the frequency woudl increase along wiht the dimension. We can imagein that at the end of each vector, the k/1000xxx is near 0 since the denominator becomes very large, which makes sin(k/10000) approximate 0s and cos(k/10000) approximates 1

Assuming the state of alternating 0s and 1s is a kind of "stable point" for different positions k, the "speed" to reach each stable point is also different. that is, the later the token is (larger k), the later the value k/10000 will be close to 0. moreover, no matter the text lengths the model is currently processing, the encoder values are stable and range from -1 to 1. Alternatively, learnable positional embeddings could also be applied to transformers, and could consistently yield similar performance. Pre-trained LMs like BERT adopt learnable position embeddings rather than sinusoidal encoding.

As stated, the overall objective is

$\mathcal{L} = - \sum_{i=1}^{N} log P(w_{i}|w_{i-n+1},...,w_{i-1})$

Here, we use the decoder of a transformer to adopt the self attention mechanism to the previous n-1 words of the current word, and the output layer will be further fed into the feed-forward sublayer. After multiple layers of propogation, the final probability distribution p is computed by a softmax function acting on the hidden representation. Compared to RNNs, transformers could better model the long-term dependency, where all tokens will be equally consdiered and computed during the attention operation

### 4.3.5 Enhancing Neural Language Models

#### Word Classification

Researchers propose a class-based language model to adopt word classification to improve the performance and speed of the LM.  All words are assigned to a unique class and the conditional probability of a word given its context can be decomposed into the probability of the words class given its previous words and the probability of the word given its class and history

Also proposed a hierarchical nn language model, which extends word classificaiton to hierarchical binary clsutering of words in the LM. Instead of simply assigning each word a unique class, it first builds a hierarchical binary tree of words according to the word similarity obtained from WordNet. next, it assigns a unique bit vector [c1(w1), c2(wi), ..., cN(wi)], for each word, which indicates the hierarchical classes of them. And then, the conditional probability of each word can be defined as

...

the hierarchical nn lm can achieve O(k/logk) speed up as compared to a std language model. However, experimental results show that it performs worse than the std LM. The reason is that the introductino of hierarchical architecture or word classes imposes anegative influence on word classificaiton by nn lms

#### Caching

One of the important extensions of neural LMs. Assumes each word in a recent context is more likely to appear again

Another cache-based lm is also used to speed up the RNN LM. main idea is to store the outputs and states of LMs for future predictions given the same contextual history

## 4.4 From Sentence to Document Representation

### 4.4.1 Memory-Based Document Representation

A direct way to learn the document representation is to regard the document as a whole. The intuition is to use inherent modules to remember the context with critical information of the target document

#### Paragraph Vector

extend word2vec to the document level

given a target word and the corresponding contexts from the document, the training objective of this strategy is to use the PV to predict the target word

- distributed memory (PV-DM)
  - adds an additional token in each document and uses token representaiton to represent the document
  - predicts the target word according to historical contexts and document representatino in the training phase
- distributed bag of words (PV-DBOW)
  - extends the idea of skip-gram to learn document representation
  - ignores the context words in the input text and directly uses the document representaiton to rpedict the target word in a randomly sampled window
  - in training, model will randomly sample a window then randomly sample a word to be the prediciton target
  - simpler than PV-dm
  - experiements so the method is also effective

when doc is too long, PV may not have enough capacity to remember enough information. could be alleviated by NNs

#### Memory Networks

uses memory units to store and maintain long-term information. compared to std nns that utilize special aggregation operations to obtain the document representation, memory networks explicitly adopt memory nn modules to store information, which could prevent information forgetting

given an input doc with n sentences d = {s1, s2, ..., sn} for th ith sentence si, the model firstly uses a feature extractor F to transform the input into a sentence-level representation h_i^s

a memory unit M is responsible for storing and updating memories according to the current inputs. In this case, the memories will be updated by certain operations. For the specific update mechanism of the memory module, there are many options to define M. Here we introduce one of the most straightforward moethods by using slots. The basic idea of this approach is to store the preresentation of each input into a separate "slot"

#### Variants of Memory Networkds

Training strategy - if the operation of each module is designed discretely, it is not easy to directly train the network via back-propagation

Dynamic memory networks present a similar methodology. After the model produces representations for all the input sentences and their current query, the query representaiton will trigger a retrival procedure based on the attention mechanism. This procedure will iteratively read the stored memories and retrieve the relevant ones to produce the output

Memory form:

hierarhical memory networks give a solution that organizes memories in a hierarchical form. This method forms a group of memories, and then multiple groups can be reorganized into higher level groups

Key-value memory network (KV-MemNN) uses a key-value strucutre to store and organize memories. The design of such a structure is to boost the proces sof retrieving memories and coust store information from different sources (e.g. text and KGs). Formally, the memories are pairs of key-values (ki, vi). suppose that we already have large-scale established key-value memories, given a query q and the corresopnding representation q; the model could use it to preselect a small group of memories by directly matching the words in the query and memories with the reverse index. After narrowing the search space, one can calculate the relevant score between query and a key:

pi = Softmax(fQ(q) * fK(ki))

where fQ(.) and fK(.) are feature mapping functions.

similar to the end-to-end memory network, in the reading stage, the vector o responsible for the final output could be calculated by a weighted sum

At this time, key-value memories conduct interactions with the output candidates. In the training pahse, all the aforementioned feature mapping functions and trainable parameters are optimized in an end-to-end manner. KV-MemNN could also be generalized to a variety of applications with different forms of knowledge by flexibly designing fQ, fK, and fV. For example, for storing world knowledge in the form of a triplet, we can regard the head entity and the relation as the key and the tail entity as the value. For textual knowledge, we can encode sentences or words directly into both key and value in practice

Although memory networks are proposed for better document modeling, it has prodoundly influenced the academic community with this idea. We can use additional modules to store information explicitly and enhance the memory capacity of neural networks. To this day, this framework is still a common idea for modeling very long texts. There fore there are three key points in designing such a network:

- representaiton learning of memory
- the matching mechanism between memory and query
- how to perform memory retrieval based on the input efficiently

#### Hierarchical Document Encoder

The basic idea is to use low-level representations to produce high-level representations. first, the word vectors obtained by pre-training with self-supervised methods can be directly used as the basic word representations. We can also optimize these word representations according to specific tasks. And there are various ways to get the sentence representation through the constituent word representations. Ex. let it pass through a layer of MLP then average over all the hidden states. Or, with LSTM, passing thru, the hidden state of the last time step contains the semantic information of the whole sentence, and can be used as a sentence representation. can stack all the sentences again, and keep last hidden unit as doc represent.

To this end, we introduce basic hierarchical modeling of doc repesentation. When there is a supervised signal, we can use this doc rep directly for nn training with doc level classification. if no supervised signal, we can self-code the doc rep which can be decoded in reverse order, i.e. first decode the edoc rep into a sentence rep then gen words sequentially

#### Hierarchical Attention Network

replace LSTM with more powerful nn with attention mechanisms

HAN = hierarchical attention network

the key insight of this model is that while doing hierarchical modeling, different attention weights are assigned to comopnents (words and sentences) using the attention mechanism to learn the docs rep dynamically

here, use bidirectional GRU - concat fwd and backward pass final hidden state to get final representation

## 4.5 Applications

- classification
- information retrieval
- reading comprehension
- open-domain question answering
- sequence labeling

### 4.5.1 Text Classification

#### Topic classification

- CNNs are commonly used as representation encoders

#### Sentiment Classification

#### Natural Language Inference

classification task involving two sentences. its objective is to determine whether the first sentence entaisl the second sentence or not

### 4.5.2 Information Retrieval

For the given query q and document d, traditional IR models estimate their relevance through lexical matches

neural information retrieval models pay more attention to garnering the query and document relevance from semantic matches. both lexical and semantic matches are essential for neural information retrieval

Neural ranking models typically fall into two groups:

- representation-based
  - learn informative representations and match them in the embedding space of queries and documents
- interaction-based
  - model the query-document matches from the interactions of their terms

Studies in the early stage primarily focus on representation-based models

### 4.5.3 Reading Comprehension

- Cloze Style
  - consists of filling in the blank sentences where the quesion contains a placeholder to be filled in
  - the answer is either from a predefined candidate set or the vocab
- Multiple-Choice
  - aims to select the best answer from a set of answer choices
  - typical to use accuracy to measure the performance on these two tasks:
    - pct of correctly answered questions
- Span Prediction
  - SQuAD
- Free-Form Answer
  - aka generative question answering

Wth nns, the machine reading comprehension system is commonly composed of three consecutive phases:

- the embedding phase
- the reasoning phase
- the prediction phase

like many other NLP tasks, embedding phase often adopts pre-trained or contextual word embeddings with RNNs, character embedding, or hybrid embedding

the query and the context are separately encoded

the reasoning phase is resp for joint learning based on the two representations and is the focus of most works

while encoding a passage, the model retains the length of the sequence and encodes the question into a fixed-length hidden rep q. the questions hidden vector is then used as a pointer to scan over the passage representation and compute scores on every position in the passage. while maintaining this similar architecture, most machine reading comprehension models vary in the interaction methods betwen the passage and the question. most of them merge two lines of info from the query and the context with the attention mechanism. they mainly differ in two aspects: the *direction* of attention and teh *dimension* of attention. Direction refers to whether using query-to-context attn or both directions. dimension refers to whether attention is only calculated at thesentence representation level, which outputs a single dim vector, or at the word embedding level, where output is an embedding matrix

#### Single Driection and Single Dimension

First attempt to apply nn to mrc constructs a bidirectional LSTM models along with attention mechanisms. Work introduces two reader models, the attentive reader and the impatient reader. after encoding passage and query into hidden states with lstm, *attentive reader* computes a scalar distribution over the passage tokens and uses it to calc the weighted sum of the passages hidden states. the *impatient reader* extends this idea further by repeatedly updating the weighted sum of passage hidden states after seeing each query token. one approach modifys the method to compute attention and simplify the prediction layer in the attentive reader with a simple bilinear term

#### Bidirectional Attention and Single Dimension

The attention-over-attention reader also computes both query-to-context and context-to-query attention but handles them differently. Instead of simply averaging the token-level query-to-context attention to obtain a final vector for prediction, attention-over-attention computes a weighted vector with a query word importance vector. The word importance vector is computed by averaging the context-to-query attention. This operation is considered to learn the contributions of individual question words explicitly

#### Bidirectional Attention Flow and Multi-Dimension

Instead of unifying the doc and query rep into a single vector with query to context attn opnly, BiDAF network computes teh attentive token rep of both query-to-context and context-to-query at each bidirectional LSTM layer to allow fine-grained information flow. it consists of the token embedding layer, the contextual embedding layer, the bidirectional attn flow layer, the lstm modeling layer, the softmax output layer. At each layer, the input is the concat of the previous layers hidden states, the query-to-context rep, and the contex-to-query rep. the rep of multipl granularities and a bidir attn flow can fully capture the interaction bw doc and query for start and end position prediction

The gated-attn reader adops the gated attn module, where each token rep of the passag is scaled by the attended query vector after each BiGRU layer. htis gated attn mech allows the query to interact directly with the token embeddings fo the passage at the semantic level. And such layer-wise interaction enables the model to learn conditional token rep given the question at different rep levels

### 4.5.4 Open-Domain Question Answering

OpenQA aims to anser open-domain questions utilizing external resources such as collections of docs, webpages, structured kgs, or automatically extracted relatinoal triples. REcently, with dev of mrc techniques, researchers attempt to answer open-domain questions via performing reading comprehension on plain texts with neural-based models.

Essentially, two critical applications are combined:

- information retrieval
- reading comprehension

DrQA has two modules:

- 1. one document retriver module to retrieve relevant articles or paragraphs
- 2. one document reader to produce the final answers from extracted articles

the doc retriever is used as first quick skim to narrow the search space and focus on potentially relevant docs. this retreiver builds TF-IDF weighted bag-of-words vectors for the docs and the questions and computes similarity scores for ranking. The retriever uses bigram counts with hash to further utilize local word order information while ensuring speed and memory efficiency. The doc reader model takes in the top 5 wiki articles from the doc retriever and extracts the final anser to the question. the doc reader predicts an asnwer span with a conf score for each article. the final pred is made by maximizing the unnormalized exponential pred scores across the docs

given each doc, the doc reader first builds a feature representation for each word in the doc, often the concat of:

- 1. word embeddings (like glove)
- 2. manual features (POS tagging, Term Frequncies)
- 3. Exact match: feature indicates whether the word in the doc can be precisely matched to one question word
- 4. Aligned question embeddings - feature aims to encode a soft alignment between words in the doc and the question in the word embedding space

Then the feature representations of the doc is fed into a multi-layer bidir LSTM to encode the contextual rep. for the question, the contextual rep is simply obtained by encoding the word embeddings using a multi layer bilstm. after that, the contextual rep is aggregated into a fixed-length vector using self-attention. in the answer prediciton phase, the start and end probability distributions are calculated

DrQA is prone to noise in retrieved texts. several approaches to address by using two separate procedures for qa: paragraph selection and answer selection, however they both only select the most relevant para among all retrieved paras to extract ansers and may lose valuable information distributed in other paras

another is strength-based and coverage-based methods for re-ranking, aggregating the answers that existing methods retrieved from all the paras. nevertheless, the challenge of noisy data is still unsolved. to address this issue, a coarse-to-fine denoising OpenQA model is developed to first screen out relevan tparas and then retreive conrrect answers

### 4.5.5 Sequence Labeling

Given an input squence {w1, ..., wN} we need to assign a label yi to each token wi. part of speech tagging and NER are the two most representative tasks

sequence labeling requires the model to caputre the correlations of words in the sequence accurately. Hence, classica pproaches use PGMs to represent the dependency structure of different words. Modern methods use powerful deep NNs to produce richer representations and adopt conditional random field (CRF) or direct token-level classification to conduct sequence labeling

#### POS tagging

#### NER

BIO schema - beginining, inside, outside

dude i really need to play with GliNER, that framing is so interesting

### 4.5.6 Sequence-to-Sequence Generation

Refers to a group of tasks that require sequence generation based on an input sequence, including machine translation, text summarization, question generation, tec. A famous model structure ofr seq2seq is an encoder-decoder structure

#### Metrics

- BLEU
  - is an adjusted precision calculation based on the count of n-grams
  - first, extracts all n-grams in the output sequence
  - then, calculates the sum of occurences of the n-grams in the ref sequence against the total number of n grams in the output squence
- ROGUE
  - group of metrics often used in evaluating text summarization systems
  - ROGUE-N (ROGUE-1 AND ROGUE-2)

#### Machine Translation

#### Text Summarization

typically seq2seq models can be applied to machine translation and text summarization since the task is of the same form

pointer-generator network is one of the most classical text summ models that combine LSTM-attention-based encoder-decoder with pointer network. the basic structure contains a single-layer LSTM decoder. apart from the std encoder-decoder pipeline, it applies an extra pointer while decoding. the pointer depends on the encoder output, the current decoder hidden states, and decoder input and calculates the probability pgen indicating how much we favor the decoder generated results. the final distribution from which the next token is drawn is a weighted sum of distribution given by the decoder dis given by attention weights of the encoder output, weach weighted by pgen and 1 - pgen. so the pointer serves as a mediator between generated tokens and copied tokens from the original input. it is especially beneficial for text summarization as copying original words from the input can help keep the semantics on the right track
