# Chapter 1. Introduction

We are witnessing the current surge in complex multimedia data. The increasing popularity of binary representation learning aims at transforming high-dimensional visual data into informative representations, and due to its compressive and discrete nature, it has been well applied in domains such as machine learning and computer science. Learning compact binary representations from images can improve the efficiency of storage, retrieval, and analytical processes. Essential aspects and applications of binary representation learning in visual imaging encompass efficient image compression to reduce storage requirements, rapid access to similar images through visual similarity searching in vast databases, and the development of content-based image retrieval systems that concentrate on the visual characteristics of images.

- formalized definition of binary representation learning
- four aspects of research
  - asymmetric discrete hashing
  - ordinal-preserving hashing
  - deep collaborative hashing
  - trustworthy depe hashing
- benchmark datasets

## 1.1 What is Binary Representation Learning

binary representations play a cruicial role in transforming complex visual data into informative representations. However, still highly compressed and are discrete in nature

Binary representation learning = hashing = learning to hash

Unlike real-valued representation learning methods, hashing offers scalable and efficient mechanism for preserving similarity with satisfactory accuracy in the fast Hamming space

Codes are typeically no more than 128 dims

The main objective of hashing is to: maintain similarity relationships while compressing features and require consistent measurements between the original visual features and the learned binary codes. Make sure that similar samples have similar binary representations

`dataset` $X = [x_{1}, ..., x_{n}] \in \mathbb{R}^{d \times n}  $ includes n image samples, and `each image` is represented as a d-dimensional feature vector $x_{i} \in \mathbb{R}^{d}$. The corresponding labels are matrix $Y = [y_{1}, ..., y_{n}] \in \{0, 1  \}^{c \times n}  $, where c is the number of classes, and y_{ji} = 1 if sample x_{i} belongs to the j-th class. The goal of learning to hash is to construct a set of hashing functions ${h_{k}(x)}_{k-1}^{l} \in \mathcal{H}  $ to project the original real-valued features from the Euclidian space onto a low-dimensional Hamming space {-1, 1} that is

$b_{i} = \mathcal{H}(x_{i}) = [h_{1}(F(x_{i})), ..., h_{l}(F(x_{i}))  ] \in \{-1, 1 \}^{l}  $

where b_{i} is the binary representation of x_{i}, and F(x_{i}) is the feature encoder of the i-th sample x_{i}, such as projection, multi-layer perceptron, CNN, RNN, Transformer

Supervised hashing is categorized into three primary appproaches:

- piecewise similarity-preserving hashing
- pairwise similarity-preserving hashing
- ranking-based similarity-preserving hashing
  - either triplets or listwise

### Piecewise Similarity-preserving Hashing

Revolves around using class labels as a supervisory measure for conducting training in classification or regression. The fundamental learning framework:

$\min_{b_{i}} \sum_{i}^{n} \|F(x_{i}) - b_{i}  \|_{p} + \mathcal{R}(b_{i}, y_{i})  $

where $\|.\|_{p}$ denotes the $l_{p}$-norm and $\mathcal{R}(b_{i}, y_{i})$ builds the relationship between the learned binary codes and the semantic labels

### Pairwise Similarity-preserving Hashing

Generates hash codes whilem maintaining the relationships between data pairs, incorporating pairwise supervision

Most prevalent in contemporary reserarch on BRL.

It is undestaood that the similarity between hash codes can be quantified by their inner product, which conversely relates to the Hamming distance.

The binary represetation expressed as $B = [b_{1}, ..., b_{n}] $, then we can formulate a widely used obj function which aims to

Objective: **minimize the quantization error bw the inner product of the binary codes and their semantic similarity**

$\|lS - B^{\top}B  \|_{F}^{2} s.t. B \in \{-1,1 \}^{l \times n}  $

where $S \in {0, 1}^{m \times m}  $ represents the semantic pairwise similarity matrix, which is determined by supervised labels, with m being the count of chosen data points. Specifically, if S_{ij} = 1, the ith and jth point are semantically similar either due to proximity as per a distance metric or because they share at least one label. S_{ij} = 0 denotes dissimilar, as non-neighbors or not sharing labels

Approach initially established in the *kernel supervised hashing (KSH)* which has gained popularity

### Ranking-based Similarity-preserging Hashing

Focuses on either preserving the triplet-based supervision or listwise supervision. Achieved by

Objective: **minimizing the empirical loss from ranking discrepancies, grounded in the provided ground-truth rank order supervision**

For each data sample, these methods generate a ranking list vector over k selected anchor points from the original dataset:

${r_{i}^{j}}_{j=1}^{k} = [r_{i}^{1}, ..., r_{i}^{k} ] \in \{1, ..., k  \}^{1 \times k}  $

where r_{i}^{k} means the ranking order of the k-th anchor point wrt the ith training sample. As such, the overall learning scheme is:

$\min_{b_{i}} \sum_{i}^{n} \|F(x_{i}) - b_{i}\|_{p} + K(b_{i}, \{r_{i}^{j} \}_{j=1}^{k} )  $

where $K(b_{i}, \{r_{i}^{j} \}_{j=1}^{k} )$ denotes the ranking/listwise similarity measurement loss for the generated binary code b_{i} of the i-th training sample x_{i}

### Back to main branch

Areas of interest:

- streamlined storage and data transmissino
- rapid search and retrieval processes
- lower memory usage
- learning that safeguards privacy
- neural network quantization
- enhancement of hardware processing speed

## 1.2 Binary Representation - Learning on Visual IMages

### 1.2.1 Asymmetric Discrete Hashing

Utilized for:

- similarity searches
- data retreival
- indexing purposes

In regular symmetric, hashing functions generate identical codes for data points irrespective of input sequence

In asymmetric, uses distinct hash functions for encoding and decoding process. This means hash codes produced for a specific data point can vary between the encoding (hashing) phase and decoding (retrieval) stage

MIPS = maximum inner product search. Given a large-scale data vector collection X with size n and a query point q

Objective: to earch for p from X with maximum inner product of $q^{\top}p $, or formally:

$p = \argmax_{x \in X} q^{\top} x  $

MIPS problem can be converted into the formalized l_{2} nearest neighbor search problem. As such, can construct two hash functions h(.) and z(.) to calculate the inner product of binary codes of the query sample and database samples. Can further defined the objective of MIPS as:

$p = argmax_{x \in X} h(q)^{\top}z(x)  $

subsequent research further injects the learning scheme into the *pairwise similarity-preserving hashing* as

$\|lS - h(Q)^{\top}z(X)  \|_{F}^{2} s.t. B \in \{-1, 1 \}^{l \times n}  $

where Q and X are all the query samples and the whole dataset samples

### 1.2.2 Ordinal-Preserving Hashing (OPH)

Crafted to retain the order-based relationship among data points

Unlike conventional methods that focus on generating compact and efficient data representations with piecewise or pairwise, OPH emphasizes on conserving the sequence or hierarchy inherent  in the original data points within hash codes. OPH aims to safeguard the ordinal connections amond data points, maintaining relationships such as "greater than" or "closer to" through the encoding process in the hash space

#### Definition 1.1 (Ordinal-Preserving Hashing)

Assume we have any triplet samples, i.e. (x_{i}, x_{j}, x_{k}), in which x_{j} and x_{k} are selected from the similarity neighbors of x_{i}. Let d(., .) denote the distance metric. For an optimal hash function \mathcal{H}, if d(x_{i}, x_{j}) \leq d(x_{i}, x_{k}), the learned binary codes under hash function \mathcal{H} satisfy d(H(x_{i}), H(x_{j})) \leq d(H(x_{i}), H(x_{k})), i.e., d(b_{i}, b_{j}) \leq d(b_{i}, b_{k})

Key use contexts where maintaining the relative sequence of data points is vital:

- ranking algorithms
- recommendation engines
- areas where hierarchy of similarities is key
  - information retrieval systems
  - collaborative filtering

### 1.2.3 Deep Collaborative Hashing

Leverages strengths of deep learning and a collaborative learning framework to create hash codes. Merges capacity of DL to learn autonomously.

Particularly useful in situations involving high-dimensional data. Shit scrot are there multimodal hashers?

The primary goal is to make hash codes that are conducive to effective similarity search and data retrieval.

#### Definition 1.2 (Deep Collaborative hashing)

Collaborative deep hashing means we employ multiple information derived from the same image modality to make comprehensive feature embedding and similarity aggregation, such that the generated unified hash codes can capture the principled common and discriminative components from multi-source information.

Specifica architecture, loss functinos, and training strategies vary across implementations and research studies

### 1.2.4 Trustworthy Deep Hashing

Focus on ensuring:

- reliability
- transparency
- security
- adherence to ethical standards

Trustworthiness encompasses:

- reliability
- integrity
- credibility

of a

- system
- technique
- procedure

The robustness of hashing methods and its ability to withstand adversarial attacks

#### Definition 1.3 (Adversarial Attack on Deep Hashing-Based Retrieval)

In deep hashing-based retrieval, given a benign query x with a label of y, the goal of non-targetted attack is to craft an adversarial example x', which could confuse the deep hashing model F to retrieve irrelevant samples to query x. In contrast, a targeted attack aims to mislead the depe hashing model into returning samples related to a given target label y_{t}. Moreover, the adversarial perturbation x' - x should be as small as possible to be imperceptible to the human eye.

#### Non-targetted attack

We prefer maximizing the hash code distance between the adversarial example and its semantically relevant samples, and simultaneously miniimizing the distance from irrelevant samples (rather than only the benign sample).

For a given clean image x, its corresponding adversarial example x' is developed by folling the objective under the L_{p} constraint:

$x' = \argmax_{x'} D_{H}(F(x'), b-{m}) s.t. \|x - x' \|_{p} \leq e$

where $\|. \|_{p} (p = 1, 2, \inf)  $ is $L_{p}$-norm that keeps the pixel difference between the adversarial example and the benign sample no more than e for the imperceptible property of adversarial perturbations. Due to $D_{H}(b_{1}, b_{2}) = (1/2) (K - b_{1}^{\top}b_{2})  $ is equivalent to

$x' = \argmax_{x'} - (1/K) b_{m}^{\top} \tanh(\alpha f_{\theta} (x')) s.t. \|x - x' \|_{p} \leq e $

where \tanh(.) is the activation fn, \alpha \in [0, 1] is the hyperparameter that controls $\tanh(\alpha f_{\theta} (x'))$ to approximate F(x')

#### Targetted Attack

The only difference between non-targetted and targetted is the objective fn. For a given benign sample x and a target label y_{t}, we first acquire the representative hash code of the target category b_{t} of y_{t} (the target class), and then the objective of the targetted attack can be defined as:

$x' = \argmin_{x'} D_{H}(F(x'), b_{t}) = \argmax_{x'} (1/K) b_{t}^{\top} 
tanh(\alpha f_{\theta} (x')) s.t. \|x - x' \|_{p \leq e} $

#### Adversarial Training for Defense

The ultimate pursuit of adversarial learning is to enhance the robustness of deep neural networks. Therefore, we jointly use the produced adversarial examples as augmented data and the original samples to optimize the deep hashing model for defense (i.e. adversarial training)

#### Definition 1.4 (Adversarial Training on Deep Hashing-Based Retrieval)

Similar to classification, adversarial training on deep hashing utilized both the benign samples $\{(x_{i}, y_{i})\}_{i-1}^{N}$ and corresponding adversarial versions $\{(x_{i}', y_{i})\}_{i-1}^{N} $ to re-optimize the parameter \theta of deep hashing models, and thereby the model could retrieve semantically relevant contents to the original label y_{i}, whether the input is a clean sample x_{i} or an adversarial sample x_{i}'.

Formally, we define the objective function of semantic aware adversarial training for deep hashing:

$min_{\theta} \mathbb{E}_{(x, y)~D} [\max_{x'} \mathcal{L}_{\alpha t}(x, x', b_{m}, \theta) ]  $

where D represents an underlying data distribution, and \theta is the network parameter. Herein, the b_{m} is the optimal representative code of the objective class

## 1.3 Evaluation Datasets and Protocols
