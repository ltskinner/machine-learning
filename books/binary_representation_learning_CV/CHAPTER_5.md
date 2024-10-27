# Chapter 5. Ordinal-Preserving Latent Graph Hashing

Current reserach on semantic-preserving hashing for similarity search primarily focuses on exploring semantic similarities among pointwise or pairwise samples in the visual space to generate discriminative hash codes. However, these approaches neglext the intrinsic latent features within the high-dimensional feature space, making it challenging to capture the underlying topological structure of data and resulting in suboptimal hash codes for image retrieval. This chapter introduces the Ordinal-preserving Latent Graph Hashing (OLGH) method, which formulates hash codes from the latent space and preserves the high-order locally topological structure of data in the learned hash codes. The approach introduces a triplet-constrained topology-preserving loss to unveil ordinal-inferred local features in binary representation learning, enabling the learning system to implicitly capture high-order similarities during the feature learning proces. Additionally, well-designed latent subspace learning is incorporated to acquire noise-free latent features based on sparse-constrained supervised learning, fully leveraging the latent under-explored characteristics of data in subspace construction. Latent ordinal graph hashing is formulated by jointly exploiting latent space construction and ordinal graph learning, with an efficient optimization algorithm developed to achieve the optimal solution. Comprehensive experiments on diverse datasets demonstrate the effectiveness and superiority of the proposed OLGH method compared to advanced learning-to-hash algorithms for fast image retrieval

## 5.1 Introduction

- data-independent hashing
- data-dependent hashing
  - in general yield more useful hash codes with shorter lengths compared to data-independent methods
  - two types:
    - unsupervised
      - aim to capture data disttibutions of distance-based similarities to establish relationship
    - (semi)-supervised
      - aim to convert high-dimensional features into compact hash codes
  - key distinction lie in their utilization of semantic supervision, such as labels, tags, or relevant feedback

three types of supervised, and how semantics are incorporated:

- pointwise supervised hashing
- pairwise supervised hashing
- triplet or listwise supervised hashing

These methods concentrate on how to utilize semantics for similarity preservation/approximation, overlooking the exploration of high-level **locality-similarity** in feature learning

supervised hashing methods in the current research primarlily concentrate on leveraging semantic knowledge to enhance the disciminant nature of hash codes

existing graph-based hashing approaches fall short in fully harnessing diverse semantics to product optimal and discriminative hash codes. Additionally, these graph-informed hashing methods prioritize sample level structures, making the computational complexity impractical for large-scale data management

Ordinal (Original ??)-Preserving Latent Graph Hashing (OLGH) adopts a unified learning framework that jointly addresses feature-level locality-similarity preservation and the acquisition of flexible semantic-preserving hash codes. Initially, a latent subspace is constructed to establish a semantic connection between high-dimensional visual features and available semantic labels. Furthermore, the method explicitly uncovers high-order local topological structures of data to capture ordinal-inferred local information, which is then integrated into the objective hash codes.

Contributions:

- OLGH (Ordinal here)
  - designed for swift similarity search on extensive datasets
  - inagural investigation into feature-level graph hashing, coupled wiht latent space construction, specifically tailored for large-scale image retrieval
- rather than rely on sample-level structures in graph learning, OLGH is formulated by delving into the high-order feature level geometric structure during the construction of triplet topological relationships. this ensure the effective preservation of high-order local topological structure of data within the resultant has codes
- establish a well-defined latent space to serve as a transitional realm. both explicit and implicit semantics seamlessly bridge the gap between low-level image representation and high-level semantic labels within this latent space
- an efficient learning algorithm

## 5.2 Ordinal (Original)-Preserving Latent Graph Hashing

### 5.2.1 Notations and Problem Definition

Forbenius norm: $\|X \|_{F} = \sqrt{\sum i = 1^{d} \sum_{j=1}^{n} x_{ij}^{2} } $

L_{21}-norm: $\|X\|_{21} = \sum i=1^{d}(\sum_{j=1}^{n} x_{ij}^{2})^{frac{1}{2}}  $

For supervised learning, we should also know the label or relevant feedback of each sample. Typically, each sample always falls into at least one class from the given categories, i.e., $L = [l_{1}, ..., l_{n}] \in \{0, 1 \}^{c \times n} $. Each label vector l_{i} is a one-hot vector to indicate the category of the corresponding sample x_{i}

Some pairwise-preserving methods employ pairwise similarity to build their models, and the pairwise similarity is defined based on the semantic labels or other feedback. In this chapter, the method is designed based on pointwise disciminative similarity preserving hashing, while the semantic labels are used for learning useful hash codes for efficient similarity search

The goal of supervised semantic-preserving hashing is to learn a set of hashing functions h(.;W) parameterized by W to encode the high-dimensional visual features into the low-dimensional hash codes B. Meanwhile, the discriminative semantics are well preserved in the objective hash codes during the space transformation from the visual space to the discrete hash space. This method first builds a latent subspace to bridge the semantic gap between the observed visual space and the objective Hamming space. Moreover, the high-order feature-level ordinal strucutre of the observed features is skillfully considered in the hash codes learning process.

#### Definition: Feature-level Ordinal-Preserving Hashing (FOPH)

Suppose we have a triplet or listwise of features x^{i}, ..., x^{i}, ..., x^{k}, ..., and x^{j}, ..., x^{k} are the neighbors of the sample x_{i}. The objective of FOPH is to preserve the triplet or listwise local relations into the learned binary representations. For simple description, we mainly consider the triplet situation, that is there are a triplet of features, x^{i}, x^{j}, x^{k}, and x^{j} and x^{k} are the two neighbors of the sample x_{i}. Moreover, we have their corresponding projection functions w^{i}, w^{j}, w^{k}. if the distance measurement is given by D(.,.), FOPH aims to preserve their collective high-order relations, which satisfies: f D(x^{i}, x^{j}) \leq D(x^{i}, x^{k}) and then if D(w^{i}, w^{j}) \leq D(w^{i}, w^{k})

Compared to the popular sample-level graph hashing algorithms, the given feature-level ordinal-preserving hashing framework has two prominent characteristics. On one hand, the proposed feature-level graph hashing can capture similarities between different feature dimensions, which can highlight the importance of feature-level samples. As such, the computational complexity and overhead can be greatly decreased compared to the sample-level grpah hashing. On the other hand, the proposed ordinal-preserving hashing model can capture the high-level similarities between features and preserve the intrinsic feature similarities in the objective hash codes, which results in high-quality hash codes for accurate similarity search

### 5.2.2 The General Framework of Graph-Based Discriminative Hashing

As we know, the essence of machine learning is to learn robust of discriminative representations for downstream tasks. In this chapter, we aim to generate discriminative hash codes by jointly considering locality preserving graph learning and semantic-preserving discriminative learning. Therefore, an effective framework of graph-based disciminative hashing is always formulated by the combination of multiple useful knowledge and representations, which are further preserved in the learned hash codes. As such, we can defined the general framework of graph-based disciminative hashing as the following formula:

$\min_{W, b_{i}} \sum_{i=1}^{n} \mathcal{H}(x_{i}, b_{i}, W) + \alpha \mathcal{R}(.) + \beta \mathcal{G} (.) s.t. b_{i} \in {-1, 1}  $

where \alpha and \beta are the weighting parameters to balance the importance of different terms. 

$\mathcal{H}(x_{i}, b_{i}, W)$  is the disciminative hash space learning fn, which transforms the high dimensional visual feature x_{i} into the compact hash code b_{i}

$\alpha \mathcal{R}(.)$ is a regularization term, which is defined to make the wholem odel generate stable solutions

$\beta \mathcal{G} (.)$ is a graph-based learning module to capture the locality knowledge between samples for learning similarity preserving hash codes

OLGH has two main components:

- the discriminative latent hashing space construction
- the feature-level ordinal-correlation preservation

### 5.2.3 Discriminative Latent Hashing Space Construction

For discriminative latent hashing space construction, we derive it from the "least linear regression formulation", which is defined as

$\min_{W, b_{i}} \sum_{i=1}^{n} \|W^{\top} x_{i} - L \|_{F}^{2} + \alpha \mathcal{R}(W)  $

Where R(W) is the regularization term, such as F-norm, l1 norm, etc. Here, we use l_{21} norm to perform a robust feature selection role in space transformation. As such, we have

$\min_{W, b_{i}} \sum_{i=1}^{n} \|W^{\top} x_{i} - L \|_{F}^{2} + \alpha \|W\|_{21} $

However, such a learning model cannot build the relationship between robust linear regression and binary hash code learning. Therefore, we propose to formulate a well-designed latent subspace to skillfully connnect the visual feature space, semantic label space, and the constructed latent space. Reformlate above as

$\min_{W,b_{i}} \sum_{i=1}^{n} (\|W^{\top}x_{i} - b_{i}\|_{F}^{2} + \gamma \|b_{i} - l_{i}\|_{F}^{2}   ) + \alpha \| W\|_{21} s.t. b_{i} \in \{-1, 1\}   $

where B is the binary hidden feature space between visual feature space and discrete label space. However, such a learning scheme is ill-posed because the latent feature space B lacks enough flexibility to capture the underlying semantic knowledge. Therefore, we introduce the latent space learning module to enable flexible and effective latent space construction. Specifically, we use a dictionary learning framework to model the latent variable B, and the linear transformation matrix A is employed to build up the correlation between semantic labels and latent space. That is,

$\min_{W, B, A} \|W^{\top} X - B\|_{F}^{2} + \gamma \|B - AL\|_{F}^{2} + \alpha \|W\|_{21} s.t. B \in \{-1, 1 \}^{r \times n}, \|a_{i}\|_{2} \leq 1  $

where a_{i} is the i-th column vector of the matrix A, and r is the length of the objective hash codes. it can be inferred that the obtained latent binary embeddings can be viewed as the linear combinations of joint semantic labels and visual-semantic features. In this way, the generated binary codes can well bridge the low-level visual features and the high-level semantic labels, which can collectively capture the visual- and label-level semantic characteristics during space communication

### 5.2.4 Feature-Level Ordinal-Correlation Preservation

The above disciminative latent hash code learning only considers how to obtain useful binary codes and semantic preservation during space transfer. However, the global knowledge in semantic combination and latent space encoding cannot fully exploit the fine-grained semantics, and the locality-level information is ignored inbinary code learning, yielding degraded binary representation. The feasible solution to this limitation is to build a graph-based learning scheme to capture the locality knowledge between features

In contrast, the popular mechanism in locality-similarity preserving learning [25] is to leveerage the manifold structure to capture geometric information in the sample-level similarities. Notably, the biggest advantage of binary code learning is efficient and compact representation learning to approach the applications of massive big data. However, when using the sample-level similarity-preserving mechanism, we have to build and compute on the n \times n matrix, leading to O(n^{2}) unaffordable computation complexity. There fore, *for the first time*, propose to leverage the feature-level ordinal correlations to preserve the high-order similarities in binary code learning

Based on the definitoin of feature-level ordinal-preserving hashing and the well-known rearrangement inequality, the ordinal similarity preservation can be transofmred to find a preferable triplet {w^{i}, w^{j}, w^{k}} which can assure the maximum inner product of $D(x^{i}, x^{j}) - D(x^{i}, x^{k})  $ and $D(w^{i}, w^{j}) - D(w^{i}, w^{k})  $. As such, identifying an appropriate projection matrix W is equivalent to solving the following problem:

$\max_{W} \sum_{i=1}^{d} \sum_{j \in N_{i}} \sum_{k\in N_{i}} s_{jk}^{i} [D(w^{i}, w^{j}) - D(w^{i}, w^{k})] $

where S^{i} is an antisymmetric matrix whose (j, k)-th entry is defined as $s_{jk}^{i} = D(x^{i}, x^{j}) - D(x^{i}, x^{k})  $, and N_{i} denotes the index of the k neareset neighbors for x^{i}. We defined a new weighting matrix $P \in \mathbb{R}^{d \times d}   $ as:

$p_{ij} \overset{\Delta}{=} \sum_{k \in N_{i}} s_{kj}^{i}, j \in N_{i} $

$p_{ij} \overset{\Delta}{=} 0, j \notin N_{i} $

According to the inference in [26], we easily obtain the equivalent reformulateion of Eq. 5.6, and the locality ordinal-correlation graph is obtained by

$G(.) = \min_{W} \sum_{i=1}^{d} \sum_{j=1}^{d} p_{ij} D(w^{i}, w^{j})  $

which can well capture the triplet neighborhood relations and the high-order ordinal knowledge from the efficient feature-level perspective

### 5.2.5 Final Objective Funcion

To further fully exploit the latent binary representation learning and ordinal-correlation preservation, we formaulate the final collective learning objective OLGH functions by combiningin Eqs. 5.5 and 5.8, i.e.:

$\min_{W, B, A} \|W^{\top}X - B\|_{F}^{2} + \gamma \|B - AL\|_{F}^{2} + \alpha \|W\|_{21} + \beta \sum_{i=1}^{d} \sum_{i=1}^{d} p_{ij} D(w^{i}, w^{j}) s.t. B \in \{-1, 1 \}^{r \times n}, \|a_{ij}\|_{2} \leq 1  $

From the above objective function, we can see that the first and second terms build a new latent space to store the binary code B, which can well bridge and absorb the knowledge between the visual features and the semantic supervision. Moreover, the generated binary codes are obtained from the combination of the semantic labels, as well as the projected features from the observed visual information. The third term can ensure a stable yet robust solution on the projection W. The final term can well preserve the high-order ordinal similarities on the feature-level relations, and the efficiency is also preserved in the learning step due to the avoidance of the n \times n matrix computation. For simplicity, we adopt the commonly used least square distance to establish the distance between any two features in the graph constraint term

## 5.3 Learning Algorithm

It is obvious that the problem 5.9 is not convex to the variables W, A, B simultenously, due to the binary and inequality constraints. However, we can obtain the closed-form solution for each of them separately. Therefore, we adopt the alternative learning algorithm to solve the resulting optimization problem, i.e., optimizing one particular variable when fixing the reamining. These steps are alternated

### W-Step

the degenerated problem:

$\min_{W} \|W^{\top}X - B\|_{F}^{2} + \alpha \|W\|_{21} + \beta \sum_{i=1}^{d} \sum_{i=1}^{d} p_{ij} \|w^{i} - w^{j}\| = \|W^{\top} X - B \|_{F}^{2} + \alpha \|W\|_{21} + \beta tr(W^{\top} \Sigma W)   $

where $\Sigma = D - \frac{P + P^{\top}}{2}  $ is the laplacian matrix, and D is a diagonal degree matrix, the (i,i)-th element of which is defined as $d_{ii} = \sum_{j=1}^{d} \frac{p_{ij} + p_{ji}}{2}  $. tr(.) is the trace norm. Moreover, due to the non-convex non-smooth problem of the l_{21}-norm regularization, we take the iteratively reweighted strategy [27] to solve it, which means, we can rewrite \|W\|_{21} as:

$\|W\|_{21} = \sum_{i=1}^{d} \|w^{i}\|_{2} = tr(W^{\top} QW)  $

where Q is a diagonal matrix, and the diagonal element $q_{ii} = \frac{1}{2\|w^{i}\|_{2}}  $. To avoid encountering infinity problem when computing q_{ii}, we usually use $q_{ii} = \frac{1}{\sqrt{\|w^{i}\|_{2} + \eta}} $ and \eta is a very small instance, e.g., e^{-5}. As such, problem 5.10 can be rewritten as

$\min_{W} \|W^{\top}X - B \|_{F}^{2} + \alpha tr(W^{\top}QW) + \beta tr(W^{\top}\Sigma W)  $

By taking the partial derivation wrt W and setting it to zero, we have the (closed form?) solution of W, i.e.

$W = (XX^{\top} + \alpha Q + \beta \Sigma)^{-1} XB  $

### A-Step

reduced problem:

$\min_{A} \|B-AL\|_{F}^{2} s.t. \|a_{i}\|_{2} \leq 1  $

This problem can be optimized by using the Legrange dual, and the analytical solution for A is given by

$A = (BL^{\top})(LL^{\top} + \Lambda)^{-1}  $

where \Lambda is a diagonal matrix that is formulated by all the Lagrange dual parameters

### B-Step

fixing all other variables irrelevant to B, we have the following optimization problem:

$\min_{B} \|W^{\top}X - B\|_{F}^{2} + \gamma \|B - AL\|_{F}^{2} s.t. B \in \{-1, 1 \}^{r \times n}  $

It is easy to obtain that tr(BB^{\top}) = rn is a constant. With some simple computations, we can derive the closed-form solution of B, i.e.,

$B = sign(W^{\top}X + \gamma AL)  $

The learning algorithm iteratively updates each variable until convergence to local min or max iters

## 5.4 Out-of-Example Extension and Analysis

### 5.4.1 Out-of-Example Extension

Inspired by the effectiveness of two-step hashing models, we further learn a series of hash functions based on simple feature transformation and then generate the hash codes for newly coming samples. Specifically, we simply take the linear regression formulation to learn the hash fn, i.e.,

$\min_{\hat{W}} \|\hat{W}^{\top}X - B \|_{F}^{2} + \|\hat{W}\|_{F}^{2}  $

the solution of which is easily achieved by

$\hat{W} = (XX^{\top} + I)^{-1} XB^{\top}  $

For any new sample feature vector x_{q}, we directly obtain the binary code of x_{q} by using $b_{q} = sign(\hat{W}^{\top} x_{q})  $

### 5.4.2 Convergence Analysis

The whole objective function is denoted as $\mathcal{L}(W^{t}, A^{t}, B^{t})  $ for the t-th iteration

#### Lemma 5.1

For any two non-zero constraints a and b, the given inequality is always satisfied:

$\sqrt{a} - \frac{a}{2\sqrt{b}} \leq \sqrt{b} - \frac{b}{2\sqrt{b}}  $

#### Lemma 5.2

If there are r non-zero vectors v_{t} and v_{t+1}, the following inequality is also always satisfied

$\sqrt{\sum_{i}^{d} \|v_{t+1}^{i}\|_{2}} - \frac{\sum_{i}^{d} \|v_{t+1}^{i}\|_{2}^{2}}{2\sqrt{\sum_{i}^{d} \|v_{t}^{i}\|_{2}}} \leq \sqrt{\sum_{i}^{d} \|v_{t}^{i}\|_{2}} - \frac{\sum_{i}^{d} \|v_{t}^{i}\|_{2}^{2}}{2\sqrt{\sum_{i}^{d} \|v_{t}^{i}\|_{2}}}  $

#### Lemma 5.3

The iteration approach for optimizing the variable W can monotonically decrease the value of the subobjective function wrt W in each iteration

#### Theorum 5.1

The proposed learning algorithm iteratively decreases the value of the objective fn in Eq. 5.10 with the increasing number of iterations and converges to a local optimal point

### 5.4.3 Computational Analysis

The computational complexity of the learning algorithm in Algorithm 5 mainly lies in the second to the fifth step. For the optimization of W, the learning algorithm consumes the complexity of O(d^{2}rn). For the computation of A, the proposed optimization algorithm costs about O(crn + c^{2}n). Moreover, the calculation of B needs O(drn + rcn) complexity. Computing the value of Q consumes O(r). Therefore, the computaitonal complexity of this whole learning algorithm is O((d^{2}r + 2cr + c^{2} + dr)n), which is linear to the number of training samples, i.e. n

## 5.5 Experimental Results

### 5.5.1 Experimental Configuration

- CIFAR-10
  - extract a GIST feature vector with 512 dim for each experiment
- Caltech-256
  - The CNNs pretrained on ImageNet datset are used to extract the deep features, which are obtained from the last fully connected layer of the NN
- ESP-GAME
  - multi-label dataset
  - 512 GIST feature vector
- MNIST
- ImageNet

### 5.5.2 Implementation Details

- unsupervised hashing algorithms
  - ITQ
  - SGH
  - OCH
- supervised discrete hashing methods
  - KSH
  - ITQ-CCA
  - LFH
  - SDH
  - NSH
  - FSDH
  - BSH
  - FDCH
  - R2SDH

Three hyperparameters for OLGH

- \alpha
- \beta
- \gamma

utilize the well-known cross-validate strategy to find the optimal parameters on each dataset and achieve the best retreival performance

### 5.5.3 Evaluation Protocols

- mean average precision

### 5.5.4 Comparison Results

#### 5.5.4.1 Experimental Comparison Results on CIFAR-10

Observations:

- OLGH can achieve the highest retreval performance
- supervised methods, such as SDH, NSH, FSSH, FDCH, and R2SDH can obtain better retrieval results than unsupervised hashing
- pairwise similarity preservation methods, such as KSH, and LFH can achieve very competitive results

#### 5.5.4.2 Experimental Comparison Results on Caltech-256

Has more classes than CIFAR-10, which can greatly challenge the robustness of the similarity search systems

#### 5.5.4.3 Experimental Comparison Results on ESP-GAME

the multi-label dataset

#### 5.5.4.4 Experimental Comparison Results on MNIST

#### 5.5.4.5 Experimental Comparison Results on ImageNet

### 5.5.5 Training Efficiency Study

Most learning to hash algorithms includes two steps, i.e. hash function learning and searching. Generally, the searching step of the given hashing algorithms is perfomed on generated binary codes, while the similarity comparison is based on the very efficient Hamming distance, O(1). As a result, most hashing algorithms have very similar querying time when it comes to new query samples

The main difference lies in the training efficiencly of different algorithms

## 5.6 Conclusion

In this chapter, we introduced a novel ordinal-preserving latent graph hashing (OLGH) method, which jointly considers high-order feature-level ordinal similarity preservation and the discriminative hash codes learning for fast similarity search. Typically, the well-designed OLGH was derived from the latent space construction; meanwhile, the high-order locally topological structure of data was preserved in the learned hash codes. Instead of using the sample-level locality-preserving strategy, we proposed to caputre the feature-level ones, which could clearly avoid using the n \times n computation in framework modeling. Moreover, the triplet ordinal topological similarities were used in the similarity measurement. Moreover, the flexibility of the learned latedn hash codes was obtained by a combination of the semantic labels and the row-sparsity-induced robust feature learning. An efficient learning algorithm with guaraneed convergence was designed to tackel the resulting optimization problem. We conducted extensive experiments to demonstrate the effectiveness of the proposed method on large-scale image search. From these results, we could conclude that our OLGH could achieve sota retrieval performance when compared to some advanced learning-to-hash methods.
