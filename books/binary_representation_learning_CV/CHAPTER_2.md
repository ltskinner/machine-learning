# Chapter 2. Scalable Supervised Asymmetric Hashing

Learning compact hash codes is widely employed for rapid similarity search, capitalizing on reduced storage requirements and efficient query speeds. However, mastering discriminative binary codes that perfectly preserve full-pairwise similarities in high-dimensional real-valued features remains a challenging task for ensuring optimal performance. To tackle this challenge, this cahpter introduces a novel method, **Scalable Supervised Asymmetric Hashing (SSAH)**.

SSAH adeptly approximates the full-pairwise similarity matrix by leveraging the maximum asymmetric inner product of two distinct non-binary embeddings. To comprehensively exploit the semantic information, the method simultaneously considers supervised label information and refined latent feature embeddings to construct high-quality hashing functions, enhancing the discriminative capability of learned binary codes. Specifically, SSAH learns two distinctibe hashing functions by minimizing regression loss for semantic label alignment and encoding loss for refined latent features. Notably, instead of utilizing only partial similarity correlations, SSAH directly employs the full-pairwise similarity matrix to prevent information loss and performance degradation. Its optimization phase adeptly manages the cumbersome computation complexity of the n \times n matrix. Additionally, an efficient alternating optimization scheme with guaranteed convergence is designed to address the resulting discrete optimization problem. Experimental results on diverse benchmark datasets highlight the superiority of SSAH over recently proposed hashing algorithms

## 2.1 Introduction

Two categories of current methods:

- data independent
  - suffer from information loss and coding reduncancy, typically requiring longer hash codes or multiple tables
- data dependent (learning to hash)
  - typically result in shorter binary codes
  - two groups:
    - `geometry-preserving`
      - unsupervised hashing
    - `semantics preserving`
      - typically outperforms unsupervised methods
      - time consuming discrete optimization, and limited scalability to large-scale data
    
Current supervised methods lack flexibility and adaptability required for efficient large-scale image retrieval tasks.

- methods overlook the *essential role* of preserving pairwise similarities, and only use a fraction of available semantic data
- imposing discrete constrants on learned hash codes can transform the problem into an NP-hard mixed-integer optimization challenge
  - attempting to simplify by shifting to a continous problem often leads to a drop in retrieval performance
- iterative discrete optimization not computationally practical for big data
- most supervised methods rely on symmetric hashing
  - this is time intensive

This chapter focuses on developing discriminative binary representations through discrete optimization.

SSAH core concept: employs an asymmetric learning structure with dual-stream real-valued codes, instead of binary ones

This is designed to better approximate the full similarity matrix, enhancing the preservation of pairwise similarities among data points. To achieve this, construct two distinct non-binary feature embeddings:

- semantic regression embedding
  - uncover semantic connections bw binary codes, pairwise wimilarities, supervised label information
  - fully leverages pairwise similarities across all data without any relaxation
- refined latent factor embedding
  - aims to emphasize intricate hidden information within the unique components of original data

Primary contributions:

- 1. the hash routine
- 2. cohesively integrated semantic embeddings with the refined latent factor construction, which significantly reduces the complex computation associated with the full n \times n similarity affinity matrix, allowing the entire dataset to be effectively utilized in the efficient learning of hash codes
- 3. an inteicately crafted optimization algorithm devised to effectively tackle the discrete programming challenge, rendering it highly scalable and capable of handling large-scale datasets with ease
- 4. the findings, that its sota on large-scale datasets

## 2.2 Scalable Supervised Asymmetric Hashing

- Matrices are bold uppercase
- vectors are bold lowercase

The Frobenius norm of matrix X is $\|X \|_{F}^{2} = tr(X^{\top} X) = tr(XX^{\top}) $ where tr(.) denotes the trace operator. The l_{2,p}- norm of matrix is denoted as $\|X  \|_{2,p}^{p}  $ which is defined as $\|X \|_{2,p}^{p} = \sum_{i=1}^{d} \|x^{i} \|_{2}^{p} \sum_{i=1}^{d} (\sum_{j=1}^{n} x_{ij}^{2} )^{p/2}  $ where x^{i} is the i-th row of matrix X . X^{\top} represents the transposed matrix of X and I indicates an identity matrix

### 2.2.1 Problem Definition

This chpater focuses on how to effectively preserve semantic information in binary codes

Suppose that the available dataset $X = [x_{1}, ..., x_{n}] \in \mathbb{R}^{d \times n}  $ includes n images samples, and each image is represented as a d-dimensional feature vector $x_{i} \in \mathbb{R}^{d} $. The corresponding labels are matrix $Y = [y_{1}, ..., y_{n}] \in {0, 1}^{c \times n} $ where c is the number of classes and y_{ji} = 1 if sample x_{i} belongs to the j-th class. The goal of learning to hash is to construct a set of hashing functions $\{h(x)\}_{k=1}^{l} \in H  $ to project the original real-values features from the Euclidian space onto a low-dimensional Hamming space {-1, 1}, e.g., $H(x_{i}) = [h_{1}(x_{i}), ..., h_{l}(x_{i})] \in {-1, 1}^{l}  $.

Meanwhile the similarity correlations of data points are preserved in the encoding phase. We denote the learned binary codes as: $B = [b_{1}, ..., b_{n}] \in {-1, 1}^{l \tims n}  $ for dataset X, where b_{i} \in {-1, 1}^{l} represents the code for the i-th data point x_{i}.

It is known that hashing code similarity can be transformed into the inner product of hash codes, which indicates the opposite of the Hamming distance

We consider the commonly used objective that minimizes the quantization error between the inner products of binary codes and semantic similarity, that is:

$\|lS - B^{\top}B  \|_{F}^{2} s.t. B \in \{-1, 1 \}^{l \times n}  $

where $S \in \{0, 1 \}^{m \times m}  $ is the semantic pairwise similarity matrix defined by the supervised labels, and m is the number of selected data points. Specifically, S_{ij} = 1 means that the i-th and j-th data points are semantically similar (neighbors measured by a distance matric or sharing at least one label); otherwise S_{ij} = 0 for dissimilar pairs (non-neighbors or no-shared labels)

Model primarily defined in kernel supervised hashing (KSH), which has 3 significant shortcomings:

- 1. relies on symmetric binary inner product approximation, coupled with contuous relaxed optimization, which can lead to a substantial cumulative quantization error
  - this is particularly problemmatic for large datasets and results in inefficient binary codes
- 2. KSH only preserves the similarities between data points within a sampled small subset, leading to unavoidable information loss and the generation of suboptimal hash functions
- 3. the model underutilizes both informative supervised labels and the inherent data information, hindering effective binary code learning

Notations

| Symbol | Description |
| - | - |
| $X \in \mathbb{R}^{d \times n} $ | The data matrix |
| $x_{i} \in \mathbb{R}^{d \times nl} $ | The i-th data point |
| $Y \in \mathbb{R}^{c \times n} $ | The class labels matrix |
| $y_{i} \in \mathbb{R}^{d \times n} $ | The label vector of the i-th data point |
| $ B \in \{-1, 1 \}^{l \times n} $ | The binary hash codes |
| $ b_{i} \in \{-1, 1 \}^{l \times n} $ | The hash code for i-th data point of x_{i} |
| $ S \in \{0, 1 \}^{n \times n} $ | The semantic pairwise similarity matrix |
| $W \in \mathbb{R}^{c \times l} $ | The projection matrix for the hash code |
| $U \in \mathbb{R}^{d \times l} $ | The basis matrix of the latent space |
| $V \in \mathbb{R}^{l \times n} $ | The latent factor matrix |
| $R \in \mathbb{R}^{l \times l} $ | The rotation matrix |
| n | The number of training data instances |
| d | The dimensionality of visual feature |
| c | The number of classes |
| l | The hash code length |

### 2.2.2 The Proposed Scalable Supervised Asymmetric Hashing

SSAH is designed to address these issues by preserving global pairwaise similarity through semantic alignment, and latent factorization embbedding. Core concept is to asymmetrically approximate global pairwise similarity, incorporating supervised label information and shareable latent feature representations within a unified learning framework

#### Robust Semantic Alignment

To preserve full pair-wise similarities on the whole n data points, the size of full similarity matrix S should be n \times n, which is a bottleneck

Therefore, construct the first asymmetric hash function by employing the linear semantic regression denoted by $h(Y) = sgn(W^{top} Y)  $, where W \in R^{x \times l} is a projection matrix that aligns the semantic matrix (of classes) onto an l-dimensional feature space, and sgn(.) denotes the element-wise indicator operator. Incorporating semantic information in representation learning can enhance the discriminative capabilities of the learned features

$min_{W,B} \|lS - sgn(W^{T} Y)^{T} B   \|_{F}^{2} s.t. B \in \{-1, 1 \}^{l \times n}  $

It is demonstrated that using real-valued embeddings can provide more accurate similarity approximation. We empirically relax the sign function with its signed magnitude and minimize the quantization error between binary codes and semantic alignment. Furthermore, the label information for large-scale datasets often contains bad labels, which can jeapordize the model. Therefore, a robust semantic alignment model for binary code learning is:

$min_{W,B} \|lS - sgn(W^{T} Y)^{T} B\|_{F}^{2}  \labmda_{1} \|W^{\top}Y - B  \|_{2,p}^{p} s.t. B \in \{-1, 1 \}^{l \times n}  $

where $\|.\|_{2,p}^{p}$ is the l_{2p}-norm. In this way, the similarities are measured by the inner product between the real-valued features W^{top}Y and the binary codes B. Based on such a simple learning strategy, both the computation cost and memory overhead can be clearly reduced.

The similarity matrix S and the label matrix Y are always tied together as a whole (A = SY^{T} \in \mathbb{R}^{n \times c}) in all optimization procedures, which can be pre-computed once outside of the main iterations. In this way, the computation cose may directly increase from O(n^{2}) to O(nc), where c << n

#### Robust Latent Factor Embedding

similarity approximation still depends on the generated binary codes, potentially leading to high-similarity fitting errors on diverse datasets. Real valued features are also ignored, despite being supervised. To address, another hashing function g(X) is introduced. This is constructed from the latent factors embedded int he data to replace the binary codes matrix

Goal is to identify a low-rank matrix that can unveil the latent information concealed within the data while filtering out redundant noise. Initially, we consider approximating the original zero-centered data X in the context of the Euclidian space, solve this:

$\min_{Z} \|X - Z  \|_{F}^{2}  s.t. rank(Z) = l $

where Z are the low-rank components of X. Based on the full rank matrix decomposition, any matrix Z \in R^{d \times n} with rank l (l < min(d, n)) can be decomposed by Z = UV, where U \in R^{dxl}, and V \in R^{l \times n}. A high-dimensional yet low-rank matrix can be approximately represented by a linear combination of basis vectors. The effective latent semantic embeddings of the original features can be obtained if the basis vectors can capture the underlying structure of data.

$\min_{U,V} \|X - UV  \|_{F}^{2} s.t. U^{\top}U = I  $

where the orthogonal constraint of U makes each column of U independent of each other. Interpret U as the basis matrix, and the encoding matrix V viewed as the latent representation of X. Notably, the orthogonal constraint on U enforces the generated bases vectors to be separated, such that th elatent representation V can faithfully preserve the similarity structure of samples in X

- the squared loss fn is susceptible to noise or outliers
  - maginfies quantization error though squaring
  - may render latent features highly reponsive to outliers int he data
- minimizing quantization loss bw latent semantic rep and discrete binary codes are essential for effectively conveying the neighborhood structures of the latent semantic space to the target Hamming space

To address, intro a resilient latent factor embedding approach that incorporates orthogonal transformation:

$\min_{U,V} \|X - UV  \|_{2,p}^{p} + \mu \|B - RV \|_{F}^{2}  s.t. U^{\top}U = I, R^{\top}R = I $

which guarantees the minimal quantization error to transfer the locality correlations of real-valued features to discrete codes. Instead of using l2-norm euclidian loss, it is used to regularize the reconstruction loss for improving robustness of the learned latent factor embeddings. The orthogonal transformation matrix R is used to scale and rotate the real-valued latent feature from the continuous domain to the discrete domain. In this way, the binar quantization loss is greatly reduced such that latent embeddings from the same class have much higher probabilities of being encoded into similar hash codes

The second binary codes B can be replaced by B = RV for real valued similarity optimization. The SSAH objective function is:

$\min_{W, U, V, R, B} \|lS - (W^{\top} Y)^{T} RV \|_{F}^{2}  + \lambda_{1} \|W^{\top}Y  \|_{2p}^{p} + \lambda_{2} (\|X - UV  \|_{2p}^{p} + \mu \|B - RV  \|_{F}^{2}) s.t. B \in \{-1, 1 \}^{l \times n}, U^{\top}U = I, R^{\top}R = I   $

where \lambda_{1}, \lambda_{2}, and \mu are penalty parameters to balance the importance of different components.

To handle linearly inseparable data, nonlinear kernel features are employed to improve the represetnation capabilities. Assume $\phi : \mathbb{R}^{d} \rightarrow \mathcal{M} \in \mathbb{R}^{m}  $ be a kernel fn from the original Euclidian space to the kernel space, where M is a Reproducing Kernel Hilbert Space (RKHS) with kernel mappings \phi(x). Inspired by nonlinear anchor feature embedding for effective hash function learning, we preprocess each data point x_{i} by

$\phi(x_{i}) = [\exp(- (\|x_{i} - a_{q} \|^{2})/(2\sigma)), ..., \exp(- (\|x_{i} - a_{m} \|^{2})/(2\sigma))  ]^{T}  $

where $\{a_{i} \}_{i=1}^{m}  $ are m anchor features randomly sampled from the original dataset. Therefore, the final objective for SSAH is given by

$\min_{W, U, V, R, B} \|lS - (W^{\top}Y)^{\top} RV  \|_{F}^{2} + \lambda_{1}\|W^{\top}Y - B \|_{2,p}^{p} + \lambda_{2}(\|\phi(X) - UV \|_{2,p}^{p} + \my \|B - RV  \|_{F}^{2} )  s.t. B \in \{-1, 1 \}^{l \times n}, U^{top}U = I, R^{\top}R = I  $

### 2.2.3 Optimization of SSAH

The above joint optimization probblem is a mixed binary optimization problem, which is non-convex with W, U, V, R and B together. A well-designed alternating optimization algorithm is developed to iteratively solve the resulting problems wrt one variable when fixing others. l2p norm is intractable so need to reformulate as:

min equation in same pattern

where E = W^{\top}Y - B and Q = \phi(X) - UV. Both D \in R^{l \times l} and K \in R^{d \times d} are diagonal matrices, and their i-th diagnoal elements are defined as $d_{ii} = p \ (2 \|e^{i} \|_{2}^{p}) $ and $ k_{ii} = p \ (2 \|q^{i} \|_{2}^{p})  $, where e^{i} and q^{i} are the i-th row of E and Q

#### W-Step

When fixing U, V, R, and B, we require to solve the follwing problem wrt W

loss = min_{W} XXXX

ok some equation soup here

A and C are fixed variables and can be pre-computed only once outside of the main iteration for saving computation time and storage load

#### U-Step

when fixing W, V, R, and B, wrt U is degenerated to

XXX

suppose XDV^{\top} is full rank, then there is a closed form solution

#### V-Step

when fixing W, U, R, and B, the subproblem wrt V is given as a minimization problem which also has a closed form solution

#### R-Step

When fixing W, U, V, R, problem wrt R is simplified to a loss problem

Cannot directly obtain its closed-form solution due to the orthogonality constraint on R. Thus, we optimize it by employing the gradient flow which provides a general feasible solution to optimize the orthogonal problem in an iterative learning strategy. Specifically, if we deonte R_{t} as its optimization result in the t-iteration, we can obtain a better solution of R_{t+1} by using the Cayley Transformation updating step

The convergence of this learning algorithm is assured, following the proof strategy. Due to the inclusion of the quadratic approximation loss for similarity preservation in the optimization problem, we refer to this problem as SSAH-Q. This is a time-sensitive operation

Inspired from the efficacy of maximum inner product search, we introduce an alternative fomulation of the problem designed to enhance SSAHs performance. Instead of utilizing the quadratic loss for similarity, directly maximize both the similarity matrix an dinnter product of asymmetrid real-valued embeddings

Discarding the quadratic term works better than the original model inpractics. This is a classic Orthogonal Procrustes Problem (OPP), and can generate a stable solution by using simple SVD

#### B-Step

When fixing W, U, V, R, the subproblem wrt B is (a messy long equation)

rewrite as maximization problem, with a closed-form solutoin

#### back to main

In this way, the overall optimization algorithm is to alternatively optimize variables, i.e., W, U, V, R, and B until achieving convergence or reaching the maximum iterations.

### 2.2.4 Out-of-Sample Extension

In the query phase, the goal is to find the most similar items from the database, and we need to first compute the binary code b of a query x_{tt} which is outside of the training data. Motivated by the success of two-step learning, we first find the optimal projection matrix \Psi that transforms the high-dimensional nonlinear data points \phi(X) into the binary hashing codes B

For simplicity, we directly use the linear regression model to train \Psi over the training dataset \phi(X). The regularized linear regression model is formulated as

$\min_{\Psi} \|B - \Psi^{\top} \phi(X) \|_{F}^{2} + \lambda \| \Psi  \|_{F}^{2}  $

where \lambda is the regularization parameter
