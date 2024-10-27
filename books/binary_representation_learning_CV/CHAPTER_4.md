# Chapter 4. Probability Ordinal-Preserving Semantic Hashing

Existing semantic hashing methods primarily conceptrate on preserving piecewise class information or pairwise correlations in learned binary codes, often neglecting the mutual triplet-letvel ordinal structure cruicial for similarity preservation. This chapter introduces a groundbreaking approach - the Probability Ordinal-preserving Semantic Hashing (POSH) framework - pioneering ordinal-preserving hashing under a non-parametric Bayesian theory. The framework derives the entire learning process for ordinal similarity-preserving hashing based on maximum posterior estimation. Probabilistic ordinal similarity preservation, probabilistic quantization function, and probabilistic semantic-preserving function are seamlessly integrated into a unified learning framework. Notably, the proposed triplet ordering correlation preservation scheme enhances the interpretatbility of learned hash codes using an anchor-induced asymmetric graph learning model. Additionally, a sparsity-guided selective quantization function minimizes space transformation losses, while a regressive semantic function enhances the flexibility of formulated semantics in hash code learning. The joint learning objective concurrently preserves the ordianl locality of original data and explroes semantics for discriminative hash code generation. An efficient alternating optimization algorithm, backed by a strictly proof convergence guarantee, is developed to solve the resulting objective problem. Extensive experiments across several large-scale datasets validate the superior performance of the proposed POSH framework compared to sota hashing-based retrieval methods

## 4.1 Introduction

- unsupervised hashing methods
- supervised hashing methods
  - divided on how they formalize semantics:
    - piecewise semantic-preserving hashing
      - employs semantic labels as ground truth for learning disciminative hash codes
      - typically employs semantic labels for regression or classification purposes, but often overlooks the correlations between samples
    - pairwise semantic-preserving hashing
      - focuses on preserving the pairwise (or triplet and listwise) relationships between samples
      - effectively captures the pairwise correlations
      - fallsh short in leveraging discriminative category information

A significant challenge remains in further uncovering the intricate ordinal triplet correlations among samples while simultaneously constructing flexible semantic embeddings to generate effective hash codes

The imposition of a discrete constraint on hash codes transforms the objective functions into NP-hard mixed-integer problems. A common approach to tackle this is through a two-step relaxation algorithm, i.e. initially discarding the discrete constraint and subsequently applying a hard thresholding operator. However, this learning strategy is often suboptimal, leading to the generation of low-quality hash codes due to substantial accumulated quantization errors.

The disparity in distribution between training and query samples is an aspect that existing hashing methods have not adequately addressed

Probability Ordinal-preserving Semantic Hashing (POSH) uniquely blends the preservation of ordinal correlations in triplet similarity with adaptable semantic reconstruction, all within a cohesive Bayesian inference framework. Inspired by ordinal subspace learning. The probabilistic component for ordinal similarity preservation is designed to optimize the posterior estimation within our precisely formulated ordinal preserving hashing model. This ensures significant preservation of both sample correlations and their distribution variances in the hash codes derived from POSH. In addition, the probabilistic quantization function aims to minimize the discrepancy between the learned features and the targeted hash codes. Furthermore, the probabilistic semantic-preserving function is tasked with developing sparsity-enhanced regressive semantics, thereby augmenting the adaptability of the reinterpreted semantic data.

**A pivotal aspect of the approach is the cevelopment of an effective alternating optimization algorithm.** This algorithm addresses the complex objective fn, streamlining the optimization of binary codes in a singular-step process.

Contributions:

- 1. POSH
  - combines elements like probabilistic ordinal similarity preservation, a probabilistic quantization fn, and a probabilistic semantic-preserving fn
  - this is super novel, the innagural effort
- 2. the concept of ordinal-preserving hashing
- 3. probabilistic quantization fn to diminish losses incurred during space transformation
  - concurrently, design the prbabilistic semantic-preserving fn with a foundation in regressive semantics, which enhances the adaptability of semantics in the process of hash code learning
- 4. development of highly efficient discrete learning algorithm characterized by its rapid convergence

## 4.2 Probability Ordinal-Preserving Hashing

### 4.2.1 Notations and Definitions

Vectors and matrices are written as boldface lowercase, e.g., x, and boldface uppercase e.g. X \in \mathbb{R}^{d \times n}. The i-th column and ithe row vectors of matrix X are represented as x_{i} \in \mathbb{R}^{d \times 1} and x_{i} \in \mathbb{R}^{1 \times n}. The i-th row and j-th column element of matrix X is denoted as x_{ij}. The l_{2}-norm of vector x is denoted as \|x \| and the corresponding Frobenius norm of matrix X is dubbed as $\|X \|_{F} = \sqrt{\sum_{i=1}^{d} \sum_{j=1}^{n} x_{ij}^{2}  }  $. X^{T}, X^{-1} and tr(X) represent the transpose, inverse, and trace of X. The l_{21}-norm of X is defined as:

$\|X \|_{21} = \sum_{i=1}^{s} \|x^{i}\|_{2} = \sum_{i=1}^{d} \sqrt{\sum_{j=1}^{n}}  $

sgn(.) is the element-wise sign operator, which indicates the symbol of each element, i.e. 1 if the element is positive and -1 otherwise

### 4.2.2 Ordinal-Preserving Hashing

Suppose we have n training samples denoted as $X = [x_{1}, x_{2}, ..., x_{n}] \in \mathbb{R}^{d \times n}  $, where d and n are the dimensionality of vidual features and the number of training samples, respectively. The corresponding semantic labels are represented as $G = [g_{1}, g_{2}, ..., g_{n}] \in \{ 0, 1\}^{c \times n}  $. $g_{i} \in \{0, 1 \}^{c \times 1}  $ is the one-hot label vector of the i-th sample. Semantic-preserving hasing technqiue aims to learn series of hash functions $\mathcal{H} = \{ h_{i}(.)  \}_{i=1}^{r}  $ to project the original visual feature $\{x_{i}\}_{i=1}^{n}  $ onto a lower dimensional discrete hamming space, i.e., $\mathcal{F}: \mathcal{H} \rightarrow \{-1, 1 \}^{r}  $, in which the binary codes $\mathcal{B} = \{b_{i}(.) \}_{i=1}^{r} \in \{-1, 1 \}^{r \times n}  $ are learned. r is the dimensionality of the learned binary codes, and $b_{i} = [h_{1}(x_{i}), h_{2}(x_{i}), ..., h_{r}(x_{i}) ]^{\top}  $ is the expected binary codes for the data point x_{i}. h_{i} is the i-th hash function. Importantly, during the feature transformation, the hashing function has the exclusive capability of preserving the similarities in the original space. Specifically, if samples x_{i} and x_{j} are similar, the Hamming distance of the obtained binary codes b_{i} and b_{j} should be minimized; otherwise, they have the maximum Hamming distance.

In this book, they focus on a broad concept of preserving ordinal similarity at a high level within the context of hash function learning. This approach goes beyond just maintaining similarity relatinships between neighborhood data points. It also aims to retain the ranking similarity correlations for each data point relative to its proximate, similar neighbors. Rather than basing our loss function on pairwise (doublet) relationsihps, we have established a hashing function that preserves ordinal relationships using triplet-based interactions.

#### Definition 4.1 (Ordinal-Preserving Hashing)

Assume we have any triplet samples, i.e., (x_{i}, x_{j}, x_{k}) in which x_{j} and x_{k} are selected from the similarity neighbors of x_{i}. Let d(. , .) denote the distance metric. For an optimal hash function H, if $d(x_{i}, x_{j}) \leq d(x_{i}, x_{k}) $, the learned binary codes under hash function H satisfy $d(H(x_{i}), H(x_{j})) \leq d(H(x_{i}), H(x_{k}))  $, i.e. $d(b_{i}, b_{j}) \leq d(b_{i}, b_{k})  $. We define such a learning process as the **ordinal-preserving hashing** scheme

### 4.2.3 Ordinal-Preserving Hashing: A Bayesian Perspective

Based on the ordinal-preserving hashing scheme, we aim to explore the triplet correlations between samples $(x_{i}, x_{j}, x_{k}) \in X  $ for learning an effective semantic-preserving hash function. Similarly, x_{j} and x_{k} are from the neighborhood group of x_{i}. Let s_{jk} denote the degree of similarities by comparing x_{i} with x_{j} and x_{k}. For similicity, we measure their similarity degree by $s_{jk}^{i} = d(x_{i}, x_{j}) - d(x_{i}, x_{k})  $, in which $s_{jk}^{i} \gt 0 indicates x_{i}  $ is more similar to x_{k} than x_{j}

Without loss of generality, let $p(b_{i}, b_{j}, b_{k}|s_{jk})  $ be the posterior probability of feature correlation transformation from the visual space to the learned Hamming space. By virtue of the prior assumption of conditional independence on each sample pair and the corresponding Bayesian formulation, the joint posterior probability density function on the given triplet samples can be reformulated as

$\prod_{\{x_{i}, x_{j}, x_{k} \} \in X}  p(b_{i}, b_{j}, b_{k}|s_{jk}^{i}) = \prod_{\{x_{i}, x_{j}, x_{k} \} \in X} p(s_{jk}|b_{i}, b_{j}, b_{k})p(b_{i})p(b_{j})p(b_{k})   $

where $p(b_{i}, b_{j}, b_{k}|s_{jk}^{i}) $ represents the likelihood probability. p(b_{i}) p(b_{j}), and p(b_{k}) are the prior probabilities of the learned binary codes. The log-posterior objective function can be derived as:

$\max \sum_{\{x_{i}, x_{j}, x_{k} \} \in X} \log p(b_{i}, b_{j}, b_{k}|s_{jk}^{i}) = \sum{\{x_{i}, x_{j}, x_{k} \} \in X} \log p(s_{jk}^{i}| b_{i}, b_{j}, b_{k}) + \sum_{x_{i} \in X} \log p(b_{i}) + \sum_{x_{j} \in X} \log p(b_{j}) + \sum_{x_{k} \in X} \log p(b_{k})  $

we can see that the first tem is the posterior probability fn, which considers the ordinal similarity preservation in hash code learning. The remaining terms are prior probability fns, each of which considers the quantization loss, semantic preserving loss, and regularization loss

#### 4.2.3.1 Probabilistic Ordinal Similarity Preservation

For generating a fast stable solution, we directly employ the exponential distribution as the likelihood probability density function. In this way, the likelihood probability fn of the above triplet-induced ordinal similarity preservation is formulated as:

$p(s_{jk}^{i}|b_{i}, b_{j}, b_{k}  ) = e^{- | d(b_{i}, b_{j}) - d(b_{i}, b_{k}) +  \xi |  }, s_{jk}^{i} \gt 0  $

$p(s_{jk}^{i}|b_{i}, b_{j}, b_{k}  ) = e^{- | - d(b_{i}, b_{j}) + d(b_{i}, b_{k}) +  \xi |  }, s_{jk}^{i} \leq 0  $

where \xi is a constant margin value. From the above setting, we can deduce that the learned binary codes are capable of preserving the ordinal relationship from the original visual space. Based on the above eq and the rearrangement inequality, by removing the margin value, the above log-likelhood probability could be reformulated as the following similarity preserving fn:

$\max \sum_{i=1}^{n} \sum_{j \in \mathcal{N}_{i}} \sum_{k \in \mathcal{N}_{i}} \log (p (s_{jk}^{i}|b_{i}, b_{j}, b_{k} )) \Rightarrow \max \sum_{i=1}^{n} \sum_{j \in \mathcal{N}_{i}} \sum_{k \in \mathcal{N}_{i}} s_{jk}^{i} [d(b_{i}, b_{j} - d(b_{i}, b_{k}))]  $

where $\mathcal{N}_{i}$ denotes the index set of the k nearest neighbors of x_{i}. It is clear that the similarity measurement matrix of S is an antisymmetric matrix

To deal with the problem 4.4, we define the matrix $W \in \mathbb{R}^{n \times n}  $ as the similarity weighting matrix to jointly preserve the neighbor ranking correlations and neighbor relationship. The j-th row and k-th column element of W is calculated by

$w_{iv} \overset{\Delta}{=} \sum_{j \in \mathcal{N}_{i}} s_{jk}^{i}, v \in \mathcal{N}_{i} $

$w_{iv} \overset{\Delta}{=} 0, otherwise $

As such, have the following lemma

#### Lemma 4.1

The ordinal similarity preserving function in eq 4.4 is equovalent to $\min  \sum_{j=1}^{n} \sum_{k=1}^{n} w_{jk} d (b_{j}, b_{k})  $

Based on Lemma 4.1, problem 4.4 can be rewritten as:

$\min \sum_{j=1}^{n} \sum_{k=1}^{n} w_{jk} d(b_{j}, b_{k}) = \sum_{j=1}^{n} \sum_{k=1} w_{jk} d(sgn(U^{\top}x_{j}), sgn(U^{\top}x_{k})) $

In practice, to avoid the time-consuming optimization on the full similarity matrix with n \times n, we propose an anchor-induced ordinal similarity measurement to model an ordinal asymmetric similarity preserving loss. Similar to 15, we first randomly select l samples as anchors, i.e. $\{ a_{p} \}_{p=1}  $, and then construct the ordinal correlations between anchor points and the whole dataset. In this way, we can reformulate the problem as an asymmetric learning scheme. Reformulated as:

$\min \mathcal{S}(U) = \sum_{j=1}^{n} \sum_{p=1}^{l} \sum_{k=1} w_{jk} \psi (sgn(U^{\top}x_{j}), sgn(U^{\top}a_{p} ), sgn(U^{\top}x_{k} ) )  $

where \psi(.) is an anchor similarity computing operator

**Remark** is might appear that this method resembles traditional locality preserving projections or anchor graph learning. However, it is important to highlight that the weighting matrix outlined in eq 4.5 differs significantly from those in prior studies. Previous approaches are typically founded on pairwise (doublet) samples and overlook the preservation of ordinal information in similarity relations. Additionally, our approach is distinct from current ranking loss methods, as our primary emphasis is on preserving multiple layers of ordinal information in hash codes.

#### 4.2.3.2 Probabilistic Quantization Function

Similarly, we also define the exponential distribution on a prior probability function. When solving the problem 4.9, we remove the discrete sgn function. To minimize the transformation loss from the obtained real-valued features to the expected binary codres, we define the following quantization loss:

$\mathcal{L}_{1} = \max \sum_{x_{i} \in X} \log p(b_{i}) = \sum_{x_{i} \in X} \log e^{-(loss_{i} (b_{i, U^{\top} x_{i}} ) + \lambda \|U\|_{21} ) } = - \sum_{x_{i} \in X} loss_{i}(b_{i}, U^{\top} x_{i}) - \lambda \|U\|_{21}, s.t. b_{i} \in \{ -1, 1 \}^{r \times 1} $

where the term loss_{i}(.,.) represents the loss function for the i-th data point, which could take the form of an l_{2}-norm loss, l_{1}-norm loss, or M-estimation lss. To encourage row sparsity in the matrix U, a robust l_{21}-norm penalty is applied. This approach ensures that each row of U directly correlates to the learned features in b_{i}, essentially serving as a mechanism for feature selection. The l_{21}-norm applied to U enables the derivation of sparse representations from the original high-dimensinoal visual features, effectively filtering out noise

#### 4.2.3.3 Probabilistic Semantic-Preserving Function

The function for semantic-preserving loss is designed to capture the relationships between the learned binary codes and the provided semantic labels, denoted as G. There are two prevalent approaches. Firstly, semantic labels can be utilized for a classification baseline on the learned binary codes, aiming to make these cores more discriminative. Alternatively, semantic labels can be converted into pairwise similarities. However, the first approach tends to overlook the correlations between samples, while the second does not effectively capture discriminative category information. In our approach, we consider using one-hot semantic labels as the original features. We then employ regressive semantics as the new representation for the generation of flexible and discriminative binary codes

$\mathcal{L}_{2} = \max \sum_{x_{i} \in X} \log p(b_{i}) = \sum_{g_{i} \in G} \log e^{- (loss_{i} (b_{i}, R^{\top} g_{i}) + \lambda \|R\|_{F}^{2}  )} = - \sum_{g_{i} \in G} loss_{i}(b_{i}, R^{\top}g_{i}) - \lambda \|R\|_{F}^{2} s.t. b_{i} \in \{-1, 1 \}^{r \times 1} $

#### 4.2.3.4 The Overall Objective Function

Generally, the proposed Probability Ordinal-preserving Semantic Hashing (POSH) model is built on the log-posterior objective fn in 4.2, which is a conjunction optimization of probabilistic ordinal similarity preservation in eq 4.9, wht probabilistic quantization function in eq 4.10, and the probabilistic semantic-preserving fn in eq 4.11. Therefore, the final objective fn of our POSH is formulated as:

$\mathcal{L} = \min \sum_{g_{i} \in G} loss_{i} (b_{i}, R^{\top}g_{i}) + \sum_{x_{i} \in X} loss_{i} (b_{i}, U^{\top}x_{i}) + \lambda(\|U\|_{21} + \|R\|_{F}^{2}) + \eta \mathcal{S}(U) s.t. b_{i} \in \{-1, 1 \}^{r \times 1}  $

where \lambda and \eta are the trade-off weightin parameters to balance the importance of different terms

## 4.3 Optimization of POSH

Without the loss of generalization, similar to the distance measurement, we directly take the simple least squared functino as the loss metric in loss(.,.). It is clear that the problem is non-convex, and it is difficult to to directly obtain an optimal overall solution wrt all the variables. Moreover, the discrete constraint on B makes it a quadratic integer programming problem. To this end, we adopt an alternating optimization strategy to effectively solve the resulting objective problem, i.e. optimizing one variable when fixing others

### U-Step

Fixing all variables irrelevant to U, have the degenerated problem

$\min_{U} \sum_{x_{i} \in X} loss_{i}(b_{i}, U^{\top}x_{i}) + \lambda \|U\|_{21} + \eta \mathcal{S}(U) = \min_{U} \sum_{x_{i} \in X} loss_{i} (b_{i}, U^{\top}x_{i}) + \lambda \|U\|_{21}  + \eta \sum_{j=1}^{n} \sum_{p=1}^{l} \sum_{k=1}^{n} w_{jk}\psi(sgn(U^{\top}x_{j}), sgn(U^{\top}a_{p}), sgn(U^{\top}x_{k})) s.t. b_{i} \in \{-1, 1 \}^{r \times 1}  $

The sgn(.) fns discrete nature makes optimizing the mentioned problem quite challenging. Therefore, in line with established hashing models, we opt to use the mangitude of each element instead. This adjustment in the quantizaztion process helps in approximating the resulting codes to a binary form. Additionally, for simplicity, we adopt the squared Euclidian distance, as suggested in 14, to measure the distance between pairwise representations. Given the definition of w_{ij} in eq 4.5, we can express the original S(U) from eq 4.8 as $\sum_{j=1}^{n} \sum_{k=1}^{n} w_{jk}\|U^{\top}x_{j} - U^{\top}x_{k} \|_{2}^2 $. This can be further reformulates in matrix form as $\min tr(B\Lambda B^{\top}) = \min tr(U^{\top} X \Lambda X^{\top} U) $. Here, $\Lambda = D - W  $ represents the normalized Laplacian matrix, with D being a diagonal matrix whose (i, i)-th diagonal entry is defined as $\sum_{j=1}^{n}  w_{ij} $. Drawing from the asymmetric anchor graph learning approach outlined in eq 4.10 and 15, the asymmetric learning operator \psi(.) can be reformlated as $\hat{\Lambda} = I - Z \Delta^{-1} Z^{\top}  $, where $\Delta = diag(Z^{\top}1)  $. Rather than utilizing the full locality similarity measurement $W \in \mathbb{R}^{n\times n}  $, we employ $Z \in \mathbb{R}^{n \times l}  $ as an ordinal-preserving anchor graph matrix. This matrix is capable of maintaining the ordinal similarities between the entire dataset $\{x_{i} \}_{i=1}^{n}  $ and the anchor samples $\{a_{p} \}_{p=1}^{l}   $. As a result, this approach significantly reduces the computational complexity of constructing ordinal similarity from O(n^{2}) to O(nl)

Due to the non-convex property of l_{21}-norm regularization, we employ the half-quadratic technique to toptimize it for obtaining more accurate results. Accordinly, the following Lemma

**Lemma 4.2** For a given implicit parameter 4, there s a dual potential function \phi(.), such that $\sqrt{r^{2} + \xi} = \inf_{q\in \mathbb{R}} \{(1/2)tr^{2} + \phi(q) $, in which q  could be determined by $\delta(q) = 1/\sqrt{q^{2} + \xi}  $

Due to the intractable l_{21}-norm regularization, 4.2 is introduced to solve the above 4.13 problem. It can now be refomulated as

$\mathcal{L} = min_{U} \|B - U^{\top}X \|_{F}^{2} + \eta tr(U^{\top}X \hat{\Lambda}X^{\top}U) + \lambda \sum_{i=1}^{r} \{(l_{ii}/2)\|u^{i}\|_{2}^{2} + \phi_{i}(t_{ii}) \}  $

where $t_{ii} = 1/\sqrt{\|u^{i}_{2}^{2} + \xi } $ is the (i, i) diagonal element of the auciliary diagonal variable matrix T. \xi is a very small perturbation, e.g. e^{-6}. \phi_{i}(.) is the i-th dual potential function. In the presence of Lemma r.2, the optimal solution of U is given by setting the derivation of L to zero, and then we have the following closed-form solution:

$U = (\eta X \hat{\Lambda} X^{\top} + XX^{\top} + \lambda T)^{-1} XB^{\top}  $

#### B-Step

Similarity, when fixing other variables that are irrelevant to B, we have

$\min_{B} \|B - R^{\top}G \|_{F}^{2} + \|B - U^{\top}X\|_{F}^{2} s.t. B \in \{-1, 1 \}^{r \times n}  $

Due to the discrete constraint on B, we have $tr(BB^{\top}) = tr(B^{T}B) = nr  $. By some simple computations, we can deduce that the follwing simplifed problem

$\max_{B} B^{\top} (R^{\top}G + U^{\top}X) s.t. B \in \{-1, 1 \}^{r \times n}  $

which has an analytic solution, i.e.

$B = sgn(R^{\top}G + U^{\top}X)  $

#### R-Step

fixing variables not relevant to r

$\mathcal{L} = min_{R} \|B - R^{\top}G\|_{F}^{2} + \lambda \|R\|_{F}^{2} $

It is clear that problem 4.20 is a standard linear regression problem, and its optimal solution is determined by setting $\frac{\partial \mathcal{L}}{\partial R} = 0  $. We have the closed form solution of R as

$R = (GG^{\top} + \lambda I)^{-1} GB^{\top}  $

To obtain the optimal solution of the problem, we iteratively optimize U, R, and B until it converges to a local optimum or meets the max iterations

## 4.4 Algorithm Extension and Theoretic Analysis

### 4.4.1 Nonlinear Anchor Feature Embedding

It is known that the representation ability of the visual featuers can significantly influence the quality of the learned hash codes. In practice, the handcrafted features always include reducnancies or noises, which are harmful to generating high-quality hash codes. To improve the interpretation of the visual features X, we introduce the kernelized anchor features to make the visual features capture the nonlinear property. Inspired by the existing work 1, we use the rRBF kernel function to project the original visual features onto an m-dimensional nonlinear feature space, i.e.

$\phi(x_{i}) = [\exp(-\|x_{i} - a_{1}\|^{2}/\sigma), ..., -\|x_{i} - a_{m} \|^{2}/\sigma]^{T}  $

where $\phi(x_{i} \in \mathbb{m \times 1})  $ is the m-dimensional nonlinear anchor feature vector for the i-th sample x_{i}. $\{a_{i} \}_{i=1}^{m}  $ are m anchor samples that are randomly selected from the visual feature space. \sigma is a predefined kernel width. Based on the given nonlinear feature embeddings, the optimization algorithm can be easily rearranged by using:

$\phi(X) = [\phi(x_{1}), ..., \phi(x_{n}) ] \in \mathbb{R}^{m \times n}  $ to replace the original visual feature matrix $X \in \mathbb{R}^{d \times n} $

### 4.4.2 Out-of-Sample Extension

With the proposed learning algorithm, we can obtain an optimal binary code B of the given training samples X. For samples outsize of the training points, we adopt a two-step hash function learning strategy to generate hash coded. First, when the optimal binary codes B are obtained, we learn the hash function matrix H by optimizing the following linear regression problem:

$\|H^{\top} \phi(X) - B \| + \|H\|_{F}^{2}  $

which has a closed-form solution

$H = (XX^{\top} + I)^{-1} XB^{\top}  $

For a new query sample q_{i}, we need to transform it to the nonlinear features by \phi(q_{i}) with the same anchor examples in eq 4.22. As such, the corresponding hash code of q_{i} is formulated by

$b = sgn(H^{\top} \phi(q_{i}))  $

### 4.4.3 Convergence Analysis

The developed optimization algorithm can iteratively find a lcoal optimum for the objective fn 4.12. Based on each updating step shown in Alg 4, the convergnce of the learning alg is satisfied by the following theorum

#### Theorum 4.1

The poposed alternating learning algorithm can monotonously decrease the value of the objective funtion until it converges to a local optimum

### 4.4.4 Computational Complexity

From alg 4, the major computation overhead lies in optimizing U, R, and B. Specifically, updating variable U requires computing matrix computation operation, and its computational cost is O(m^{2}n + mnr). for updating R, it is the same as the linear regression problem, and the computational burden is O(cnr). For calculating B, it is a simple sign operator on two r \times n matrices, and it comsumes O(2rn). Therefore, the total computational complexity of POSH in each iteration is O((m^{2} + mr + (c+3)r)n) ({m, r, r} << n), which is linear to the number of samples

## 4.5 Experimental Evaluation

### 4.5.1 Datasets

CIFAR-10, MNIST, NUSE-WIDE

### 4.5.2 Baseline Methods and Implementation Details

Compare against

- unsupervised:
  - ITQ
  - SGH
  - OCH
  - JSH
- supervised:
  - ITQ-CCA
  - LFH
  - COSDISH
  - TASH
  - SDH
  - SDHR
  - FSDH
  - BSH
  - R2SDH
  - DBC-LDL
  - FSSH

### 4.5.3 Evaluation Protocols

- mean average precision (MAP)
- top-K precision
- NDCG@100

### 4.5.4 Experimental Results on CIFAR-10

### 4.5.5 Experimental Results on MNIST

### 4.5.6 Experimental Results on NUS-WIDE

Difficult real-world multi-label dataset. The challenge of image retrieval on this dataset lies in recognizing each subject from multiple semantic labels in feature learning and extracting meaningful representations against clustered background

### 4.5.7 Comparison with State-of-the-Art Deep Hashing Models

- DQN
- DHN
- DSH
- DSDH
- DPSH
- DTQ
- DPQ
- DCRH

AlexNet was used as the base model for all deep hashing methods

POSH outperformed most other deep hashing models in most cases, but I feel like AlexNet is kindof old. I wish they had justified why AlexNet instead of something else. I also dont yet understand the interfaces behind which the deep hashing methods operate for producing representations in the Hamming space

POSH performance can be attributed to

1. Unlike deep hashing models that produce relaxed binary codes followed by a thresholding step, POSH directly optimizes the discrete optimization problem. The binary codes generated through the approach are optimally designed to capture each images intrinsic properties and preserve the correlations among different images
2. POSH utilizes regressive semantic embeddings derived directly from the complete semantic labels, ensuring a comprehensive consideration of semantics within a singular learning scheme. This approach is more effective than deep hashing methods that employ a fragmented, batch-wise learning strategy
3. POSH implements a probabilistic ordinal-preserving scheme for learning discriminative binary codes. This scheme is adept at capturing optimal ordinal semantic similartiies, leading to probabilistic and semantically consistent hash codes

### 4.5.8 Experimental Analysis

#### 4.5.8.1 Parameter Sensitivity Analysis

Two key hyperparemeters

- \eta - above 1 is good and stable
- \lambda - results relatively uneffected by this. this observation confirms significance of incorporating smooth regularization in the model

#### 4.5.8.2 Convergence Study

#### 4.5.8.3 Visualization Analysis

holy shit the t-SNE visualization is really crisp. Very clear clusters by label

## 4.6 Conclusion

In this chapter, we introduced a cutting-edge method called Probability Ordinal-Preserving Semantic Hashing (POSH), grounded in a general non-parametric Bayesian framework, designed for scalable image retrieval. POSH employs an ordinal-preserving hashing strategy based on the maximum posteriori estimation, encompassing three key modules:

- a probabilistic ordinal similarity preservation module
- a probabilistic quantization fn
- a probabilistic semantic-preserving function

This method distinctively integrates triplet ordinal similarities and flexbile semantics within a cohesive learning framework. An effective algorithm was devised to tackle the associated optimization challenge, with theoretical analysis suggesting its potential for rapid convergence and manageable computational load. Comprehensive experiemental evaluations on three large-scale benchmark datasets have shown that POSH outshines various contemporary sota hashing methods across diverse evaluation metrics.
