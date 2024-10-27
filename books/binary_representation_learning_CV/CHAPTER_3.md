# Chapter 3. Inductive Structure Consistent Hashing

Semantic-preserving hashing enhances multimedia retrieval by transferring knowledge from original data to hash codes, preserving both visual and semantic similarities. However, a significant bottleneck arises in effectively bridging trilateral domain gaps - visual, semantic, and hashing spaces - to further elevate retrieval accuracy. This chapter introduces the Inductive Structure Consistent Hashing (ISCH) method which coordinates semantic correlations among the visual feature space, binary class space, and discrete hashing space. An inductive semantic space is formulated using a multi-layer stacking class-encoder, transforming naive class information into flexible semantic embeddings. A semantic dictionary learning model aids visual-semantic bridging and guides the class-encoder to mitigate the visual semantic bias. The visual descriptors and semantic class representations are regularized througha coinciding alignment module. To generate privileged hash codes, semantic and prototype binary code learning jointly quantifies semantic and latent visual representations into unified discrete hash codes. An efficient optimization algorithm addresses the resulting discrete programming problem. Comprehensive experiments showcase the superiority of ISCH over sota alternatives under different eval protocols, emphasizing its effectiveness in addressing the trilateral domain gaps and improving multimedia retrieval accuracy

## 3.1 Introduction

Creating concise hash codes is key to enhancing search speeds while maintaining the quality of retrieval

Unsupervised hashing focuses on creating hash functions that maintain the metric distances, distributions, or topological characteristics of unlabeled data points. However, these methods often miss out on the distinct properties embedded in category labels, which are cruicial for classification.

Supervised on the other hand aims to learn and preserve semantic information

Deep hashing utilizes nonlinear hash functions. Most deep hashin gmethods learn continuous features which are subsequently binariezed in a separate step. This approach of continuous relaxation fails to produce truly compact binary codes during optimization, straying from the fundamental concept of hashing. Furthermore, **these models are not yet suitable for real-time applications, as they are complex to fine-tune and optimize and require extensive pre-training of the complete neural network**.

Given the disparate distributions and representations of features across different spaces, a significant challenge in this area is to bridge the trilateral domain gaps. This involves identifying and aligning consistent elements of items across these domains. Traditinoal retrieval models typically concentrate on either maintaining the pairwise (or triplet) similarities among data items linked by common semantic labels, or aligning the generated hash codes with the binary class space. However, these methods often underutilize categories, employing them merely for building similarities or as regressions targets. The existing hashing approaches use discrete optimization of binary codes through bit-by-bit or two-step relaxation techniques leading to time-intensive or suboptimal learning processes

The essence of ISCH in establishing an intermediate semantic space. This space serves as a cruicial link to bridge the gaps between the visual space, the original class label space, and the Hamming space in which learning occurs. This ISCH involves transforming binary class labels into adaptable semantic embeddings. These embeddings ensure that items within the same category exhibit similarities, yet remain distinct from items in other categories. The creation of this intermediate semantic space facilitates the smooth transfer of informative semantics across different domains, effectively addressing the issue of visual-semantic bias. Furthermore, we implement a structure alignment strategy that constructs class prototypes in a semantic dictionary. This aids in the visual-semantic transformation, blending the class structures from the visual and semantic spaces into a refined, unified space. Cuncurrently we integrate disciminative smenatic features with high-order proximal visual embeddings into cohesive hashing codes, enabling the quantificaiton of both visual and semantic vectors into a shared Hamming space. The key contributions are:

There is a great diagram here

ISCH overview:

- inductive semantic space learning (class labels -> semantics)
- visual-semantic space bridging module (visual features -> semantics)
- semantic and prototype hash code learnings ({ semantics, visual features} <-> Hamming)

The learning framework can interactively communicate the discriminative components among the visual feature space, latent semantic space, and the learned discrete Hamming space

- 1. ISCH
  - designed to facilitate the transfer of flexible semantic knowledge across three domains:
    - visual space
    - semantic space
    - discrete Hamming space
- 2. Constructing an inductive semantic space that serves as an intermediary, connecting the visual input space with the semantic label space and the target Hamming space.
  - This connection is facilitated though a nonlinear class encoding scheme, which leverages the semantic relationships among classes to enhance discriminative learning
- 3. Innovative approach for learning an embedded semantic dictionary
  - this method focuses on uncovering the advantages and interconnecions between visual features and semantic representaitons, aiming for structural alignment.
  - Through this process, the developed semantic dictionary effectively channels the visual knowledge of data into the constructed semantic framework
  - also, we ensure the preservation of both high-order proximal visual features and precisely calibrated semantic factors within the discriminative binary codes that are learned
- 4. a formulated discrete learning algorithm specifically designed to topimize the resulting objective function.
  - optimization aims to minimize quantization error

## 3.2 Inductive Structure Consistent hashing

### 3.2.1 Notation and Problem Formulation

For a given matrix A, we denote a^{i} and a_{i} as the i-the row and column vectors of A. ||.||_{F} and Tr(.) denote the Frobenius norm and the trace operator of a matrix, respectively. The squared Frobenius norm is defined as:

$\|A \|_{F}^{2} = Tr(AA^{\top}) = Tr(A^{\top}A) = \sum_{i,j} A_{ij}^{2}   $

sgn(.) regers to the element-wise sign function. I_{r} denotes an identity matrix with r dimensino

Notations

| Symbol | Description |
| - | - |
| X \in \mathbb{R}^{d \times n} | Training data feature matrix |
| x_{i} \in \mathbb{R}^{d \times l} | The i-th feature vector of X |
| L \in \mathbb{R}^{c \times n} | The class label matrix |
| l_{i} \in \mathbb{R}^{d \times n} | The i-th label vector of L |
| B \in \{-1, 1 \}^{r \times n} | The objective binary hash codes |
| W_{i} | The i-th encoding matrix |
| R, Q \in \mathbb{R}^{r \times r} | The orthogonal rotation matrices |
| V \in \mathbb{R}^{r \times n} | The inductive semantic matrix |
| v_{i} \in \mathbb{R}^{r \times l} | The i-th vector of V |
| D \in \mathbb{R}^{d \times r} | The learned semantic dictionary |
| d_{i} \in \mathbb{R}^{d \times l} | the i-th vector of D |
| Z \in \mathbb{R}^{d \times r} | The orthogonal basis matrix |
| G \in \mathbb{R}^{r \times n} | The coefficient matrix |
| U_{r} \in \mathbb{R}^{r \times n} | The optimal basis factorization matrix |
| q | The feature vector of a query sample |
| n | The number of training instances |
| d | The dimensionality of visual features |
| c | The number of classes |
| r | The hash code length |

Suppose we have a dataset comprising n input pairs, denoted as:

$D = \{(x_{i}, l_{i} ), i = 1, ..., n \} \subseteq X \times L $

Here, $X = \{x_{i} \}_{i=1}^{n} \in \mathbb{R}^{d \times n}  $ represents a centralized visual feature space of dimension d, while $L = \{l_{i} \}_{i=1}^{n} \in \mathbb{R}^{c \times n}  $ corresponds to point-wise semantic labels, concepts, or tags associated with each input, with c being the dimension of a semantic vector.

The primary objective in learning to hash is to map the high-dimensional data space X into a r-bit discrete Hamming space $\mathcal{F}: X \rightarrow B \in \{-1, +1 \}^{r \times n}  $. This transformation aims to approximate nearest neighbors using compact binary codes. In the context of image retrieval, the challenge is to effectively return relevant samples from the database X in response to a given query image q.

### 3.2.2 Inductive Semantic Space Construction

Labels or concepts are commonly employed as a benchmark for semantic-preserving hashing as they provide an *ideal* way to gauge semantic differences between samples. It might seem logical to use these labels directly as feature representations. However, this approach is not necessarily the most effective in terms of semantic representation. This is because fixed binary semantic codes offer limited flexibility in adapting to various code lenghts and fully capturing all informative attributes present in images. Specifically, in single-label datasets, an image's semantic vector is represented by a single non-zero element, indicating its semantic relationship with other images. In contrast, multi-lable scenarios involve each image being characterized by multiple tags to describe its features. However, these multiple labels are often subjective and typically require expertise for accurate identification. For instance, consider two label vectors l_{i} = [1, 0, 1, 1] and l_{j} = [1, 1, 1, 0]. While these might be deemed similar due to overlapping categories, their squared Euclidian Distance and Hamming distance (as applied in this work) are both d(l_{i}, l_{j}) = 2. For 4-bit codes, this distance is not minimal, highlighting a limitation in this approach to semantic representation

By imitating the fruit fly olfactory circuit, we can employ the sparse binary random projections to map semantic representations onto a new semantic space, i.e., WL where L is the piecewise semantic matrix. Furthermore, in order to capture more characteristics underlying the discrete category descriptors L, we propose to learn a non-iterative stacking **class-encoder** $f_{ce}(L;\theta) $ (\theta is the learning parameter) to generate the semantic embeddings of class representations

$\min_{V, \theta} \|V - f_{ce}(L;\theta)  \|_{F}^{2}  $

where $f_{ce}(L;\theta) = g(W^{\top}_{M}^{\top} ... g(W_{i}^{\top}... g(W_{1}^{T}L ) ) )  $ is a stacking encoder model with M layers, g is an activation function (such as tanh and ReLU), $\theta = \{W_{1}, ..., W_{M} \} \in {\mathbb{R}^{p_{1} \times p_{2}}, ..., \mathbb{R}^{p_{M-1} \times p_{M} }  } $ is a learning parameter set, and p_{i} is the number of units for the i-th layer (p_{1} = c and p_{m} = r). To approximate the binary projection strategy, we utilize the tanh function activation, i.e., $g(x) = (e^{x} - e^{-x} )/(e^{x} + e^{-x} )  $. To effectively train the nonlinear function, we emply a three-layer stacking class-encoder, i.e., $f_{ce}(L;\theta) = g(W_{2}^{\top} g(W_{1}^{\top} L))  $

By constructing the latent semantic space V, we can learn discriminateive binary codes by minimizing the quantization error of projecting semantic embeddings to the vertices of a binary hypercuble. In pursuance of minimum space transferring error, we encourage to exchange the principal components between two spaces by using an orthogonal rotation matrix

We have:

$\min_{V, Q, B, \theta} \|V - f_{ce}(L;\theta)  \|_{F}^{2} + \beta \|B - QV \|_{F}^{2}  s.t. B \in \{-1, 1 \}^{r \times n}, B^{\top}1 = 0, Q^{\top}Q = I_{r} $

where \beta is the weighting parameter, and Q is the rotation matrix. The second constraint is the bit balance condition to maximize the information of each bit, which means the number of data items mapped to each binary code is the same

In the context of our problem, our model is capable of offering the necessary flexibility to transform the original c-dimensional discrete class vector into an r-dimensional induced semantic embedding. However, this *independent* learning approach might lead to instability and trivial results. The primary concern here is the uncertainty regarding the effectiveness of these autonomously learned semantic embeddings in maintaining similariy, which could result in the generation of suboptimal binary codes seen in 2.3. To address this issue, we introduce a visual semantic bridgin model. This model is designed to "supervise" the construction of the semantic space by mapping visual elements onto the induced embeddings, thereby ensuring a more robust and effective learning process

### 3.2.3 Visual to Semantic Bridging

Establishing links between the visual space and the learned semantic space is cruicial, particularly because the class-encoder lacks initial knowledge to define a distinct hyperspace. Additionally, a substantial structural disparity frequently exists between the visual and semantic spaces, primarily due to their independent nature. Visual vectors, being low level, tend to include a greater amount of non-descriptive information. In contrast, high-level semantic embeddings are more refined and are positioned at the forefront of vision-related tasks.

In the process of space transformation, our objective is to devise a relational function that can transfer knowledge from a d-dimensional visual vector x_{i} \in X to its corresponding r-dimensional semantic embedding vector v_{i} \in V. While a regression-based approach is a direct method for briding the gap between these spaces, it falls short in uncovering the intrinsic structure of the semantic space and in effectively capturing the nuances of the visual-to-semantic relationships. To address these shortcomings, we propose, for the first time, to conceptualize the transformation from the visual space to the semantic space as a problem of semantic dictionary learning within the context of binary code learning. This approach ensures a harmonious alignment between the learned semantic embeddings and the visual features.

To achieve th eaim of knowledge transfer, a transition dictinoary is constructed to deliver the principled visual components from the visual space X to the deduced semantic sapce V. Specifically, we encode each visual feature vector as a linear combination of informative basis entries, which are the atoms of the accumulated dictionary. Let D \in \mathbb{R}^{d \times m} be a dictionary, and we propose to reconstruct a visual feature x_{i} by using Dv_{i}, where v_{i} is the encoded embedding of its semantic label, i.e., f_{ce}(l;, \theta). In this way, we can formulate

$\min_{D,Q,\theta} \|X-DV  \|_{F}^{2} + \|V - f_{ce}(l; \theta) \|_{F}^{2}  s.t. \|d_{i}  \|_{2}^{2}  $

This is very different from the conventional dictionary learning by sparse encoding whereby v_{i} needs to be estimated together with D. By virtue of taking the inductive semantic embedding V as the coding coefficients, each dictionary basis is naturally equipped with the semantic meaning of visual features, resulting in a discriminative *semantic dictionary*. For each x_{i}, its corresponding v_{i} is the aligned semantic embedding of its class label l_{i} using the translation dictionary D in the calibration space. Therefore, the semantic dictionary D can reconcile the semantic space with the conceptual visual feature space, which is also a space calibration process

### Remark 3.1

In the research, the term "calibration process" refers to ensuring that the semantic factors created by the inductive semantic space are interactively adjusted to align with the core components of visual semantics. Specifically, the semantic space we construct acts as an impartial inductive intermediary. However, this inductively formulated space often struggles to fully articulate and grasp the inherent intrinsic components that are integral to visual semantics. Therefore, the role of the visual-semantic bridging model is cruicial as it facilitates semantic calibration between the visual and inductive spaces. This ensures that there is semantic consistency maintained throughout the alignment of these spaces and the subsequent transfer of knowledge

### 3.2.4 Prototype Binary Code Learning

To preserve the full common information of the prototype visual features into the learned binary codes B, we further consider using **matrix factorization for hash code learning, since it has been proved to show superior performance in latent space learning**. In particular, to obtain the latent embedding with r-th order proximity preservation, truncated singular value decomposition is employed to formulate the best rank-r approximation of X, that is, $\|X - X_{r} \|_{F} \leq \|X - M \|_{F}  $, where $X_{r} = U_{r}\Sigma_{r}V_{r}^{\top}  $ and M is any matrix of rank at most r. As such, an intuitive way of learning the prototype binary codes is to threshold the latent embeddings with a sign function, i.e., $sgn(\Sigma_{r}^{1/2} V_{r}^{\top})  $. However, such a procedure may lead to a suboptimal solution with large quantization loss. Alternatively, we consider the following low-rank matrix factorization problem:

$\min_{Z, G} \|X - ZG  \|_{F}^{2}  s.t. Z^{\top}Z = I_{r} $

which can be easily optimized by an alternative learning manner. Clearly, the above problem has one critical point, that is, $Z = U_{r} \in \mathbb{R}^{d \times r}  $ and $G = U_{r}X \in \mathbb{R}^{r \times n}  $, where the gradient wrt {Z, G} vanishes. We can observe that Z and G are regarded as the orthogonal basis matrix and latent coefficient matrix, respectively. As for our prototype binary code learning, the latent embedding G is expected to discretely represent the low-dimensional codes of visual feature X. Therefore, we apply the optimal rotation matrix to transform the real-valued G to discrete B by the orthogonal matrix $R \in \mathbb{R}^{r \times r}  $, i.e., B = RG. It is easy to deduce that $(ZR)^{\top}ZR = I_{r}  $. In this way, when denoting $\hat{Z} = ZR $, we have

$\min_{\hat{Z}, B} \|X - \hat{Z}B  \|_{F}^{2} s.t. \hat{Z}^{top}\hat{Z} = I_{r}, B \in \{-1, 1 \}^{r \times n}, B^{\top}1 = 0  $

### 3.2.5 Objective Function

To preserve the interactive connections between the visual space, the constructed semantic space, and the expected discrete Hamming space, we can combine the equations above to form the overall loss fn of the ISCH

$\min_{D, Q, \hat{Z}, V, B, \theta} \|X - DV \|_{F}^{2} + \beta \|B - QV \|_{F}^{2} + \gamma \|X - \hat{Z}B \|_{F}^{2} + \|V - f_{ce}(L;\theta) \|_{F}^{2} s.t. B \in \{-1, 1 \}^{r \times n}, B^{\top}1 = 0, \|d_{i} \|_{2}^{2}, Q^{\top}Q = I, \hat{Z}^{\top}\hat{Z} = I_{r} $

where \beta and \gamma are the weighting parameters to balance the importance of different terms. The objective function aims to minimize the transferring error from visual to semantic features, meanwhile preserving the consistent visual and semantic structures in the learned discrete hashing codes

### 3.2.6 Optimization

The problem is a non-convex problem due to the discrete contraint and non-convex multi-layer encoder. Because there is no direct solution to all variables, we employ an iterative learning scheme to generate a local optimum by alternatively updating each parameter when fixing others. We first randomly initialize varibales \theta, binary codes B, and rothogonal matrices Q and \hat{Z} and then encode the semantic labels L into V. The whole alternating procedure is shown as follows.

#### D-Step

When fixing other variables, problem 3.6 can be degenerated as the following subproblem

$\min_{D} \|X - DV   \|_{F}^{2} s.t. \|d_{i} \|_{2}^{2} \leq 1  $

which is a classical **dictionary learning** problem and can be solved by its Langrange dual.

The Langrange dual function is:

$\mathcal{L}(\labmda) = \inf(\|X - DV \|_{F}^{2} + \sum_{i=1}^{r} \lambda_{i}(\|d_{i} \|^{2} -1) )  $

Where \lambda_{i} is the Lagrange multiplier of the i-th ihnequality constraint of dictionary atoms. The solution of above is:

$D = XV^{\top}(VV^{\top} + \Lambda)^{-1}  $

where \Lambda is a diagonal matrix and its diagonal entry $\Lambda_{ii} = \lambda_{i} (i=1, 2, ..., r) $

#### Q-Step

When fixing other variables, problem wrt Q becomes the following subproblem:

$\min_{Q} \|B - QV  \|_{F}^{2} s.t. Q^{\top}Q = I_{r}  $

which is the well-known Orthogonal Procrustes problem and has a closed-form solution given by the following lemma

##### Lemma 3.1

Suppose U_{0} and P_{0} are the left and right singular matrices of Singular Value Decomposition (SVD) on (BV^{\top}). Then the optimal solution of 3.10 is Q* = U_{0}P_{0}^{\top}

### \hat{Z}-Step

Similarly, when fixing other variables, problem wrt \hat{Z} becomes the following subproblem:

$\min_{\hat{Z}} \| X - \hat{Z}B \|_{F}^{2} s.t. \hat{Z}^{\top}\hat{Z} = I_{r}  $

which has th esame form of problem 3.10. Based on Lemma 3.1, the analytic solution for problem 3.11 is given by:

$\hat{Z} = U_{1}P_{1}^{\top}  $ where U_{1} and P_{1} are the left and right singular matrices of XB^{\top} respectively

### V-Step

V will be reformulated as the following subproblem

$\min_{V} \| X - DV \|_{F}^{2} + \beta \| B - QV \|_{F}^{2} + \| V - f_{ce}(F;\theta)\|_{F}^{2}  $

which has the closed form solution by wetting the gradient of the problem wrt V to zero. We have

$V = (D^{\top}D + (\beta + 1) I_{r})^{-1} (D^{\top}X + Q^{\top}B + f_{ce}(L;\thata) )$

Clearly there is a nonlinear stacking encoder term $\| V - f_{ce}(L;\theta) \|_{F}^{2}  $, which may jeapordize the final efficient convergence of problem 3.13. For seeking a local optimum with fast convergence, we remove the class-encoder term here, and then the following approximated solution is given by:

$D = (D^{\top}D + \beta I_{r} )^{-1} (D^{\top}X + Q^{\top}B  )  $

### B-Step

$min_{B} \beta \| B - QV \|_{F}^{2} + \gamma \| X - \hat{Z}B \|_{F}^{2}  s.t. B \in \{-1, 1 \}^{r \times n}, B^{\top}1 = 0  $

Due to $Tr(BB^{\top}) = rn $, $Q^{\top}Q = I  $, and $\hat{Z}^{\top}Z = I  $, this can be rewritten as the following optimization problem:

$\max_{B} B^{\top} (\beta QV + \gamma \hat{Z}^{\top}X) s.t. B \in \{-1, 1 \}^{r \times n}, B^{\top}1 = 0  $

which means mapping $\Psi = \beta QV + \gamma \hat{Z}^{\top}X  $ onto a balanced Hamming space. The optimal solution for the problem is:

$B = sgn(\Psi - median(\Psi) )  $

where median(\Psi) is the median function of matrix \Psi

### F-Step

When fixing the other variables, we need to update the three-layer stacking class-encoder term, i.e. $\min_{\theta} \| V - f_{ce}(L;\theta)\|_{F}^{2} = \| V - W_{2}^{\top} g(W_{1}^{\top} L) \|_{F}^{2}  $, which can be solved by gradient descent (GD). To avoid the over-fitting of weights, we add the Frobenius norm as the regularization term, and then we need to minimize:

$\mathcal{F}(W_{1}, W_{2}) = \| V - W_{2}^{T} g(W_{1}^{T}L)\|_{F}^{2} + \eta( \|W_{1}\|_{F}^{2} + \|W_{2}\|_{F}^{2} )  $

where \eta is a regularization parameter (\eta = 0.01 in experiments)

Some derivative equations

The optimization algorithm iteratively updates the setps until it reaches either a local optimum or the pre-established max number of iterations. To minimize computational complexity, we opt for a modified approach where the update of the F step is performed after serveral main iteratinos, specifically every three iterations. This adjustment does not compromise the fundamental goal of constructing the inductive semantic space, which is to identify a viable and adaptable latent subspace. Such a subspace is cruicial for facilitating effective communication between the visual space, the semantic class space, and the discrete Hamming space

### 3.2.7 Out-of-Sample Extension

Typically, an effective hashing system should be capable of extending its learning to generate high-quality hash codes for new instances that were not part of the training dataset, without the need to retrain the entire model - this is called the `"out-of-sample" problem`

Thus far, focus has been on learning optimal binary codes B for a predefined set of training data X using our learning algorithm. To address the generation of binary codes for new data points, as inspired bu the successful two-step hashing technique 21, it is necessary to establish a hash mapping function: $F: X \rightarrow B  $. This function is designed to create binary codes for any new data encountered. In this context, our work employs a straightforward linear regression method to learn this requisite hash function.

Specifically, given the learned binary codes B, the mapping function is formulated by optimizing:

$\min_{P} \| P^{T}X - B\|_{F}^{2} + \| P \|_{F}^{2}  $

which has a closed-form solution, i.e.,

$P = (XX^{T} + I)^{-1} XB^{T}  $

As such, we can generate the binary codes for the new query q by using sgn(P^{T}q)

## 3.3 Experiments

### 3.3.1 Datasets

- CIFAR-10
- NUSWIDE
- ImageNet
- MSCOCOC

use two single-label datasets and two multi-label datasets. Use both handcrafted features and deep learning features to thoroughly asses the performance of various methods

For CIFAR-10, extract 512 dimensional GIST features

for NUS-WIDE, use provided 500-dimensional bag of words features

For both datasets, implement 1000-dimensinoal nonlinear anchor features to enhance feature interpretation, following the approach in 27

For ImageNet and MUSCOCO, we evaluate the 4096-dimensional deep CNN features extracted from the pre-trained VGG16-fc7 layer.

### 3.3.2 Experimental Settings

#### 3.3.2.1 Comparison Methods

Unsupervised hashing methods ITQ, SGH, DSH, OCH

- ITQ
  - Well-known, which constructs an orthogonal rotation matrix to minimize the quantization error of mapping the real-valued features to compact binary codes
- SGH
  - graph hashing method that approximates the whole cumbersome large graph through feature transformation and employs a sequential learning model for bit-wise hash functions generation
- DSH
  - utilizes the geometric structure of the data to guide and select the random projection functions as the optimal hash functions
- OCH
  - considers the permutation relations among samples based on the constructed ordinal graph to preserve the ranking similarities into the objective binary codes

12 sota supervised hashing methods BRE, KSH, ITQ-CCA, FashHash, LFH, SDH, NSH, SDHR, FSDH, BSH, FDCH, and R2SDH

- BRE
  - minimizes the reconstruction error between Euclidian distances of original data and Hamming distances of the corresponding binary codes
- KSH
  - formulates a kernel-based supervised hashing scheme by preserving pairwise correlations between samples into the learned binary codes
- ITQ-CCA
  - initializes the lower dimensional features by using the supervised CCA algorithm within the ITQ framework
- FashHash
  - employs decision trees as the nonlinear hash functions for supervised hash code learning
- LFH
  - learns similarity-preserving binary codes based on latent factor models with stochastic learning
- SDH
  - integrates optimal binary codes learning and the linear classification loss
- NSH
  - builds the structure-preserving transformation between the label vectors and target binary codes
- SDHR
  - optimizes SDH by retargetting the regression space with a large margin constraint
- FSDH
  - reformulates SDH by regressing the class labels to the corresponding hash codes
- BSH
  - supervised semantic hashing method under the Bayesian probabilistic treatment, which can adaptively figure out the importance of different hash bits and identify the regularization hyper-parameters to generate compact but informative hash codes
- FDCH
  - employs the regressive labels as the semantic embeddings with a drift error term for hash code learning
- R2SDH
  - improves the robustness of SDH by replacing the original least regression loss by a correntropy loss, meanwhile a rotational transformation on the label matrix is added to promote its flexibility

#### 3.3.2.2 Evaluation Protocols

- Mean Average Precision
- precision at top 500 results
- Normalized Discounted Cumulative Gain at rank 100

### 3.3.3 Experimental Results

#### 3.3.3.1 Accuracy Comparison with the State of the Arts

Across the board, retrieval accuracy, as measured by the three metrics, remains consistent regardless of the changes in code lengths

The effectiveness of the method is largely due to the use of induced mid-level semantic representations, which possess clear semantic meanings capable of effectively bridging the gap between low-level visual features and high-level semantic information

#### Remark 3.2

The ISCH models performance can be largely attributed to its effective preservation of semantic consistency across three distinct domains. This is achieved by transforming semantic labels into a malleable inductive space using an efficient stacking class-encoder. Alongside, the thoughfully crafted visual-semantic bridging approach fosters an interactive learning environment for semantic calibration, thereby augmenting the inductive semantics to more accurately reflect key visual semantic elements. Cruically, the comprehensive learning framework of ISCH is adept at retaining rich and discering semantic information from both the visual and category-level semantics. This integration is cruicial for the generation of learned hash codes. It is this capability to encapsulate such informative and discriminative semantic content that underpins the remarkable performance of ISCH on the CIFAR-10 dataset

#### Remark 3.3

is competitive on MSCOCO, but not OP. one potential explanation for this is the inherent challenge posed by the datasets multi-label class-imbalance characteristic. specifically, the inductive space derived from the presence of multiple imbalanced class labels can lead to the complex issue of label coupling. Technically, the visual-semantic bridging module addresses this label bias problem, but the visual features extracted from the VGG network, which was pre-trained on the single-label ImageNet, may predominantly influence the class-encoder toward capturing fundamental semantic information. Consequently, out models learned features tend to preserve the primary semantic components of visual semantics, resulting in impressive top-K ranking accuracy

#### 3.3.3.2 Comparison with Deep Hashing Methods

Deep hashing methods can naturally encode effective deep representations by any nonlinear hashing functions

- CNNH+
- DLBHC
- DNNH
- DHN
- DSH
- KSH-CNN
- DSRH
- DRSCH
- BDNN
- DPSH
- SuBiC
- HashNet
- DVStH

bro, this approach "consistently surpasses all of the deep hashing methods used for comparison"

This compelling performance differential underscores the effectiveness of our proposed learning framework. Several key factors:

- 1. The preservation of structurally consistent semantics within the learned binary codes during a single-step encoding process sets the method apart from deep learning approaches that **rely on multiple batch-wise learning iterations**. This structural consistency enhances the quality of our binary codes
- 2. the adept transformation of semantic labels into adaptable parameters for hashing function learning. the transformation enables optimizing the semantics used to guide the generation of discriminative binary codes, resulting in hash codes that maintain semantic coherence

### 3.3.4 Further Evaluation

#### 3.3.4.1 Ablation Study

Four degradation variants:

- ISCH-I removes the inductive semantic space and randomly initializes V
- ISCH-S replaces f_{ce} with a sinle linear projection
- ISCH-V removes the visual-semantic bridging module
- ISCH-P removes the prototype binary codes learning part

#### 3.3.4.2 Efficiency Comparison

#### 3.3.4.3 Convergence and Sensitivity Analysis

#### 3.3.4.4 Visualization

## 3.4 Conclusion
