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
