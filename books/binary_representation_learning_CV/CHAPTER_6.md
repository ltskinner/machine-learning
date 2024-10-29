# Chapter 6. Deep Collaborative Graph Hashing

The remarkable success of deep hashing in large-scale image retreival is attributed to its potent discriminative representation through deep learning and the computational efficiency of compact hash code learning. However, existing deep semantic-preserving hashing approaches often treat semantic labels as ground truth for classification or transform them into prevalent pairwise similarities, overlooking interactive correlations between visual semantics in images and category-level labels. Moreover, these strategies utilized fixed piecewise or pairwise semantics as optimization objectives, limiting flexibility in semantic representation and adaptive knowledge communication during hash code learning. This chapter introduces a pioneering approach, Deep Collaborative Graph Hashing (DCGH), considering multi-level semantic embeddings, latent common space construction, and intrinsic structure mining for discriminative hash code learning in large-scale image retrieval. DCGH introduces the first collaborative graph hashing for image retrieval. Instead of a conventional single-flow visual network, a dual-stream feature encoding network jointly explores multi-level semantic information across visual and semantic features. A shared latent space is established based on space reconstruction to concurrently explore information and bridge the semantic gap between visual and semantic space. Additionally, a graph convolutional network preserves latent structural realtions in optimal pairwise similarity-preserving hash codes. The entire learning framework is optimized end to end. Extensive experiments across diverse datasets demonstrate DCGH's superb image retreival performance against sota supervised hashing methods. The incorporation of collaborative graph hashing facilitates capturing intricate semantic correlations for enhanced large-scale image retrieval.

## 6.1 Introduction

Data-dependent hash codes which show superior performance to data-independent hashing methods, such as locality-sensitive hashing

Current research on hashing mainly tackles two problems:

- discriminative representaiton learning
- semantic gap bridging from real-valued features to discrete hash codes

supervised hashing outperforms unsupervised in most cases

deep hashing, not neural hashing

deep networks act as nonlinear hash functions

Still challenges with deep hashing. Existing deep hashing methods always employ the piecewise label for classification purposes or exploit pairwise or multiwise similarities between samples for similarity preservation. However, using fixed semantics for producing hash codes lacks enough flexibility in semantic utilization, which directly limits the representation capability of semantic transfer during feature learning. Moreover, conventional deep hashing architectures only use single-stream visual encoding network for hash code learning, which fails to explore the corelations between semantics and visual embedding. Furthermore, the multi-level semantic knowledge is also under-explored to collaboratively capture multi-source semantics for collective semantic preserving hash code learning.

DCGH:

Designs a new dual-stream deep hashing network to jointly encode the visual images and binary semantic labels into a joint hash code space, which can capture the correlations between underlying semantics in visual space and fine-grained semantics in semantic labels. A well-designed latent space is built to generate shared semantic features to produce semantic hash codes. Meanwhile, the semantic structural similarities are preserved based on the graph convolutional network. Furthermore, the final binary codes are derived from three branches: visual representation, semantic encoding, and shared latent embedding

Key contributions:

- present DCGH
  - simultaneously considers multi-level semantic interaction, shared latent space construction, and structural similarity prservation in one unified learning framwork
- concieve a dual-stream deep representation learnin gnetwork
  - as well as a common latent space with semantic graph structure
- the visual, semantic, and latent representations are collectively leveraged to enhance the correlations among data points, which are endowed with maximum intra-class compactness and inter-class separation
- experiments

## 6.2 Deep Collaborative Graph Hashing

### 6.2.1 Notation and Problem Frame

Pairwise similarity matrix S to indicate semantic relevance between any pair of data points 1 = similar, -1 = dissimilar

Goal of deep supervised hashing is to learn an encoding fn f(.;\theta) parameterized by \theta, which can encode the input images X into compact hash codes B. DCGH aims to embed high dimensional images into k-dimensional binary codes

#### Definition of Collaborative Deep Hashing

Collaborative deep hashing means we employ multiple information derived from the same image modality to make comprehensive feature embedding and similarity aggregation, such that the generated unified hash codes can capture the principled common and discriminative components from multi-source information

Three branches:

- visual image representation learning framework
- semantic encoding network
- collective latent embedding network

The dual-stream feature encoding network contains an image representation learning network and a label encoding network, which are jointly used for generating visual and semantic hash codes. Moreover, an autoencoder network is used to build the latent relations between visual and semantic encoding, and a latent space is deduced to capture the common underlying semantics across visual-semantic representations. In addition, a multi-layer GCN is imposed on latent embeddings to explore the structural relations between samples. The final optimized hash codes are colalboratively achieved in a joint learning framework

### 6.2.2 Visual Image Representation Learning

Similar to deep pairwise similarity-preserving hashing, we take the pairwise similarity between samples into account in a visual feature learning. If two samples are similar, their hash codes should have minimum Hamming distance

Based on the equivalent inference on the relationship of inner product and Hamming distance, we construct a deep visual pairwise similarity-preserving hashing network based on the deep CNN architecture, and have:

$\min_{b_{i}, b_{j}, \theta_{v}}  \mathcal{R}_{1} = \sum_{i=1}^{n} \sum_{j=1}^{n} (b_{i}^{\top}b_{j} - ks_{ij} )^{2}  s.t. b_{i} = sgn(f_{v}(x_{i};\theta_{v})), b_{i}, b_{j} \in \{-1, 1\}^{k} $

where b_{i} is the encoded binary code for sample x_{i} and f_{v}(.;\theta_{v}) is the visual encoding network with learnable parameters \theta_{v}. Use AlexNet. Due to difficult optimization problem with the **discrete constraint**, we can make a simply relaxed approximation of above eqn with a matrix-form loss fn, and we add a classification loss to make the learned binary codes category-level separable. The loss fn can be formulated as:

$\min_{B_{v}, \theta_{v}} \mathcal{R}_{1} = \|B_{v}^{\top}B_{v} - kS \|_{F}^{2} + \eta \|\hat{L} - L \|_{F}^{2} s.t. B_{v} = sgn(f_{v}(X;\theta_{v})) B \in \{-1, 1 \}^{k \times n}   $

where $\|.\|_{F}$ is the Frobenius norm for matrices, and $\hat{L}$ is the predicted classification results of the learned binary codes B. Moreover, to make the visual encoder network globally derivable, we replace the discrete sgn(.) with the hyperbolic tangent function tanh(.). We denote $H_{v} = \tanh(f_{v}(X;\theta_{v})) \in \mathbb{R}^{k \times n}  $ is the output of the visual encoder. Intuitively, we expect that H_{v} is maximally approximate to the binary codes B, which is acted as a quantization loss for pursuiing minimum quantization error. The above can be reformulated as:

$\min_{B_{v}, \theta_{v}} \mathcal{R}_{1} = \|H_{v}^{\top}B_{v} - kS\|_{F}^{2} + \gamma \|H_{v} - B_{v}\|_{F}^{2} + \eta \| \hat{L} - L\|_{F}^{2} s.t. H_{v} = \tanh(f_{v}(X; \theta_{v})), B_{v} \in \{-1, 1 \}^{k \times n}  $

where \gamma and \eta are weighting coefficients to balance the importance of different terms in representation learning

### 6.2.3 Semantic Feature Encoder Network

This module is designed to learn flexible semantic embeddings from the available labels, meanwhile picking up the ignored semantic information in traditional supervised hash methods. Those conventional deep hashing methods usually utilize the labels for constructing pairwise similarity matrix or directly as the grount truth for classification and regression.

It is notable that visual semantics are the intrinsic semantic knowledge for image representation. However, these existing methods employ fixed semantic for feature encoding, and the underlying semantic information is discarded during the process of space transformation instead of being involved in the hash learning process. For example, "bird" is obviously closer to "cat" than "truck", which fails to be reflected in one-hot label vectors or classification loss. As a result, these methods lack the flexibility to learn discriminative hash codes.

Notably, for supervised learning, labels are annotations are always utilized for ground truth in learning to hash, but very few works realize t**hey are also the ideal measurement for semantic difference among samples**.

Therefore, we take the semantic labels or annotations as binary features to learn flexible lengths of binary codes, which is an advisable choice. Here, a semantic feature encoder that projects the given labels to flexible semantic features is constructed; meanwhile, the pairwise similarities are preserved in the objective hash codes.

Objective fn for the semantic feature encoder module:

$\min_{B_{l}, \theta_{l}} \mathcal{R}_{2} = \|H_{l}^{\top} B_{l} - kS\|_{F}^{2} + \gamma \|H_{l} - B_{l}\|_{F}^{2} s.t. H_{l} = \tanh(f_{l}(L;\theta_{l})), B_{l} \in \{-1, 1 \}^{k \times n}  $

where \gamma is the hyper-parameters to weigh the importance of different terms. $H_{l} = \tanh(f_{l}(L;\theta_{l}))$ is the output of the semantic encoding network with the network parameters \theta_{l}. It takes full advantage of semantic information revealed by the semantic labels and then will be integrated into the end-to-end framework to guide hash code learning

### 6.2.4 Collective Latent Graph Embedding

The above two branches can embed the visual features and semantic into respective feature spaces, while the joint semantic knowledge and features are less explored in the feature learning process. Generally, these above two feature encoders intrinsically generate particular semantic features from two distinctive views. That means, they naturally have different semantic meanings from two heterogeneous perspectives, while the essential representation of two views should be derived from a shared underlying representation that can descrive the nature of data and reveal the underlying structures across different embedding spaces

To achieve the goal of homogenous representation, derived from the shared underling representation, propose a graph-regularized latent space construction module to link the relationship between visual features and semantic features into one organic whole. Instead of simple concatenation, introduce the feature encoder and decoder scheme. In the meantime, the common latent embeddings are generated

The latent embeddings can be derived from the observed spaces in an interactive learning paradigm. We denote the latent embeddings as U, while H_{l} and H_{v} are the visual and semantic features, generated by the visual and semantic encoder modules, respectively. To this end, we build the latent space U, which is encoded from the semantic embeddings H_{l}, meanwhile can be decoded to the visual features H_{v}. We take one conversion link that is $H_{l} \rightarrow U \rightarrow H_{v}  $, i.e.

$\min_{\Theta_{e}, \Theta_{d}} \mathcal{R}_{3} = \| H_{v} - g(U;\theta_{d})\|_{F}^{2} s.t. U = g'(H_{l};\theta_{e})  $

which formulates the reconstruction purpose from the semantic embeddings to the intrinsic vidual features. g(.;\theta_{e}) and g'(.;theta_{d}) are the encoder and decoder networks, respectively.

A typical GCN learns an encoder f_{gcn}(.,.) to extract features on a predefined graph G = (N,E), in which N and E denote the nodes and edges, respectively. Herein, the graph nodes are represented by the latent features, i.e., U = [u_{1}, ..., u_{n}], while the edges are constructed based on the adjacent matrix $A = \{a_{ij} \}_{i,j = 1}^{n}  $, which is defined as $a_{ij} = l_{i} \times l_{j}  $. Following 17, we use noramlized adjacent matrix $\tilde{A} = A + I_{n}  $, where I_{n} is the identity matrix, for undirected graph G. The m-th layer of the GCN is represented by $F^{m} = f_{gcn}(F^{m-1}, \tilde{A})  $, where m > 1. For the initial graph layer, we use $F^{0} = \{u_{i} \}_{i=1}^{n}  $. As such, we utilize the propogation rule of multi-layer GCN for feature encoding, which is defined as:

$F^{m} = f_{gcn}(F^{m-1}, \tilde{A}) = \sigma(\tilde{D}^{\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}F^{m-1}W^{m}   )  $

where \sigma(.,.) is an activation operator like "Relu". \tilde{D} is the diagnoal node degree matrix of \tilde{A}, and W^{m} is the learnable weight parameters that are played like a convolution filter in CNNs. Similar to the above visual and semantic encoding modules, we feed the outputs of graph encoding into the pairwise similarity preserving loss, that is:

$\min_{B_{g}, H_{g}, \thata_{g}} \mathcal{R} = \|H_{g}^{\top} B_{g} - kS\|_{F}^{2} + \gamma \|H_{g} - B_{g}\|_{F}^{2} s.t. H_{f} = \tanh(f_{g}(F^{m};\theta_{g})), B_{g} \in \{-1, 1\}^{k \times n}   $

where $ f_{g}(F^{m};\theta_{g}) $ is the feature encoder with parameters \theta_{g}

### 6.2.5 The Final Objective Function

$R = R_{1} + \alpha R_{2} + \beta (R_{3} + R_{4})  $

### 6.2.6 Training Strategy

Three overall encoding networks:

- R1 visual representation learning network
- R2 semantic binary encoding network
- R3+R4 latent graph ginary embedding network

Concretely, we optimize them in an iterative learning manner, i.e., $R_{1} \rightarrow R_{2} \rightarrow \{R_{3} + R_{4} \} $, which have binary codes $B_{*} (*= v, l, g)  $ and these network parameters, i.e., $\theta_{*} (*=v, l, e, d, g)  $. During the training process, we set the final $B = sgn(H_{v} + \alpha H_{l} + \beta H_{g})  $, and we only have one optimal binary representation B for any branch of encoding network. That means the main optimization process involves network parameter updating and binary code learning

#### 6.2.6.1 Network Parameter Updating

For the network parameters \theta_{*}, we utilize the well-known alternative optimization strategy, that is, updating one parameter when fixing others. Specifically, we employ the stochasitc gradient descent (SGD) with the packpropogation algorithm for network automatic optimization in pytorch

#### 6.2.6.2 Binary Codes Learning

Dye to the binary quadratic problem (BQP) with discrete constraint, we employ the discrete cyclic coordinate descent (DCC) to solve it. It should be noted that these three modules are build on pairwise similarity-preserving loss, which has a similar optimization pattern. We take the latent graph binary embedding network as an example for detailed optimization, which can be simply transferred to the other two modules. When fixing other network parameters, the problem R3+R4 coul dbe degenerated into:

$\min_{B_{g}}  \|H_{g}^{\top} B_{g} - kS \|_{F}^{2} + \gamma \|H_{g} - B_{g}\|_{F}^{2}  + const $

$= \|H_{g}^{\top}B_{g}\|_{F}^{2} - 2tr(B_{g}^{\top}(kH_{g}S + \gamma H_{g})) + const  $

$= \|H_{g}^{\top}B_{g}\|_{F}^{2} - 2tr(B_{g}^{\top} Q) + const   $

$= tr(2(u^{\top}(U')^{\top}B' - q^{\top}) z) + const , s.t. B_{g} \in \{-1, 1 \}^{k \times n}   $

where tr(.) is the trace operator on matrices and const denotes constant parts that are irrelevant to the parameter B. Moreover, Q = kH_{g}S + \gamma H_{g}. u^{\top} is the row of H_{g} and U' denotes the matrix of H_{g} excluding u. Simiarly, let z^{\top} be the row of B, and B' denotes the matrix of B excluding z. q^{\top} is the row of Q, and Q' denotes the matrix of Q excluding q. As such, we can optimize B in a column by column format. The optimal solution of z can be updated by

$z = sgn(q - B'^{\top} U' u)  $

#### Algorithm 6: The DCGH Learning Problem

#### 6.2.7 Rationality of Collaborative Learning

For the rationality of collaborative learning for visual image retreival in this algorithm, there are three explicit meanings that should be clarified:

1. We mainly conduct single-modal image retreival which is an end-to-end learning scheme
2. construct multi-level semantics for collaboratively learning jointly binary code
3. third collaborative learning scheme lies in the usage of common feature learning

## 6.3 Experimental Results

### 6.3.1 Dataset

- CIFAR-10
- MIRFLICKR
- NUS-WIDE
  - due to class imbalance, select samples from top 21 categories with each class having at least 5000 samples

### 6.3.2 Implementation Details

Use a pre-trained AlexNet network, though any other deep visual encoding network could replace it

shallow hasiing methods:

- LSH
- ITQ
- LFH
- SDH

deep hashing methods

- DSDH
- SCRH
- DPSH
- DBDH
- HashNet
- BGDH
- SSDH
- DSH-GANs
- DPQ
- DRLIH
- TSSDH
- DCDH

### 6.3.3 Evaluation Metrics

- mean average precision

### 6.3.4 Comparison Results

Pairwise similarity-preserving hashing methods (such as DPSH, DBSH, HashNet, and SSDH) outperforms piecewise semantic-preserving methods (such as CNNH, NINH, and DSDH)

### 6.3.5 Ablation Study

### 6.3.6 Parameter Sensitivity

DCGH is not sensitive to the values of the hyperparameters [\alpha, \beta, \gamma, \eta ]

### 6.3.7 Visualization

## 6.4 Conclusion

- novel two-stream feature encoding network
- space reconstruction module, fed into a GCN
