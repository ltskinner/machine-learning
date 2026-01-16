# Representation Learning

`Representation learning` is the broader process of learning meaningful features or patterns from raw data, aiming to uncover high-level abstractions applicable across tasks (e.g. learning latent variables from images or text)

## Key Subsets of Representation Learning

| Subset | Description | Example Methods | Example Values | Explanation |
| - | - | - | - | - |
| Sparse Representation Learning | Encodes data into vectors with many zero elements to capture only essential features | LASSO, Sparese, Coding, OMP | [0, 0, 3.2, 0, 0, 0, 1.1, 0] | Most elements are zero, focusing on few non-zero features for a concise and efficient representation |
| Embedding Learning | Maps discrete inputs into a continuous vector spaces, preserving semantic or relational similarity | Word2Vec, Node2Vec, GloVe | [0.21, -1.03, 0.47, 0.89, -0.44] | Values are dense and can include both positive and negative numbers to capture semantic relations |
| Dense Representation Learning | Distributes information across all dimensions in vectors for richer representations | PCA, Autoencoders, Transformers | [1.1, 2.3, -0.7, 0.8, 1.4, 0.2] | All dimensions include meaningful information, with values spanning positive and negative ranges to capture complex patterns |
| Manifold Learning | Learns low-dimensional maniforlds embedded in high-dimensional data | t-SNE, UMAP, Isomap | [0.12, -0.34, 1.45, 0.89, -0.21] | Values reflect distances along latent manifolds, often including negative numbers to maintain relative positions |
| Disentangled Representation Learning | Separates different generative factors of data into independent components | Beta-VAUE, FactorVAE, InfoGAN | [2.1, 0, 0, 3.0, 1.1] | Some factors dominate while others are supressed to disentangle independent sources of variation |
| Metric Learning | Learns representations that preserve specific distance or similarity measures | Saimese Networks, Triplet Loss, Contrastive Loss | [0.5, 1.0, 0.3, 1.2, 0.8] | Values represent distances or similarities, often scaled to align with target metrics |
| Graph Representation Learning | Maps nodes or substructures from graphs into vector spaces, preserving graph properties | Graph Neural Networks, Node2Vec, GraphSAGE | [0.4, 0.7, -0.2, 1.1, 0.9] | Values reflect relational properties of graph nodes, with a mix of positive and negative numbers capturing various features |
| Self-Supervised Representation Learning | Learns representations without labels by creating auxiliary tasks | SlimCLR, BYOL, MoCo | [0.6, -1.1, 0.3, 0.7, -0.8] | Values capture complex latent features, with positive and negative elements representing contrasts learned from auxiliary tasks |

## Sparse Representation Methods

Objective: Represent data as a **sparse combination of basis elements**, where most elements are zero, capturing only the most critical features

Seeks to minimize the number of active components, through L1 or L0 regularization

- Sparse Coding
- Dictionary Learning
- LASSO (Least Absolute Shrinkage and Selection Operator)
- ElasticNet
- OMP (Orthogonal Matching Pursuit)

## Embedding Learning

Objective: Focuses on mapping **discrete or categorical data** (like words, nodes, or users) into a continuous vector space, preserving the **semantic or relational similarities**

`Embedding` contains meaningful, dense representations that capture relationships within the *input space*

Methods:

- Word2Vec
- FastText
- GloVe (Global Vectors for Word Representation)
- Node2Vec
- Doc2Vec
- BERT
- ELMo
- GraphSAGE

## Dense Representation Methods

Objective: Deals with transforming input data **(continuous or unstructured)** into a lower-dimensional, dense feature space, typically for tasks like dimensionality reduction or encoding.

`Encoding` converts data to a new format for processing

Methods:

- Principal Component Analysis (PCA)
- Matrix Factorization (e.g. SVD, NMF)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- Autoencoders
- Variational Autoencoders
- CNNS
- RNN LSTM/GRU
- Transformers
