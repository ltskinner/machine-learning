# Chapter 9 - Knowledge Representation Learning and Knowledge-Guided NLP

Knowledge is an important characteristic of human intelligence and reflects the complexity of human beings. To this end, many efforts have been devoted to organizing various human knowledge to improve the ability of machines in language understanding, such as world knowledge, linguistic knowledge, commonsense knowledge, and domain knowledge. Starting from this chapter, our view turns to representin rich human knowledge and using knowledge representations to improve NLP models. In this chapter, taking world knowledge as an example, we present a general framework of organizing and utilizing knowledge, including knoweldge representaiton learning, knowledge guided NLP, and knowledge acquisition. For linguistic knowledge, commonsense knowledge, and domain knowledge, we will introduce them in detail in subsequent chapters considering their unique knowledge properties

## 9.1 Introduction

Knowledge representation learning aims to encode symbolic knowledge into distributed representations so that knowledge can be more accessible to machines. Then, knowledge-guided NLP is explored to leverage knwoeldge representaitons to improve NLP models. Finally, based on knoweldge-guided models, we can perform knowledge acquisition to extract more knowledge from plain text to enrich existing knowledge systems

focus on world knowledge

## 9.2 Symbolic Knowledge and Model Knowledge

### 9.2.1 Symbolic Knowledge

holy fuck knowledge graphs were only proposed in 2012

KGs arrange the structured multi-relational data of both concrete and abstract entities in the real world, which can be regarded as graph-structured KBs. In addition to describing world knoweldge in conventinoal forms such as strings, the emergence of KGs provides a new tool to organize world knowledge from the persepctive of entities and relations. Since KGs are very suitable for organizing the massive amount of knowledge stored in the Web corpa for faster knowledge retrieval, the construction of KGs has been blooming in recent years and attracted wide attention from academia and industry.

### 9.2.2 Model Knowledge

For grammar rules, expert systems, and even KGs, one of the pain points of symbolic knowledge systems is their weak generalization. In addition, it is also difficult to process symbolic knowledge using the numerical computing operations that machines are good at. Therefore, it becomes important to establish a knowledge framework based on numerical computing and with a strong generalization ability to serve the processing of natural language

Different from symbolic knowledge, which is abstracted by human beings and regarded as **human-friendly** knowledge, the intrinsic nature of statistical learning is to capture the distribution patterns of data from statistics and uses these patterns to abstract implicit knowledge that can be used to solve specific problems. Although such implicit knowledge captured by statistical learning methods may not directly satisfy human intuition, the knowledge is adept at describing correlation information in data and is easy to compute numerically. In other words, this kind of knowledge based on numerical features and continuous probability models is more **machine-friendly**. Considering that the structure of probability models is also a kind of prior knowledge, here we introduce the concept of **model-knowledge** to describe this machine-friendly knoweldge.

### 9.2.3 Integrating Symbolic Knowledge and Model Knowledge

- Symbolic knowledge is suited for reasoning and modeling causality
- Model knowledge is suited for integrating information and modeling correlation

In order to integrate both symbolic and model knowledge, three challanges have to be addressed:

- 1. How to represent knowledge (especially symbolic knowledge) in a machine-friendly form so that current NLP models can utilize the knowledge?
- 2. How to use knowledge representations to guide specific NLP models?
- 3. How to continually acquire knowledge from large-scale plain text instead of handcrafted efforts?

## 9.3 Knowledge Representation Learning

As symbolic systems scale, face two challanges:

- data sparsity
- computational inefficiency

these challenges indicate that symbolic systems are not an inherently machine-friendly form of knowledge organization. Specifically, data sparsity is a common problem in many fields. Ex, when we use KGs to describe general world knowledge, the number of entities (nodes) in the KG can be enormous, while the number of relations (edges) in KGs is typically few, i.e. there are often no relations between two randomly selected entities in the real world

To solve the above problems, distributed knowledge representations are introduced, i.e. lwo-dimensional continuous embeddings are used to represent symbolic knowledge. The sparsity problem is alleviated owing to these distributed representatinos, and the computational efficiency is also improved.

We take KGs that orgnaize rich world knoweldge as an example

G = (E, R, T) to denote a KG, in which E = {e1, e1, ...} is the entity set, R = {r1, r2, ...} is the relation set, and T is the fact set. We use h, t \in E to represent head and tail entities, and **h**, **t** to represent their entity embeddings. A triplet $<h, r, t> \in T $ is a factual record, where h, t are entities and r is the relation between h and t

Given a triplet $<h, r, t> $, a score function f(h, r, t) is used by knowledge representaiton learning methods to measure whether <h, r, t> is a fact or fallacy. Generally, the larger the value of f(h, r, t), the higher probability that <h, r, t> is true. Based on f(h, r, t), knowledge representation can be learned with

$argmin_{\theta} \sum_{<h, r, t> \in T} \sum_{<\tilde{h}, \tilde{r}, \tilde{t} \in \tilde{T}>} max \{ 0, f(\tilde{h}, \tilde{r}, \tilde{t}) | \gamma - f(h, r, t) \}$

where \theta is the learnable embeddings of entities and relations, <h, r, t> indicates positive facts (i.e. triplets in T), and tilde <h, r, t> indicates negative facts (triplets that do not exit in KGs), \gamma > 0 is a hyperparamter used as a margin. A givver \gamma means to learn a wider gap between f(hrt) and tilde f(hrt). Considering there are no explicit triples in KGs, T~ is usually defined as:

$\tilde{T} = \{<\tilde{h}, r, t>| \tilde{h} \in E, <h, r, t> \in T \} \cup \{<h, \tilde{r}, t>|\tilde{r} \in R, <h, r, t> \in T \} \cup \{<h, r, \tilde{t}>|\tilde{t} \in E, <h, r, t> \in T  \} - T $

Which means T~ is build by corrupting the entities and relations of the triplets in T. Different from the margin-based loss function, some methods apply a likelihood-based loss function to learn knowledge representaitons as

$argmin_{\theta} \sum_{<h, r, t> \in T} log [1 + \exp(- f(h, r, t))] + \sum_{<\tilde{h}, \tilde{r}, \tilde{t}> \in \tilde{T}} log[1 + \exp(f(\tilde{h}, \tilde{r}, \tilde{t}))] $

Some typical knoweldge representation learning methods as well as their score functions

### 9.3.1 Linear Representation

Linear representaiton methods formalize relations as linear transformations between entities, which is a simple and basic way to learn knowledge representations

#### Structured Embeddings (SE)

typical. all SE entities are embedded into a d- dimensional space. SE designs two relation-specific matrices Mr1, and Mr,2 \in R^dxd for each relation r, which are both used to transform the embeddings of entities. Score fn defined as:

$f(h, r, t) = - \| M_{r, 1} h - M_{r,2} t  \| $

where ||.|| is the vector norm. The assumption of SE is that the head and tail embeddings should be as close as possible after being transformed into a relation-specific space. Therefore, SE uses the margin-based loss fn to learn representations

#### Semantic Matching Energy (SME)

builds more complex linear transformations than SE. given a triplet <h, r, t>, h and r are combined using a projection function to get a new embedding I_h,r. Similarly, given t and r, we can get I_t,r. Then, a pointwise multiplication function is applied on I_h,r and I_t,r to get the score of this triplet. SME introduces different projection functions to build f(h, r, t), one in linear form:

$f(h, r, t) = I_{h,r}^{\top} I_{t,r} , I_{h, r} = M_{1}h + M_{2}r + b_{1}, I_{t,r} = M_{3}t + M_{4} r + b_{2}     $

and the other is in the bilinear form

$f(h, r, t) = I_{h, r}^{\top} I_{t, r}, I_{h, r} = (M_{1} h \cdot M_{2}r) + b_{1}, I_{t, r} = (M_{3} t \cdot M_{4}r) + b_{2}   $

where \cdot si the elementwise (Hadamard) product. M1, M2, M3, M4 are learnable transformation matrices, and b1, and b2 are learnable bias vectors. This is suitable for dealing with the score functions build with vector norm operations, while the likelihood-based loss fn is more usually used to process the score functions build with inner product operations. Since SME uses the inner product operation to build its score function, the likelihood-based loss fn is thus used to learn representations

#### Latent Factor Model (LFM)

Aims to model large KGs based on a bilinear structure. By modeling entities as embeddings and relations as matrices, the score function of LFM is defined as:

$f(h, r, t) = h^{\top} M_{r} t  $

where matrix Mr is the representaiton of the relation r. Similar to SME, LFM adopts the likelihood-based loss fn to learn repreesntations. DistMult restricts Mr to diagonal matrix and reduces parameter size and computational complexity and achieves better performance

#### RESCAL

Based on matrix factorization. By modeling the entities as embeddings and relations as matrices, RESCAL adopts a score function the same to LFM. However, RESCAL employs neither the margin-based nor the likelihood-based loss fn to learn knowledge representations instead, a three way tensor $\overrightarrow{X} \in \mathbb{R}^{|\mathcal{E}| \times |\mathcal{E}| \times |\mathcal{R}|    } $ is adopted. In the tensor X, to modes repsectively stand for head and tail entities, while the thirdm ode stands for relations. The entries of X are determined by the existence of the corresopnding triplet facts. That is, $\overrightarrow{X}_{ijk} = 1  $ if the triplet <i-th entitie, kth-relation, jth-entity> exists in the training set, and otherwise $\overrightarrow{X}_{ijk} = 0. $ To capture the inherent structure of all triplets, given $\overrightarrow{X} = \{X_{1, ..., X_{|\mathcal{R}|}}  \} $ for each slice $X_{n} = \overrightarrow{X}_{[:,:, n]}  $. RESCAL assumes the following factorization for Xn holds

$Xn \approx EM_{r_{n}} E^{\top}  $

where $E \in \mathbb{R}^{|\mathcal{E}| \times d}  $ stands for the d-dimensional entity representations of all entities and $M_{r_{n}} \in \mathbb{R}^{d \times d}   $ represents the interactions between entities speicfic to the n-th relation r_n. Following this tensor factorization assumpion, the learning objective of RESCAL is defined as

$argmin_{E,M} (1/2) ( \sum_{n=1}^{|\mathcal{R}|} \| X_{n} - EM_{r_{n}} E^{\top}  \|_{F}^{2}  ) + (1/2) \lambda ( \|E \|_{F}^{2} + \sum_{n=1}^{|\mathcal{R}|} \|M_{r_{n}} \|_{F}^{2}   )      $

Where M = {M_r1, M_r2, ..., M_r|R|} is the collection of all relation matrices, ||.||_F is the Frobenius vector norm, and \lambda is a hyper-parameter to contro lthe second regularization term.

#### HOlographic Embeddings (HolE)

Proposed as an enhanced version of RESCAL. RESCAL works well with multi-relational data but suffers from a high computational complexity. HolE employs an operation named circular correlation to generate representations. The circular correlation operation $\star : \mathbb{R}^{d} \times \mathbb{R}^{d} \rightarrow \mathbb{R}^{d} $ between two entities h and t is

$[h \star t]_{k} = \sum_{i=1}^{d} [h]_{i} [t]_{(k+i) mod d+1}  $

where [.]i means the ith vector element. The score function is defined as

$f(h, r, t) = -r^{\top} (h \star t)  $

HolE adopts the likelihoood=based loss function to learn representations

The circular correlation operation brings several advantages. First, is noncommutative (i.e. h \star t \ne t \star h) which makes it capable of modeling asymmetric relations in KGs. Second, the circular correlation operation has a lower computational complexity compared to the tensor product operation in RESDCAL. Moreover, the circular correlation operation could be further accelerated with the help of fast Fourier transform (FFT), which is formalized as

$h \star t = \mathcal{F}^{-1} \bar{(\mathcal{F}(h)} \cdot \mathcal{F} (t))  $

Where F(.) and F(.)^-1 represent the FFt operation and its inverse operation. Respectively, F(.)bar denotes the complex conjugate of F(.) and \cdot stands for th element-wise (Hadamard) product. Due to the FFt operation, the computational complexity of the circular correlation operation is O(d log d) which is much lower than that of the tensor product operation.
