# Chapter 7 - Cross-Modal Representation Learning

Cross-modal representation learning is an essential part of representation learning, which aims to learn semantic representations for different modalities including text, audio, image, and video, etc, and their connections. In this chapter, we introduce the development of cross-modal representation learning from shallow to deep, and from respective to unified in terms of model architectures and learning mechanisms for different modalities and tasks. After that, we review how cross-modal capabilities can contribute to complex real-world applications

## 7.1 Introduction

Modalities are measn if information exchange between human beings. Concretely, each modality is an independent channel of sensory input or output for intelligent systems, main types being: text, audio, image, video, and senseable data

Cross-modal representation learning is an important topic of representatino learning.

To learn cross-modal representations, models typically need to first understand the heterogeneous data from each modality with complex semantic composition. Various deep neural architectures have been developed to incorporate the inductive vias for the heterogeneous data from different modalities.

Difference between modalities:

- basic unit
  - difference between text and other modalities lies in the information density of basic units
  - text has high information density
  - symbols are the basic unit
    - symbols already carry high level semantics
    - images and speech are direct recordings of real-world signals, so challenging to recognize high-level semantics
- Modal structure
  - major difference
  - text and speech exhibit sequential depencency between basic units
  - information in images is spatially presented, leading to invariance in shift and scale in images
  - single frames in videos are spatially presented, and different frames are organized in a sequential structure
  - To account for these structures RNNs and CNNs are good

Models are also challenged with establishing **cross-modal mapping** for cross-modal information alignment and fusion

A unified model simultaneously dealing with different modalities and tasks is beginning to take shape, which can be a promosing foundation and path to realizing general intelligent systems in the future.

## 7.2 Cross-Modal Capabilities

Real-world cross-modal application usually requires a comprehensive mastery of multiple cross-modal capabilities. Three big categories

- Cross-Modal Understanding
  - Required to perform semantic understanding based on given image and query of the tasks
    - qa, grounding text to image regions, semantic relations
  - fine-grained cross-modal alignment and fusion between image regions and text tokens are important to achieve strong cross-modal understanging performance
- Cross-Modal Retrieval
  - given a large candidate set of text and images, and a query from one modality, models are asked to retreive the corresponding data from other modalities
  - due to the large number of retrieval candidates, corss model retrieval methods need to model the holistic semantic relations between data from different modalities in an efficient and scalable way
- Cross-Modal Generation
  - image to text generation, vice versa
  - image-to-text need to establish fine-grained mappings between text generation and image understanding, and achieve a good trade-off between diversity and fidelity in describing the visual content
  - text-to-image presents more challanges on the vision side, such as image generation with high-resolution and good computation efficiency

## 7.3 Shallow Cross-Modal Representation Learning

Early works have investigated fusing cross-modal info in shallow representations, such as word representations

There are weird implicit semantic relatedness - like *eat* and *stare_at* because when eating, tend to stare at the food

### Word Embedding With Visual Context

In most word rep learning, only local context information from text is considered. Global information is often neglected. The image associated with the text can provide such global information for word representation learning. Therefore, some words have proposed to extend word embedding models by using visual information as additional global features

Xu et all approach:

Input of model is an image I and a word sequence describing it (i.e. the image caption). Based on a vanilla CBOW model, when we consider a certain word wt in a sequence, its local text feature is the average of embeddings of words in a window, i.e. {w_t-k, ..., w_t-1, w_t+1, ..., w_t+k}. the visual feature is computed directly from the image I using a CNN and then used as the global feature. The local feature and the global feature are then concatenated into the aggregate context feature h, based on which the word probability is computed:

$P(w_{t}| w_{t-k}, ..., w_{t-1}, w_{t+1}, ..., w_{t+k}; I) = (\exp(w_{t}^{\top} h))/(\sum_{i} \exp(w_{i}^{\top} h)) $

By maximizing the logarithm probability of the target words, the language modeling loss will be back propagated to local text features (i.e. word embeddings), global visual features (i.e. visual encoder) and all other parameters. Despite the simplicity, this accomplishes joint learning for a set of word embeddings, a LM, and the model used for visual encoding

In addition to image pixel feature, the co-occured words in image captions and objects in images can also serve as the additional visual context

### Word Embedding with Visual Target

Besides additional context, visual information can also serve as learning targets to capture fine-grained semantics for word rep learning. For ex, the implicit abstract scene or topic behind the images (e.g. birthday celebration) can serve as discrete visual signals for word rep learning.

A pair of the visual scene and a related word sequence (I, w) is taken as input. At each training step, a window is used upon the word sequence w, forming a subsequence Sw. Based on the context feature (i.e. average word embeddings of Sw), the model produces a probability distribution over the discrete-valued target function g(.) that incorporates visual information. Opimization fn:

$\mathcal{L} = -log P ( g(I)| S_{w})  $

The most important part of the model is the function g(.). INtuitively, g(.) should map the visual scene I into the set {1, 2, ..., k} indicating what kind of abstract scene it is. In practice, it is learned offline using k-means clustering, and each cluster represents the semantics of one kind of visual scene.

## 7.4 Deep Cross-Modal Representation Learning

When dealing with cross-modal tasks, supervised task learning in deep neural architectures can produce deeper cross-modal representations that better fuse and align the cross-modal information

### 7.4.1 Cross-Modal Understanding

Aims to perform semantic recognition and reasoning on a given image and text.

#### Visual Question Answering (VQA)

lots of datasets:

- VQA
- GQA
- VQA-CP
- COCO-QA
- FM-IQA

To address the VWA task, researchers have proposed to adopt attention mechanism for fine-grained vision-language alignment and reasoning, and leverage external knowledge to provide rich context info for QA

##### Attention Mechanism

Image regions are first encoded into feature representations {I1, I2, ..., Ik} via CNN encoders. Then, the attention score alpha_j over the image regions is computed as:

$\alpha_{j} = (W_{1} I_{j} + b_{1})^{\top} (W_{2} q + b_{2}) $

where W1, W2, b1, b2 and trainable parameters and q is the question representation. A larger attention score indicates higher relevance beteen the image region and the question, and larger contribution to the final fused representations and answer prediction.

The question-aware image feature is obtained via a convex combination of the region features based on the normalized attention scores to produce the ansewr

Some questions are only related to some small reasons, which encourages researchers to use stacked attention to further refine the attention distribution for noise filtering. Yang et al further extend the single-layer attention model by stacking multiple layers. The key idea is to gradually filter out noises and pinpoint the regions that are highly relevant to the answer by reasoning through multiple stacked attention layers progressively

The above models only attend to images. Intuitively, questions should also be attended to select informative tokens, and vice versa. Co-attention mechanism between fine-grained image region and text tokens by:

$Z = tanh(Q^{\top} WI) $

Where Z_ij represents the affinity of the ith word and jth region, which is produced from a bilinear operation between the text token feature matrix Q and image region feature matrix I. The co-attention affinity matrix Z is then used to produce the attention scores over text tokens and images regions

##### External Knowledge as Additional Context

Two kinds of knowledge:

- implicit from related text and laguage models
- explicit from knowledge graphs

is proposed to enhance scene understanding through rich attributes, captions, and related text descriptions from knowledge bases. the representation of the rich context information can serve as the initial vector of RNNs, which then further encode the question to produce the answer in a seq2seq fashion. IN this way, the information from attributes and captions and complementary external knowledge from KBs can be utilized. Some other works jointly reason over the descriptions from PTMs, and explicit knowledge from KGs for VQA

#### Visual Relation Detection

Visual relation detection or scene graph generation: the task of detecting objects in an image and understanding the semantic relation between them. Aims to produce scene graphs where nodes correspond to objects and directed edges correspond to visual relations between objects.

Detecting objects are usually conducted by off-the-shelf object detectors, and the key challenge of the task lies in understanding the complex interactions between objects

##### Reasoning with Graph Structures

Aim to pass and fuse semantic information of objects and relations based on the graph structure for complex relational reasoning

Xu et al. propose to iterartively exchange and refine the visual information on the dual graph of objects and relations. Also psoposed to construct a heterogeneous graph consisting of different levels of context information, indlcuding objects, triplets, and region captions, to boost the performance of visual relation detection. Specifically, a graph is constructed to align these three levels of information and perform feature refinement via message passing. During message passing, each node in the graph is associated with a gate to selecte meaningful information and filter out noise from neighboring nodes. By leveraging complementary information from different levels, the features of objects, triplets, and image regions are expected to be mutually improved to improve the performances of the corresopnding tasks.

To further model the inherent dependency of the scene graph generation task, proposed to decompose the task into a mixture of two phases:

- extracting primary relations from the input image first (object pair extractor)
- then completing the scene graph with reasoning (visual relation predictor)

The authors propose a hybrid scene graph generator (HRE) that integrates the two phases in a unified framework. Specifically, HRE employs a simple visual relation detector to identify primary relations in an image, and a differentiable logic programming model which completes the scene graph iteratively. At each time step, the object pair selector considers all object pairs P- whose relations have not been determined, from which the next object pair is chosen to determine the relation. A greedy strategy is adopted which selects the object pair with the highest relation score. The visual relation predictor considers all the object pairs P+ whose relations have been determined and the target object pair to predict the target relation. The prediction result of the target object pair is then added to P+ to benefit future predictions.

##### External Knowledge as Supervision and Regularization

Leveraging language knowledge information is helpful, because language and kgs can provide high-level priors to supervise or regularize visual relation learning. Been shown that language priors from word embeddings can effectively regularize visual relation learning. Notably, can align commonsense kbs with images, which can automatically create large-scale noisy-labeled relation data to provide distant supervision for visual relation learning. Also proposed to alleviate the noise in distant supervision by refining the probabilistic soft relation labels in an iterative fashion. In this way, distantly supervised models can achieve promising performance without any human annotation, and also significantly improve over fully supervised models when human-labeled data is available

IETrans is inspired by visual distant supervision proposed to further generate large-scale fine-grained scene graphs via data transfer. To allevaite the long-tail distributions of visual relations, visual distant supervision technique is adopted to augment relation labels from external unlabeled data. Moreover, given an entity pair, human annotators prefer to label general relations (thus uninformative e.g. *on*) than informative relations (e.g. *riding*) for simplicity, which leads to semantic ambiguity in human annotated data. To address the problem, labels of general relations are transferred to informative ones based on the confusion matrix of relations, which encourages more informative scene graph generation. IETrans can enable large-scale scene graph generation with over 1800 fine-grained relation types

Worth noting: the task of scene graph generation resembles document level relation extraction in many aspects. Both tasks seek to extract structured graphs consisting of entities and relations. also, they need to model the complex dependencies between entities and relations in rich context.

both are tasks that should be researched further.

### 7.4.2

Need to retrieve information across different modalities

Huge number of retrieeval candidates, so retreival requies efficient computation of semantic similarities. This is typically achieved by learning disciminative cross-modal representations from different modalities in a common learning space - two categories:

- real-valued representation-based methods
- binary-valued representation-based methods

#### Real-Valued Representations

encoded into dense vectors, which can be challenged by inferior efficiency, but are more investigated due to their superior performance. In this line of reaearch, real-valued approaches can be further divided into two categories:

##### Weakly Supervised Methods

Cross-modal correlation is learned from the naturally paired cross-modal data. Images on the internet are usually paired with textual captions, which can be easily collected in large-scale to train cross-modal retrieval models

To learn discriminative representations, contrastive-style learning methods are usually adopted to encourage close representations of paired data (i.e. positive samples), and distrinct representations of unpaired data (i.e. negative samples). Many works use a bidirectional hinge loss for an image-caption pair (I, s) as:

$\mathcal{L}(I,s) = \sum_{\hat{s}} max(0, s(I, \hat{s}) - s(I,s) + \gamma) + \sum_{\hat{I}} max(0, s(s, \hat{I}) - s(I, s) + \gamma)  $

where \gamma is a hyper-parameter denoting the margin and Ihat and shat are negative candidates. The objective maximizes the margin of paired and unpaired representations for both image and text as queries. The holistic similarity between images and text can be obtained by aggregating the local similarities between fine-grained image regions and text tokens (e.g. the oaverage of the local similarities)

By summing the loss over all negatives, the negative instances are equally treated. A problem of equal treatment of negatives. A problem of equal treatment of negatives is that the large number of easy negatives can dominate the loss. VSE++ addresses this issue by proposing to mine hard negatives online, by only using the negative that achieves the largest hinge loss in the mini-bathc. Despite the simplicity, the VSE++ achieves significant improvement and is adopted by many following works. VSE-C creates more challenging adversarial negatives by replacing fine-grained concepts in the paired text. By augmenting adversarial instances, VSE-C also alleviates the correlation bias of of concepts in the dataset, and thus improves the robustness of the model. Wu et all establish more fine-grained connections between image and text. The sentence semantics is factorized into a composition of nouns, attribute nouns, and relational triplets, where each component is encouraged to be explicitly aligned to images.

In summary, since only natural image-caption pairs are required, weakly supervised methods can be easily scaled to leverage large amounts of data.

##### Supervised Methods

Investigates supervised learning on labeled image-caption data to learn more discriminative cross-modal representations. A semantic label is given for the contecnt of each image-caption pair (e.g. horse, dog) and the cross modal representations of the same class label are encouraged to be close to each other. The labeled data can provide high-level semantic supervision for cross-modal representation learning, and usually leads to better image-text retrieval performance

Natural unlabeled image-caption pairs can be insufficient, let alone, labeled data. This motivates transfer learning from the domains wehre large amounts of unlabeled/labeled data rea available. A major challenge of transfer learning lies in the domain discrepancy between the source domain and the target domain. To address, the distribution discrepancy between different domains is measured by the max discrepancy (MMD) in the reproduced kernel Hilbert space. By minimizing the MMD loss, the image representations from source and target domains are encouraged to have the same distribution to facilitate knowledge transfer.

In addition to unlabeled image-caption pairs, can further transfer knoweldge from labeled image-caption pairs. Since both domains contain image and text, domain discrepancies comr from both modal-level discrepancies in the same modality, and correlation-level discrepancies in image-text correlation patterns between different domains. An MMD loss is imposed on both modal -level and correlation-level to reduce the domain discrepancies between the source and target domains.

#### Binary-Valued Representations

Information from each modality is encoded into a common Hamming space, which yields better efficiency for both computation and storage. However, due to the limited expressiveness of binary valued representations, the performance of such models could be affected by the loss of valuable information. Therefore, real-valued representation-based methods are more widely investigated

Search is not only place this is useful, can frame many things as image-text retrieval:

- retrieving labels from the category set for image classification
- retrieving sentences from text corpus for image captioning

### 7.4.3 Cross-Modal Generation

Given the information in one modality (e.g. the text description or image of a horse), can we generate its couterpart in another modality? Com-ared with other capabilities, cross-modal generation is more challenging for two reasons:

- 1. A comprehensive understanding of the source modal is required
  - for example in image-to-text, not only objects but also relations have to be detected
- 2. Semantic-preserving natural language sentences or images have to be generated

Image captioning

- early words retrieve related text to produce the caption
- then, encoder-decoder frameworks
- attention and graph are top rn

#### Attention Mechanisms

Visual attention into the encoder-decoder image captioning model. The major bottleneck of vanilla encoder-decoder framework is that rich information from an image is represented in one static representation to produce a complex sentence. In contrast, Xu et al encode each image grid region into representations, and allow the decoder to generate each text token based on a dynamic image representaiton of related regions

Was found that the implicitly learned attention is not guaranteed to be closely related to text tokens. To alleviate the problem, proposed to explicitly supervise the attention distribution over image grids for text tokens. For each object in text, the supervision can come from visual grounding annotations, or textual similarities of detected object tags. Makes attn more explainable, and improves performance since related visual information is better selected. Karpathy makes explicit alignment betwen image regions and sentence fragments before generating a description for the image. The explicit alignment is achieved by maximizing the similarity of image-caption pairs, where the holistic similarity is aggregated by the local alignment between image regions and text fragments.

The attention computed over uniform image grids can split and corrupt high-level semantics (e.g. holistic objects). Propose only doing attention over detected objects.

#### Scene Graphs as Scene Abstractinos

Scene graphs have ben adopted to help describe the complex scene. Scene graphs represent objects and their relations in a graph structure, are benficial for:

- 1. Scene graphs can provide high-level semantics of objects and their interactions for deep understanding of the scene
  - (there is a general consensus that it is visual relations, rather than objects alone, which determine the semantics of the scene)
- 2. Compared with pixel features, the high-level semantics can be better aligned with textual descriptions

some works employ graph Nns over the scene graph. the object information passes along the relation edges based on the graph neural networks. Similar to vanilla attention, the decoder dynamically attends to the scene graph when generating each token. Scene graphs can also be extracted from the paired text during training. In this view, scene graphs can serve as a common intermediate representaiton to transfer the prior from large-scale text to improve image captioning.

Text to image, key problem is image generation. Three categories are:

- VAE-based
- GAN-based
- Diffusion-based

Typical research problems include:

- high-resolution image generation
- stable training of image gen models
- efficient image generation
- conditional image generation

## 7.5 Depe Cross-Modal Pre-training

Key idea is to fully exploid the self-supervised signals from large-scale data to pre-train generic deep cross-modal representations. Pre-training is typically performed to learn cross-modal capabilities based on Transformer architectures and self-supervised tasks, which is largely unified and agnostic to specific tasks. Then, the pre-trained deep cross-modal representations can be tuned to downstream tasks.

The key to cross-modal representation learning is to establish fine-grained connections between cross-modal signals.

A common archtiecture suitable for modleing data is critical. Early works try to fully exploit the inductive bias of each modality. The convolution and pooling are designed to model the scale and shift invariant property images in CNNs, and recurrent computation is devised to model the sequential dependency of text in RNNs. However, their highly specialized design hinders the generalization to other modalities

The stacked self-attention of Transformers reflects a more general principle of information exchange and aggregation. Tranformers also scale better in both data and parameters, where larger data and parameter scale can typically always lead to better performance

### 7.5.1 Input Representations

#### Token-Based Representations

- Images or image patches are presented as discrete tokens
- Tokens can be obtained from clustering, or discrete variatonal auto-encoders
- detailed visual information might be lost in the fixed discrete tokens

#### Object-Based Representations

- Salient objects (e.g. object features, labels, and locations) in an image are used to represent the image content
- objects carry more high-level information, and can be better aligned with concepts in text
- object-based methods rely on external object detectors to obtain input representations, which can be expensive in both annotation and computation
- the background information in images by also be lost

#### Patch-Based Representations

- patch-based methods (e.g. ViT) and their pre-training (MAE) can achieve sota
- significantly faster than object based bc no external detector used
- howver, since objects are not explicitly modeled, can have difficulty in dealing with object position-sensitive tasks
  - to address, some works propose to treat positions as discrete tokens
  - this enables unified explicit modeling of text and positions
  - PEVL retrains the order of discretized positions by an ordering-aware reconstruction objective, which achieves competitive performance on various vision-language tasks

### 7.5.2 Model Architectures

#### Transformer Encoder Architectures

Inspired by BERT

##### Single-Stream Methods

- Image and text input representations are fed into a single Transformer encoder, which jointly encodes cross-modal information with shared parameters
- good, especially for cross-modal understanging
- most widely used vision-language architecture
- however, not easy to perform cross-modal generation and retrieval via ss-methods

##### Two-Stream Methods

Images and text inputs are encoded into a common semantic space by separate unimodal encoders in a similar way to cross-modal retrieval.

- highly efficient, can process web-level data, yield open recognition capabilities
- CLIP is good for zero-shot open-vocabulary image classification
- however, since fine-grained cross-modal interactions cannot be modeled, the perfformance of two-stream models may be limited on complex cross-modal understanding tasks

##### Hybrid Methods

encode image and text first by separate unimodal encoders, then fuse the unimodal representations using a cross-modal encoder. The rationale is that modal-specific information can be better encoded in separate unimodal encoders before cross-modal fusion

#### Transformer Decoder Architectures

- have not been widely used cause usually bidir is required
- howver, convenient for generating images by producing visual tokens in an auto-regressive fashion
- DALL-E models text tokens and image tokens auto-regressively to perform text to image gen

#### Transformer Encoder-decoder Architectures

Image and prefix-text are encoded using encoders, and suffix-text are generated via decoders

This is becoming increasingly popular, since image and text can be well encoded, and the decocer is flexible to deal with various vision language tasks in a unified fashion. Flamingo bridges frozen alrge language PTMs with vision encoders, which produces strong in-context few-shot learning capabilities for vision-language tasks

### 7.5.3 Pre-Training Tasks

Pre-training tasks aim to fully exploit self-supervised learning signals from large scale cross-modal data. The pre-training cross-modal data includes:

- 1. image captionn pairs annotated by humans or crawled from the Internet
- 2. collections of labeled downstream datasets

#### Text-Oriented Tasks

- MLM task
- Left-to-right LM for auto-regressive

#### Image-Oriented

resort to objects, image tokens, high-masking rates

- 1. object based pre-training tasks reconstruct high-level semantics given by object detectors
  - after masking the image regions, the pre-training task can be
    - reconstructing the discrete object labels
    - reconstructing continuous object label distributions
    - regressing the region features
- 2. Image token-based pre-training aim to reconstruct the masked discrete visual tokens
  - however, both objects and visual tokens require external tools to obtain
- 3. Masked patch-based methods directly reconstruct pixels from masked image grid patches
  - MAE find that high masking rates are key to learning high-level semantics from image pixel reconstruction

#### Image-Text-Oriented Tasks

Text and image oriented impose local supervision on text tokens and image regions. In comparisoin, image-text-oriented tasks pay more attention to holistic semantic matching betwen image and text

- 1. popular pre-training task that conducts binary classification of a given image-text pair to judge the matching degree
- 2. Image-text contastrive learning tasks encourage paired image and text representations to be close in a common spenatic space via contrastive learning

Here, mostly two stream or hypbrid architectures are used

### 7.5.4 Adaptation Approaches

General capabilites can be learned suring sspt. During fine tuning, new parameters and objective forms are typically introduced, leading to significant gap between pre-training and downstream tuning.

#### Data-Efficient Prompt Learning

by reformulating downstream tasks into the same form as pre-training, the gap between pre-training and downstream tuning can be maximally mitigated. Therefore, vision-language pre-training models can be efficiently adapted to downstream tasks with only few-shot and even zerp-shot examples

However, can be very difficult to explicitly establish fine-grained cross modal connections via natural language prompts for various position-sensitive tasks, such as:

- visual grounding
- visual commonsense reasoning
- visual relation extraction

CPT explicitly bridges image regions and text via natural color-based coreferential markers. by reformulaing cross-modal tasks into a fill-in-the-blank problem, pre-trained vision LMs can be prompted to achieve strong few-shot and even zerp-shot performance on position-sensitive tasks

#### Parameter-Efficient Prompt Learning

inspired by delta tuning

some works propose to only tune several prompt vecctors, instead of full model parameters, to adapt the pre-trained vision language model. The prompt vectors can be static across different samples or conditional on specific samples. The tunable parameters can also be lightweight adapters. Since only pivotal parameters need to be tuned, parameter-efficient prompt learning methods can better avoid overfitting on few-shot data, and therefore achieve better few-shot performance compared with full parameter fine tuning. However, since new parameters are introduced, it can be difficult for peft methods to deal with zero-shot tasks

## 7.6 Applications

Many real-world applications require multiple cross-modal capabilities

### Cross-Modal Perception

continuously perceieve cross-modal information from both human instructions and the environment

- 1. Human instructions
  - go there
- 2. Environment
  - this cannot be a dog

### Cross-Modal Reasoning

Plan generation

### Cross-Modal Interactions

Decision making
