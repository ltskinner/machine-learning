# Chapter 14 - Ten Key Problems of Pre-trained Models: An Outlook of Representation Learning

The aforementioned representation learning methods have shown their effectiveness in various NLP scenarios and tasks. Large-scale pre-trained language models (i.e., big models) are the sota of representation learning for NLP and beyond. With the rapid growth of data scale and development of computation devices, big models bring us to a new era of AI and NLP. Standing on the new giants of big models, there are many new challenges and opportunities for representation learning. In the last chapter, we will provide a 2023 outlok for the future directions of representation learning techniques for NLP by summarizing ten key open problems for pre-trained models.

## 14.1 Pre-trained Models: New Era of Representation Learning

### Unified Architecture of Representation Learning

like, lots of different architectures. CNN, RNN, Transformer, etc. Typically the best one gets adopted by each modality. This dominant architecture unifies modalities.

Transformers may be it, but also may not.

### Unified Model Capability for Multiple Tasks

The unified model capability revealed by big pre-trained models makes them completely different from conventional machine learning approaches including statistical learning and deep learning. It requires the exploration of a new theoretical foundaiton and efficient optimization methods conditioned on pre-trainged big models

## 14.2 Ten Key Problems of Pre-trained Models

### 14.2.1 P1: Theoretical Foundation of Pre-trained Models

Self-contained and rogorous mathematical theories could effecaciously guide the ameliorations(improvement) of neural structures, pre-training objetives, and adaptations of PTMs and even pave the road to more powerful AI

"we hold the mindset of seekers"

#### What is the Appropriate Mathematical Description of the Generalization Capability?

Some argue that the probability theory framework that is widely used to describe generative models is intractable in the situation of capturing the correlations of high-dimensional variables. Under this circumstance, other mathematical tools need to be adopted and evaluated to interpret the utilities of NNs.

Recent progress in **geometric deep learning** elaborates different types of NNs through the lense of symmetry and invariance.

#### Why Does Pre-training Bring the Generalization to Downstream Tasks?

#### How Are the Model Capacity and Capabilities Related?

interesting that models just get better as they get bigger

### 14.2.2 P2: Next Generation Model Architecture

what could the next chen architecture for NNs be?

form historical perspective, we find that many of the earlier breakthroughs were inspired by other disciplines

#### Dynamical Systems Inspired Architectures

A dynamical system is a systems whose state is evolving over time. some advantages for NNs

- 1. GPU memory efficiency
- 2. Adaptive computational time
  - ideally, models should spend less time on simple samples and more time on complex ones
  - current architectures treat each instance equally

#### Geometry Inspired Architectures

Humans live in a Euclidian world. Therefore, we natually accept the assumption that the geometry of the neural networks should also be Euclidian

Considering the non-Euclidian geometries in the neural networks brings several benefits:

- 1. Greater capability in modeling structured features both theoretically and empirically.
  - many real-life graphs are known to be tree like
  - however even ehwn the dimension of the euclidian space is unbounded, tree structures still cannot be embedded with arbitrarily low distortion, i.e. some information will always be lost
  - However it can be easily achieved in a two-dimensional hyperbolic space, which is a non-euclidian hyperspace
  - in practice, there have also been a lot of graph-related works demonstrating the effectiveness of low-dimensional hyperbolic models
- 3. Combinability with the dynamical system
  - From the perspective of geometry, the layers in neural networks can be seen as transformations on the coordinate representation of the data maniforld
  - Has the potential to provide a more intuitive understanding of how NNs gradually transform the data from input features to features taht can eventually be used for classification

#### Neuroscience Inspired Architectures

Inpsired by sparsity of neuronal interconnections in human brain, researchers have experimented with designing neural networks with sparsity from two dimensions:

- spatial sparsity
- temporal sparsity

Mixture of Experts divides each layer into several experts and includes a router to route every input into only a few experts. MoE models have shown to reach SOTA on several benchmarks with fewer computational costs.

Human Neurons do not transmit signals every time step. Spike NNNs (SNNs) mimic the behavior of information propagation between neurons interconnected by synapses. Neuromorphic computing

### 14.2.3 P3: High-Performance Computing of Big Models

#### High-Performance Computational Infrastructure

#### High-Performance Algorithms

#### High Performance Application

### 14.2.4 P4: Effective and Efficient Adaptation

#### Computationally Practical Adaptation

Dark clouds over delta tuning. It is difficult to assess the optimal amount of tunable parameters, and the convergence of delta tuning is relatively slower than full parameter fint tuning.

#### Task-Wise Effective Adaptation

Empirical evidence shows that inserting additional contexts i.e. prompts and transferring downstream tasks to pre-training tasks could substantially shrink the gap and yield promising performance, especially in low-data regimes.

#### Advanced Adaptation with Complex Reasoning

### 14.2.5 P5: Controllable Generation with Pre-trained Models

Generating data distribution is a long-standing challenge for the machine learning community due to its inherent high dimensionality and intractability.

Consider all modalities in this case

#### A Unified Framework for Diverse Controls

- prompt-based methods
  - either by injecting a control code or continuous parameters, we can leverage PTM with diverse controls.
  - the major drawback is that prompt-based methods usually have coarser control granularity of smaller control power, thus incapable of handling hard constraints like copying a span of text
- distribution modification methods

#### Compositionality of Controls

In addition to the diversity, controllable generation is also expected to be multi-dimensional and multi-grained to allow more intricate combinations of controls.

Compositionality like semantic compositionality

#### Well-Recognized Evaluation Benchmark

Difficulties in establishing a benchmark for controlled generation:

- 1. Human language is rich in expressions, and the same meaning can take on many nuances
- 2. Control requirements are intractable and diverse
  - ex. topic satisfaction or emotional tendencies are difficult to measure quantitatively
- 3. Evaluation should take into account potential degraded factors such as quality and efficiency

### 14.2.6 P6: Safe and Ethical Big Models

#### Evaluating Safety and Ethical Levels

#### Governing Big Models

#### Building Inherently Safe Models

To achieve human-level robustness, the models:

- 1. know what they know and do not know (i.e. calibrated)
- 2. learn from mistakes and correct themselves

Two directions:

- 1. Incorporating knowledge
- 2. Cognitive learning

### 14.2.7 P7: Cross-Modal Computation

#### Big Cross-Modal Models with Efficient Pre-training and Adaptation

Some works have explored more efficient pre-training methods by reusing unimodal models that have been well pre-trained and focusing on connecting PTMs from different modalities

#### More Unified Representation with More Modalities

Traditional cross-modal works typically design highly specialized model architectures to maximally exploit the inductive bias of modalities and tasks.

#### Embodied Cross-Modal Reasoning and Cognition

Obstacles for complex reasoning and cognition:

- 1. For modalities with low information density (image and audio)
- 2. text with high information density, it can be natural to perform complex reasoning based on the abstract symbolic tokens
  - many AI researchers believe that true recognition capability cannot arise from learning only from text

A more promising direction will be an embodied cross-modal reasoning model. Concrete signals from other modalities can be effectively aggregate into a text-based central unit for high-level semantic reasoning.

### 14.2.8 P8: Cognitive Learning

An essential measurement of general AI is whether neural models can correctly percieve, understand, and interact with the world, i.e. the cognitive ability. A prototype of general intelligence can be viewed as the capability of manipulating existing tools (e.g., search engines, databases, web-side mail systems, etc), conducting cognitive planning with complex reasoning, and interactig with the real world to acquire and organize information/knowledge.

#### Understanding Human Instructions and Interacting with Tools

How could PTMs better understand users instructions and interact with existing tools to complete a specific task? Fulfilling this goal requires preicesly:

- 1. Mapping the natural language instructions in the semantic space to the cognitive space of the model
- 2. mapping the cognitive ability of the model to the action space of the tool, so as to correctly perform the operation and use the tool

The realization of this goal has profound practical significance:

- 1. for onee thing, an ideal next generation of human-computer interaction will be based on natural language rather than a GUI
- 2. the bar of utilizing complex tools will be greatly lowered

#### Cognitive Planning and Reasoning for Complex Tasks

#### Integrating Information from the Real World

### 14.2.9 P9: Innovative Applications of Big Models

Classes:

- New Breakthroughs
- New Scenarios

Prerequisites that an application scenario can turn to big model systems for help:

- Plenty of Domain Data
- Documented Domain Knowledge

### 14.2.10 P10: Big Model Systems Accessible to Users

Historical successful cases like Databse Management systems (DBMS) are a pattern. Propose building a unified management system of big models, i.e., big model systems (BMS)

Design principles to consider:

- Data Form and Operation Abstraction of Big Models
- Efficient Computation and Management of Big Models

## 14.3 Summary

"Let's work together on these exciting topics to contribute novel techniques and applications of AI in the future"

man, great book thank you
