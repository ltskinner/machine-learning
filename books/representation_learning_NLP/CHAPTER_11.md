# Chapter 11 - Legal Knowledge Representation Learning

The law guarantees the regular funcitoning of the nation and society. in recent years, legal AI, which aims to apply artificial intelligence techniques to perform legal tasks, has received significatn attention. Legal AI can provide a handy reference and convenient legal services for legal professionals and non-specialists, thus benefiting real-word legal practice. Different from general open-domain tasks, legal tasks have a high demand for understanding and applying expert knowledge. Therefore, enhancing models with various legal knowledge is a key issue of legal AI. In this chapter, we summarize the existing knowledge-intensive legal AI approaches regarding knowledge representation, acqusitino, and application. Besides, future directions and ethical considerations are also discussed to promote the development of legal AI

## 11.1 Introduction

Legal AI aims to empower legal tasks with AI techniques and help legal professionals to deal with repetitive and cumbersome mental work.

The core of legan NLP tasks lies in automatic case analysis and case understanding, which requires the morel to understand the legal facts and corresponding knowledge

Unlike the widely used triple knowledge in the open domain, the structure of legal knowledge is complex and diverse.

## 11.2 Typical Tasks and Real-World Applications

Though many tasks have been intensively studied, not all of them have been used in real-world systems due to unsatsifactory performance and ethical considerations.

### Legal Judgement Prediction (LJP)

Aims to predict the judgement results when given the fact description and claims

### Legal Information Retrieval (Legal IR)

Aims to retrieve similar cases, laws, regulations and other informatino for supporting legal case analysis. Essential for both civil and common law. Manual retrieval is time-consuming and labor- intensive

Challanges:

- 1. long text matching
- 2. Diverse definitions of similarity
  - not the literal caluclations
  - like, by each axis similarity could be measured (case value, crime committed, facts)
  - current gen relies on hand crafter rules and expert knowledge. ontologies are popular

### Legal Question Anwering (Legal QA)

Five challenging reasoning types:

- 1. lexical matching
- 2. concept understanding
- 3. Numerical analysis
- 4. Multi-paragraph
- 5. Multi-hop reasoning

### Real-World Application

#### Legal Information Applications

Aim to elelectronically manage and coordinate information and personnel in complex legal services. Help strore and transfer information efficiently and reduce the communication costs between legal practicioners and the public

#### Legal Intelligent Applications

Focus on the understanding, reasoning, and prediction of legal data to help achieve efficient knowledge acquisition and data analaysis. Retrieval and recommendation

## 11.3 Legal Knowledge Representation and Acquisition

### 11.3.1 Legal Textual Knowledge

#### Laws and Regulations

#### Legal Cases

#### Legal Textual Knowledge Representation Learning

- OpenCLaP
- Legal-BERT
- Lawformer

### 11.3.2 Legal Structured Knowledge

Laws and regulatinos usually can be translated into the structure of "if...then...", which can be further converted into logical rules; the facts of a legal case usually consist of several key events; then it can be represented as a structured event timeline

#### Legal Relation Knowledge

triples

#### Legal Event Knowledge

Recognizing the events and the corresponding causal relations between these events is the basis of legal case analysis

Legal events can be regarded as a summarization of legal cases

#### Legal Element Knowledge

Legal elements, aka legal attributes, refer to properties of cases, based on which we can directly make judgement decisions. Usually formalized as a multi-label text classification

#### Legal Logical Knowledge

Laws and regulations are in nature logical statements, and legal case analysis is a process of determining whether defendants violate the logical propositions contained in the law.

Two categories:

- Coarse-grained heurisitc logic
- Fine-grained first-order logic

### 11.3.3 Discussion

#### Tectual Knowledge

Has characteristics:

- 1. High coverage
  - almost all scenarios can find their counterparts inthe textual knowledge
- 2. Updating over time
  - text knowledge growing over time

#### Structured Knolwedge

- 1. Concise and condensed
  - vital information that allows for a quick grasp of the cases specifics
- 2. Interpretable
  - Symbolic representation can provide intermediate interpretations for prediction resutls

#### Towards Model Knowledge

model knowledge or modeladge.

Modeledge refers to knowledge implicitly contained in models. Different from textual and structured knowledge which are explicit human-friendly knowledge, implicit modeledge is machine-friendly and can be easily utilized by AI systems. How to transform textaul and structured legal knowledge into model knowledge is a popular research topic

## 11.4 Knowledge-Guided Legal NLP

Aim to embed explicit textual and structured knowledge into implicit model form

### 11.4.1 Input Augmentation

#### Text Concatenation

concat the knowledge text with the original text and directly feed the concat into the model without architecture modification

#### Embedding Augmentation

fuse knowledge embeddings with original text embeddings

### 11.4.2 Architecture Reformulation

Methods that design model architectures according to heuristic rules in the legal domain.

like output layers and which NN arch to use, etc

#### Inspiration from Human Thought Process

#### Inspiration from Knowledge Structure

### 11.4.3 Objective Regularization

Integrate legal knowledge into the objective functions

#### Regularization on New Targets

Constructing additional supervision signals

#### Regularizaiton on Existing Targets

Attempt to construct extra constraints between different subtasks to improve the consistency across different tasks

### 11.4.4 Parameter Transfer

Methods that train models on source tasks and then transfer the parameters to the target task to achieve knowledge transfer

#### Pre-trained Models

Transfer parameters trained with self-supervised tasks to downstream applications

#### Cross-task Transfer

attempt to train models on some source supervised task and then transfer the model to target tasks

## 11.5 Outlook

Four directions for future research:

- More Data
- More Knowledge
- More Interpretability
- More Intelligence

## 11.6 Ethical Considerations

### Ethical Risks

- Model Bias
- Misuse

### Application Principles

- People-Oriented
  - like, its designed to help folks so needs to provide explanable references
- Human-in-the-Loop
- Transparency

## 11.7 Open Compeititons and Benchmarks

## 11.8 Summary and Futher Readings
