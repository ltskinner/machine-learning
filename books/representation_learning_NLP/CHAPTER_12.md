# Chapter 12 - Biomedical Knowledge Representation Learning

As a sublect closely related to our life and understanding of the world, biomedicine keeps drawing much attention from researchers in recent years. To help improve the efficiency of people and accelerate the progress of this subject, AI techniques especially NLP methods are widely adopted in biomedical research. In this chapter, with biomedical knowledge at the core, we launch a discussion on knowledge representation and acquistion as well as biomedical knowledge-guided NLP tasks and explain them in detail with practical scenarios. We also discuss current research progress and several future directions.

## 12.1 Introduction

### 12.1.1 Perspectives for Biomediadl NLP

use NLP to improve human experts efficiency by mining useful information and finding potentital implicit laws automaticall, and this is closely related to two branches of biology: computational biology and bioinformatics

Computational biology emphaizes solving biological problems with the favor of computer science.

Bioinformatics studies the collection, processing, storage, dissemination, analysis, and interpretation of biological information. research primarily focused on genomics and proteomics

- NLP tasks in biomedical domain text
- NLP methods for biomedical materials (like data mining genetic sequences)

### 12.1.3 Role of Knowledge in Biomedical NLP

Emphasize the **knowledge representation, knowledge acquisition, and knowledge-guided NLP**

## 12.2 Biomedical Knowledge Representation Acquisition

### 12.2.1 Biomedical Knowledge from Natural Language

What is special about biomedical texts is that we have to achieve a deep comprehension of the key biomedical terms

#### Term-Oriented Biomedical Knowledge

Not solved.

##### Biomedical Term Representations

##### Biomedical Term Knowledge Acquisition

Many subtasks: NER, classification, linking

#### Language-Described Biomedical Knowledge

### 12.2.2 Biomedical Knowledge from Biomedical Language Materials

#### Genetic Language

- Basic Tasks for Genetic Sequence Processing
- Features of Genetic Language
- Genetic Language Tokenization
- Genetic Sequence Reprsentation

#### Protein Language

special language with low readability and a small vocabulary (like DNA)

- Basic Tasks for Protien Sequence Processing
- Landmark Work for Protein Spatial Structure Analysis
  - AlphaFold from DeepMind

#### Chemical Langauge

- Early Fashions for Chemical Substance Representation
- Graph Representations
  - Graph Transformer is popular
- Linear Text and Other Representations

## 12.3 Knowledge-Guided Biomedical NLP

### 12.3.1 Input Augmentation

Knowledge may also come from linguistic rules, experimental results, and other unstrcutured records. The problem for input augmentation is to help select helpful information, encode, and fuse it with the processing input

#### Encoding Knowledge Graph

- 1. Improving word embeddings with the help of KGs
- 2. Augmenting th einputs with knowledge
- 3. Mounting the knowledge by extra modules

#### Encoding Other Information

### 12.3.2 Architecture Reformulation

Human prior knowledge is sometimes reflected in the design of model architectures, as we have mentioned in the representation learning of biomedical data.

### 12.3.3 Objective Regularization

Formalizing new tasks from extra knowledge can change the optimization target of the modela nd guide the model to finish the target task better

Usually, we conduct multi-task training in the downstream *adaptation* period. Some researchers also exploer objective regularization in the *pre-training* period

#### Multi-task Adaptation

#### Multi-task Pre-training

### 12.3.4 Parameter Transfer

#### Cross-Domain Transfer

#### Cross-Task Transfer

## 12.4 Typical Applications

### 12.4.1 Literatyre Processing

#### Literature Screening

In our usual academic search process, we first screen the mass of litarature returned by our search engine. We require the information retrieval model to return a relevance score ranking according to the query conditions

#### Information Extraction

#### Result Analysis and Question Answering

### 12.4.2 Retrosynthetic Prediction

- Chemical Reaction Classification
- Single-Step Reaction Prediction
- Multi-step Reaction Prediction

### 12.4.3 Diagnosis Assistance

## 12.5 Advanced Topics

- Knowledgeable Warm Start
  - in recommendation algs there is the "cold-start problem", which describes impaired performance when lacking user history
- Cross-Modal Knowledge Processing
  - design tokenozer to utilize different structures uniformly
- Interpretability, Privacy, and Ease of Use

## 12.6 Summary and Futher Readings
