# Chapter 13 - OpenBMB: Big Model Systems for Large-Scale Representation Learning

Big pre-trained models (PTMs) have received increasing attention in recent years from academia and industry for their excellent performance on downstream tasks. However, huge computing power and sophisticated technical expertise are required to develop big models, discouraging many institutes and researchers. In order to facilitate the popularization of big models, we introduce OpenBMB, an open-source suite of big models, to break the barriers of computation and expertise of big model applications.

## 13.1 Introduction

- Compute barrier
- Expertise barrier

## 13.2 BMTrain: Efficient Training Toolkit for Big Models

### 13.2.1 Data Parallelism

### 13.2.2 ZeRO Optimization

ZeRO (zero redundancy optimizer) is a strategy that allows efficient training for models with a parameter size far exceeding the capacity of one single GPU

### 13.2.3 Quickstart of BMTrain

## 13.3. OpenPrompt and OpenDelta: Efficient Tuning Toolkit for Big Models

### 13.3.1 Serving Multiple Tasks with a Unified Big Model

### 13.3.2 Quickstart of OpenPrompt

### 13.3.3 QuickStart of OpenDelta

## 13.4 BMCook: Efficient Compression Toolkit for Big Models

quantization, distillation, pruning

### 13.4.1 Model Quantization

Aims to represent the parameters of big models with those lower-bit data types, rather than 32 bit fp. Using lower-bit data types, both memory and compute costs can be significantly reductd.

Two paradigms of quantization:

- post-training quantization (PTQ)
  - aims to directly quantize model parameters after the model learning is complete
  - simple but may bring significant performance degradation
- quantization-aware training (QAT)
  - proposed to alleviate the degradation caused by quantization
  - simlulates the quantization process durin glearning models so that the model parameters can be quantized with the guide of training data

### Model Distillation

Transfer model knowledge from larger teacher models to smaller student models. Conventional distillation methods mainly focus on adding the KL divergence between output results of teacher models and those of student models as an additional training objective

For PTMs, the distillation methods add the MSE loss between student and teacher models' hidden states

### 13.4.3 Model Pruning

- structure pruning
  - aims to remove complete redundant modules such as model layers
- unstructured pruning
  - focuses on removing individual parameters

### 13.4.4 Model MoEfication

Since transformers adopt ReLU as the activation fn of the FFN, bringing a sparse activation phenomenon, can only use the part of FFNs for a specific input without affecting the model performance. MoEficiation is proposed to transform Transformers to the mixture-of-expert (MoE) versions, which can significantly reduce the compute costs of Transformers. Model MoEficiation only selects parts of model parameters for computation, rather than changing or removeing model parameters. Can be viewed as a post-processing technique that can be applied to an already compressed model to further improve efficiency

### 13.4.5 QuickStart of BMCook

## 13.5 BMInf: Efficient Inference Toolkit for BigModels

TensorRT is like ONNX

### 13.5.1 Accelerating Big Model Inference

### 13.5.2 Reducing the Memory Footprint of Big Models

### 13.5.3 QuickStart BMInf

## 13.6 Summary and Further Readings
