---
language: en
license: mit
library_name: keras
tags:
- deep clustering
- semi-supervision
- image classification
datasets: fashion_mnist
metrics: accuracy
---

# Model card for SemiSupervised DCEC

## Table of Contents
- [Model Details](#model-details)
- [Intended Use](#intended-use)
- [Factors](#factors)
- [Metrics](#metrics)
- [Data](#data)
- [Training Procedure](#training-procedure)
- [Quantitative Analyses](#quantitative-analyses)
- [Caveats and Recommendations](#caveats-and-recommendations)

## Model Details

An ML model for performing image classification using a small amount of supervision. It was developed by Nicola Fiorentino in September 2022, starting from the previous work _Deep clustering with convolutional autoencoders_ of Xifeng Guo, Xinwang Liu, En Zhu and Jianping Yin. A complete description of the model is provided in the following [paper](../reports/paper.pdf).

### Motivations

Traditional clustering algorithms exhibit low performance when dealing with high dimensional inputs. In this situation, neural networks can be used to learn features which are suitable for clustering purposes. This approach, called deep clustering, aims to map high dimensional data to a smaller space and simultaneously find a set of clusters in this embedded space. Since clustering is an unsupervised technique, it has the advantage of not requiring human annotations. However, it is not forced to learn useful features for classification tasks. A semi-supervised approach allows to overcome this limitation.

### Architecture Overview

The model consists of a convolutional autoencoder with a clustering layer connected to the embedded layer. The autoencoder allows to learn useful features from images. The clustering layer computes the probability of assigning embedded features to clusters. By providing the labels of some instances as ground truth, the model guarantees a consistency between learned features and background knowledge. A graphical representation of the architecture is provided below.

![](../reports/figures/model_architecture.png)

### Architecture Details

The structure of the encoder is conv $^5_{32}$ → conv $^5_{64}$ → conv $^3_{128}$, where conv $^k_n$ denotes a convolutional layer with $n$ filters, kernel size of $k×k$ and stride 2 as default. The encoder is followed by a fully connected layer with 10 units. The decoder mirrors the encoder structure.

## Intended Use

The model can be used to perform image classification whenever the availability of labeled data is scarce. It can be adopted by those users who cannot rely on manual annotation of the entire dataset for reasons of cost or time.

## Factors

Model performance may vary depending on the complexity of the image dataset. In addition, performance is affected by the amount of supervision available.

## Metrics

Since the model leverages some supervision, accuracy can be used for performance evaluation. The adopted dataset is perfectly balanced, so there is no risk of obtaining misleading results.

## Data

The model is trained on the Fashion MNIST dataset, containing 70,000 images equally distributed in 10 different classes. Instances are 28×28 grayscale images depicting Zalando articles. More information can be found in the [dataset card](../data/README.md).

Pixel values are normalized to range from 0 to 1. To provide some supervision, we assumed that 5% of the available instances have been manually labeled. Such instances are random sampled from the original dataset by using a stratified approach.

## Training Procedure

The training phase consists of two steps. First, the convolutional autoencoder is pretrained to learn the initial embedded features. Secondly, feature learning and clustering are jointly optimized by minimizing the sum of a reconstruction loss and a clustering loss. The training process stops if the change of label assignments between two consecutive updates is less than a given threshold or the maximum number of iterations is reached.

The autoencoder is pretrained for 200 epochs with a batch size of 256. The convergence threshold is set to 0.1% with a maximum number of iterations of 20,000.

## Quantitative Analyses

When no supervision is provided, the best possible mapping between clusters and class labels results in an accuracy of 0.58. If we assume that 5% of the instances have been manually labeled, the accuracy computed on the unlabeled instances reaches 0.76.

## Caveats and Recommendations

Since labeled data is expensive to acquire, the model allows to reach a fair trade-off between costs and performance. However, to extensively verify its effectiveness, more complex image datasets should be considered.