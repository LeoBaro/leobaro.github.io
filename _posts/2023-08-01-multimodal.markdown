---
layout: post
title:  "An introduction on multimodal learning"
date:   2023-08-01 10:00:00 +0200
categories: deep-learning
#toc: true
---
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true
    }
  });
  </script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


<p>Estimated reading time: ?.</p>

Introduction
============
This blog post gives an introduction of the field of Multimodal Machine Learning. In practice these are my notes from the course held by the Carnegie Mellon University: [11-777 Multimodal Machine Learning, 2022 Fall](https://cmu-multicomp-lab.github.io/mmml-course/fall2022/) and the review paper of [P. Liang, et al. (2023)](https://arxiv.org/abs/2209.03430).


Towards multimodality
=====================
Let's start by giving a definition of *modality*:
> :bulb: **A modality is a way something is expressed or perceived**

A multimodal dataset contains information spread over multiple modalities. Let's think about a recording of a public speech for example. Utterances (audio modality) carry most of the information but also the body language of the speaker or facial expressions (video modality) complement and enrich the audio information to convey the final message. Can machine learning models extract knowledge from multiple modalities?

As shown by *Figure 1*, multimodal machine learning has gained particular attention in the scientific community over the last 10 years. 

| <img src="/assets/2023-08-01-multimodal/trend.jpg" alt="multimodal trend" width="800"/> | 
|:--:|                 
| *Figure 1*: the number of publications in the multimodal machine learning field over recent yers. Credits: [Dimensions](https://app.dimensions.ai).|

From 2010, researchers start to use deep learning to exploit multimodal data. In the beginning with Deep Boltzmann Machines ([N. Srivastava, R. Salakhutdinov, 2012](https://papers.nips.cc/paper_files/paper/2012/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)) and then, [K. Xu et al. 2016](https://arxiv.org/abs/1502.03044) exploited the [attention mechanism](https://leobaro.github.io/deep-learning/2023/06/29/transformers.html) to bring multimodal research under the umbrella of computer vision and natural language processing for image captioning. After that moment, research in the multimodal modal domain exploded thanks to several key factors: the availability of new large-scale multimodal datasets and processing hardware but also, advances in computer vision and language processing models enabled the representation of etherogenous data (images, text),into homogeneous embedded vectors with meaningful features. As research on transformers, models optimization and  distributed training continued as well as the availability of new multimodal datasets increased, multimodal deep learning spreaded over multiple fields. Let's just think about robotics, which aim is to develop intelligent autonomous agents capable of integrating and learning from different modalities. But also other real world tasks such as multimodal sentiment and emotion recognition, multimodal QA, multimodal dialog, event recognition, multimedia information retrieval and so on.

The properties of multimodal data
=====================================

We're used to unimodal source of data, which spectrum goes from raw (close to the sensor that captured it) to more abstract representations. For example, a microphone records speech (sound waves) that can be translated to written language (tokens), and analyzed to extract high orders descriptions such as sentiment. This is true also for other modalities such as images, and nothing stop us to integrate all these information togheter, regardless of the abstraction level. Integrating different modalities that are closer to the raw part of the spectrum is more challenging because they tend to be more heteregeneous, with respect to higher level ones. **Heterogeneity** is a key concept in multimodal learning, and it's manifested along the following dimensions:
* **Representation**: it refers on the representation used in the samples space. Examples: set of characters (text) vs matrixes (images) vs nodes and edges (graphs). 
* **Distribution**: different modalities have different distributions and frequencies of samples (the number of objects per image has in general a lower frequency than the number of words in a sentence).
* **Structure**: images have spatial structure while sentences have sequential structure. In addition, the latent space has its own structure that can be different across modalities. 
* **Information**: the total information that can be extracted from a modality. It depends on the abstraction level: an image data can be exploited from raw pixels to object categories.
* **Noise**: occlusion in images, typos in NLP, missing data, and so on.
* **Relevance**: to a specific task that we want to address. 

So, altought different modalities can be very different and they can bring its own unique information, they can be *connected* by sharing complementary information. **Connection** is another key concept in multimodal learning. Which is the form of these connections? We can see it from a bottom-up (statistical) or top-down views (semantic). Statistically speaking, elements that co-occur (cor)relate to each other and this is called *association*. On the semantic side, a *corrispondence* is the presence of the same element (with the same semantic) in both element of different modalities. A stronger connection type is the statistical *dependency*, which identifies casual relations. The equivalent in the semantic world is called *relationship*. TODO: examples. This two properties can help to define multimodal as:

> :bulb: **multimodal: the science of heterogeneous and interconnected data**

In particular, the goal of multimodal learning is to exploit the complementary information coming from multiple connected modalities, through their **Interaction**, the third key concept. It comes into play when the model integrate multiple modalities to perform inference. The result of the interaction could be **new information** (*Response* in *Figure 2*) that can help the model to make better predictions.  

| ![multimodal trend](/assets/2023-08-01-multimodal/inference.jpg)| 
|:--:|                 
| *Figure 2*: |

Regarding on how the response change we can distinguish several types of interactions. We talk about **redundancy** if the modalities give similar answers. In this case the multimodal response can be **enhanced** (higher confidence) or it can be **equivalent** to the unimodal one (no interaction). What if the two modalities give different responses? For example, let's ask the question:

> **is this a dangerous animal?**

🤔🤔🤔

| ![shark fin](/assets/2023-08-01-multimodal/interactions.jpg)| 
|:--:|                 
| *Figure 3*: non-redundant interactions.|

*Figure 3* shows a disagreement between the modalities. We have **dominance** if one modality response takes over the other (e.g. the multimodal response is "*yes*".). We have **independence** if the responses do not interact. **Modulation** is when one modality response enhance or reduce the other one. Finally, the best scenario is when the **emergence** property give birth to new information.   



Multimodal Machine Learning challenges
======================================

The core technical challenges of multimodal machine learning are summarized in *Figure 4*. 

| ![multimodal representation learning challenges](/assets/2023-08-01-multimodal/multimodal-challenges.jpg)| 
|:--:|                 
| *Figure 4*: Representation and Alignment are at the core of every multimodal problem and they are mandatory to perform Reasoning. Reasoning can give the final answer or maybe we could be interested in learning Generation or Transference. Finally, we want Quantification to understand and improve the multimodal models. Credits to [P. Liang, et al. (2023)](https://arxiv.org/abs/2209.03430). |

#### Representation
The first challenge is the building block for most multimodul problem: learning a multimodal **representation** that reflect cross-modal interactions between individual elements across different modalities. There are three main approaches (and sub-challenges) for generating a representation for multimodal data: 
* **Fusion** sub-challenge: information from multiple modalities is integrated to reduce the number of separate representations.
* **Coordination** sub-challenge: the number of representations is equal to the number of modalities but the representations are contextualized in order to incorporate information from multiple modalities.
* **Fission** sub-challenge: the number of representations is greater then the number of modalities and they reflect knowledge about internal multimodal structure. 

#### Alignment
The goal of alignment is to identify cross-modal connections between the elements of multiple modalities. Let's think about a video of a person giving a public speak. There're connections between the gestures and the spoken words. In this context, we need to take into account also the structure of the modalities: spatial, sequential, hiererachical, and so on. There are three sub-challenges here:
* **Discrete Alignment** sub-challenge: it's the problem of finding connections between discrete elements. This can be *local* (find connections between two elements) or *global* (find which elements can be connected by which connections). 
* **Continuos Alignment** sub-challenge: in this case the modalities elements are not discretized a priori such as timeseries (price of a stock over time) or spatio-temporal data (weather images). 
* **Contextualized Representation** sub-challenge: its goal is to detect all modality connections and interactions to learn better representation.

#### Reasoning
The goal of reasoning is to exploit multimodal alignment to perform inference. TODO  

#### Generation
The goal of this challenge is to be able to learn a generative process to produce raw modalities that reflect cross-modal interactions, structure and coherence. The sub-challenges in this context are the following:
* **Summarization** sub-challenge: generate for each modality a summarization that captures the most relevant information.
* **Translation** sub-challenge: transform one modality to another while preserving the information.
* **Creation** sub-challenge: generate novel data of multiple modalities starting from a small set of initial examples or latent conditional variables.

#### Transference
This is the ability of transfer knowledge from one modality to assist another weak modality (lack of annotated data, presence of noise). There're three sub-challenges:
* **Cross-modal transfer** sub-challenge: the idea here is to extend the concept unimodal transfer learning to multiple modalities. This means to train a model on one modality and then fine-tune or condition it to another modality. 
* **Co-learning** sub-challenge: the transfer of information in co-learning is done in two main ways. The first one is done by learning a joint and coordinated representation space using both modalities as input (the second modality is only available during training) and check how the model perform on the first modality during testing. Also the second strategy is based on a joint representation space between two modalities but in this case the model learns a generative process to translate the first modality into the second during inference. 
* **Model induction** sub-challenge: here the models are trained separately one their modality, but then, their predictions are used to psuedo-label new examples that enrich the training sets of the other.

#### Quantification
Quantification aims to address empirical and theoretical studies to improve the robustness, interpretability and reliability of multimodal models. The quantification process in divided into three sub-challenges:
* **Heterogeneity** sub-challenge: understand which modality contributes most to the learning process, characterize biases and noise for each modality.
* **Cross-modal interconnections** sub-challenge: understanding and visualize the multimodal connections (how the modalities are related, what they share?) and  the interactions (how the modalities interact during inference?).
* **Learning process** sub-challenge: the main topics are understanding the generalization capabilities across modalities and tasks, optimizations for efficient training and trade-offs study between performance, robustness and complexity.



Additional resources
====================

#### Links 
*  [Paul Liang's reading list](https://github.com/pliang279/awesome-multimodal-ml#applications-and-datasets)



#### Multimodal Datasets

Affect recognition:
* AVEC
* MOSEI
* Social-IQ
* MELD

Multimodal QA:
* TVQA
* WebQA
* VisualQA

Media description:
* MSCOCO 
* NLVR / NLVR2

Navigation
* Room-2-Room / Room-Across-Room
* Winoground

Dialog
* ...
  
Event recognition
* EPIC-Kitchens

Information retrieval
* IKEA

More datasets:
* https://johnarevalo.github.io/2015/02/18/multimodal-datasets.html