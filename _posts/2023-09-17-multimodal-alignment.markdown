---
layout: post
title:  "An introduction on Multimodal Learning (Part 2 - Multimodal Alignment)"
date:   2023-09-18 10:00:00 +0200
categories: deep-learning multi-modal-learning
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


<p>Estimated reading time: ? minutes.</p>

Introduction
============
This is the second blog post of a series about *Multimodal Machine Learning*. In practice, these are my notes from the course held by Carnegie Mellon University: [11-777 Multimodal Machine Learning, 2022 Fall](https://cmu-multicomp-lab.github.io/mmml-course/fall2022/) and the review paper of [P. Liang, et al. (2023)](https://arxiv.org/abs/2209.03430). 
In this post, I am going to give you an introduction to the *Multimodal Machine Learning* field and in particular to the second big research sub-topic called *Multimodal Alignment*. 

If you missed part one on *Multimodal Representation*, [here's the link](https://leobaro.github.io/deep-learning/2023/08/01/multimodal-representation.html).


## Alignment
The second big research topic in multimodal learning is *alignment*. Alignment refers to the process of identifying and modeling cross-modal connections between all elements of multiple modalities. This ensures that the representations of different modalities are compatible or synchronized in the same space so that they can be effectively compared, fused, or used jointly in machine learning tasks.

The three major sub-challenges of alignment are shown in Fig.1. Discrete alignment involves discrete elements such as the list of objects in an image. Continuous alignment involves continuous signals in which it's not straightforward to segment individual objects. Contextualized Representation is about learning representation over structured data and it involves alignment + representation. In this case, we're not aligning single elements but we consider the context: the meaning of a word depending on the other words, the meaning of an object depending on the other object or the meaning of an object depending on the other words and objects.

|                            ![multimodal alignment](/assets/2023-09-17-multimodal-alignment/alignment.jpg)                             |
| :-----------------------------------------------------------------------------------------------------------------------------------: |
| *Figure 1*: different sub-challenges of multimodal alignment. Credits to: [P. Liang, et al. (2023)](https://arxiv.org/abs/2209.03430) |

In which sense two elements are connected? In a statistical they can co-occur or correlate. In this case the connection is called *association*. If we instead consider the semantic of the elements, they are connected in the sense they have similar meaning (grounding). 

|   ![multimodal alignment](/assets/2023-09-17-multimodal-alignment/connections.jpg)    |
| :-----------------------------------------------------------------------------------: |
| *Figure 2*: A connection between two multimodal elements can be an *association* (statistical bottom-up view) or a *correspondence* (top-down semantic view). Credits to: [P. Liang, et al. (2023)](https://arxiv.org/abs/2209.03430) |


















Right after BERT, VisualBERT by [Harold et al. (2019)](https://arxiv.org/abs/1908.03557), has been proposed as one of the first multi-modal transformer architectures, capable of processing natural language and visual data. In VisualBERT, word tokens and patch image embeddings are concatenated together (early-fusion) in a weakly aligned fashion: the text caption associated with the images is not explicitly related with the patches and the goal of the transformer is to align not only the words but also the visual objects. 

Another approach, ViLBERT by [Jiasen et al. 2019]() uses cross-attention between the image embeddings and the contextualized text embeddings. 

[Sun et al. 2019]() 
[Miech et al. 2020](end-to-end learning of visual representation from uncurated instructional videos)
[Zhu and Yang 2020](ActBert) 

All these approaches use a CNN. A new generation of pure vision transformer 

