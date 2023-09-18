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


## Alignment
The second big research topic in multimodal learning is Alignment. Alignment refers to the process of ensuring that the representations of different modalities are compatible or synchronized in the same space so that they can be effectively compared, fused, or used jointly in machine learning tasks.



Right after BERT, VisualBERT by [Harold et al. (2019)](https://arxiv.org/abs/1908.03557), has been proposed as one of the first multi-modal transformer architectures, capable of processing natural language and visual data. In VisualBERT, word tokens and patch image embeddings are concatenated together (early-fusion) in a weakly aligned fashion: the text caption associated with the images is not explicitly related with the patches and the goal of the transformer is to align not only the words but also the visual objects. 

Another approach, ViLBERT by [Jiasen et al. 2019]() uses cross-attention between the image embeddings and the contextualized text embeddings. 

[Sun et al. 2019]() 
[Miech et al. 2020](end-to-end learning of visual representation from uncurated instructional videos)
[Zhu and Yang 2020](ActBert) 

All these approaches use a CNN. A new generation of pure vision transformer 

