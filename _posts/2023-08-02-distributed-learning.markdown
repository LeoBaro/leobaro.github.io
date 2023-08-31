---
layout: post
title:  ""
date:   2023-08-02 21:00:00 +0200
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
In recent times it has been shown that larger model training improve model quality. The development of neural architectures based on [Transformers](https://leobaro.github.io/deep-learning/2023/06/29/transformers.html) let the number of parameters to drammaticaly increase, from 340M of BERT to 530 billion (!) of the largest (at the time of writing) Megatron-Turing NLG model [D. Narayanan et al. (2021)](https://arxiv.org/abs/2104.04473).
| ![number of parameters](/assets/2023-08-02-2023-08-02-distributed-training/model_size.jpg)| 
|:--:|                 
| *Figure 1*: the dramatic increase in the model size over recent years. Credits: [Large Language Models: A New Moore's Law?](https://huggingface.co/blog/large-language-models).|
Training such big models is challenging for two main reasons: first, large model does not fit in GPU memory even if a multiple-GPU server node is avaialable and second, the computation complexity of the training process is too high, requiring too much time to make the model reaching the desired accuracy. The only way to cope with this is to enable a **large scale training infrastructure**. What is needed? A cluster of machines of course, but also software libraries that implement the core optimizations on computing, communication, memory, IO and hide the complexity of running a distributed training. 

Two distribution models
=======================
Two not-exclusive approaches to achieve parallelization are **data parallelism** and **model parallelism**. The first approach is based on splitting the data over multiple GPUs (same or different nodes) while the model is replicated. Each replica will use separate batches of training samples. Hence the computed gradients will be different. We don't want the replicas to diverge, so a synchronization step is performed to recall all the computed gradients, combine them, and backpropagate them similarly for all the replicas. The second approach of model parallelism is needed when the model does not fit within a single GPU. The obvious solution is to split the model and distribute its pieces over multiple GPUs (potentially on multiple nodes). The bigger the model, the hungrier it is, so typically, data parallelism must also be exploited when model parallelism is necessary. Finally, distributing the computations led to the increase of communications overhead between GPUs to perform the synchronization steps. 

In recent times, new advanced paradigms have been develop in order to push the parallelization capabilities to its limit (Tensor and Pipeline parallelism), providing also optimizations such as mixed-precision and quantization.

Advanced techniques for distributed training
============================================

#### ZeRO Data&Model Parallelism
The two ideas of distributing the data and the model is clear but it's not straightfoward to implement it: when a model is segmented over multiple GPUs, also gradients and optimizer states are distributed as well, not to mention all the activations for each layer. The naive way, as mention before, is to replicate all of them multiple times for each GPU, but this is not a scalable approach. At the other end, the most efficient way to reduce the memory footprint on the GPUs is to eliminate the redundancy: each GPU has its own exclusive piece of model's weights, gradients and optimizer's state as well as its own subset of training data. This is nice but it's not enough. We need to orchestrate the forward pass and compute the loss. The Zero Redundancy Optimizer (ZeRO) ([S. Rajbhandari et al. 2020](https://arxiv.org/abs/1910.02054)), an open source library for memory optimization, implements the following procedure: let's say model M and the training data are splitted in four pieces over four GPUs: $M_0 + D_0 : GPU_0$, $M_1 + D_1 : GPU_1$, $M_2 + D_2 : GPU_2$ and $M_3 + D_3 : GPU_3$. The forward pass works like this:
* $GPU_0$ broadcasts $M_0$ to $GPU_1, GPU_2$ and $GPU_3$. 
* Each GPU makes the forward pass on $M_0$ using its own subset of training data.
* The activations are saved in a temporary buffer.
* Each GPU (except $GPU_0$) deletes its copy of $M_0$.
This procedure is repeated for each GPU and the loss is computed on each GPU for its own dataset subset. Next, backpropagation needs to do its magic. 
* $GPU_3$ broadcasts $M_3$ to $GPU_0, GPU_1$ and $GPU_2$.
* Each GPU compute the gradients using its own dataset subset.
* $GPU_0, GPU_1$ and $GPU_2$ propagate the computed gradients to $GPU_3$ which accumulates them all. 
* $GPU_0, GPU_1$ and $GPU_2$ can delete their own temporary copy of $M_3$ and gradients. 
* Each GPU can now delete all $M_3$ activations computed during the forward step.
Now each GPU has its own accumulated gradients from all dataset subsets. Finally, we can perform the parameters update: the optimization step can execute in parallel on each GPU, which results in a new set of model weights. Just to recap: rather than copying the complete model parameters, gradients, and optimizer states, each GPU exclusively retains a specific portion. Subsequently, during execution, when the entire layer parameters are required for a specific layer, all GPUs harmonize to exchange the absent segments amongst themselves. [This Microsoft's blog post](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) provides a visual representation of it. 

#### DeepSpeed
DeepSpeed is an open source  


While ZeRO primarily benefits large models during distributed training across a cluster of devices, we also introduce new technology, highly optimized transformer kernels and asynchronous I/O, that boosts compute and I/O speed of training on each individual GPU.

We achieve the fastest BERT training record: 44 minutes on 1,024 NVIDIA V100 GPUs.

The system performance improvements of DeepSpeed on BERT pretraining primarily come from our highly optimized transformer kernels, where we employ two lines of optimizations.




#### Tensor Parallelism
In order to achieve more fine-grained parallelism, we need to go deeper. Pipeline Parallelism aims to split the computations between layers while Tensor Parallelism reaches the lowe-level tensor multiplications layer.  

the next step is to parallelize the tensor computations. 

#### Pipeline Parallelism




DeepSpeed and ZeRO are two libraries that works togheter to optimize the memory at scale. 



Let's start with model sharding: 



In Tensor Parallelism (TP) each GPU processes only a slice of a tensor and only aggregates the full tensor for operations that require the whole thing.

 pipeline parallelism.


Software Libraries
==================
PyTorch DP, DDP, FSDP 
PyTorch Lightining
HuggingFace Accelerate
..




More articles
=============
https://huggingface.co/blog/bloom-megatron-deepspeed
https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/
https://www.deepspeed.ai/training/
https://github.com/microsoft/DeepSpeed
https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/
https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/
https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/