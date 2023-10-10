---
layout: post
title:  "Distributed training"
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
Two not-exclusive approaches to achieve parallelization are **data parallelism** and **model parallelism**.

### Data parallelism

The first approach is based on splitting the training data over multiple GPUs (same or different nodes) while the model is replicated. Each replica will use separate batches of training samples. However, each replica's weights can not be updated by its own computed gradients if we don't want the replicas to diverge. Hence a synchronization step is performed to recall all the computed gradients, combine them, and backpropagate them similarly for all the replicas. In the context of HPC this paradigm is called *Single Program Multiple Data* or *SPMD* since the same application runs on all machines but each one operates on different portions of the training dataset. 

There's multiple ways of mapping processes to nodes, since each node can run multiple processes and each process can use multiple GPUs. In order to achieve a good I/O, we can assume the following heuristic: each process uses only one GPU and each node will run $G$ processes, where $G$ is equal to the number of GPUs. 

There's a terminology that allows us to express the mapping between processes and nodes/GPUs. The *World Size* (*W*) is the total number of application processes running across all the nodes at one time and the *Local World Size* (L) the number of processes running on each node. Each application has a Global Rank $[0, W-1]$ and a Local Rank $[0, G-1]$, as shown in the image below. 

| ![distributed topology](/assets/2023-08-02-2023-08-02-distributed-training/topologies.png)| 
|:--:|                 
| *Figure*: .  Credits: [PyTorch docs](https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md).|

Pytorch provides an abstraction to implement the paradigm explained above, called [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). It can be used to distribute the training over multiple GPUs, in a single or multiple machines. It replicates the model and the optimizer only once during the inizialization and it achieves data parallelism using a [DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) that sends training batches to GPUs through a backend distributed communication library (supporting *gloo*, *mpi* and *nccl*). 

```python
model = DistributedDataParallel(model, device_ids=[device_id])
```

The following video shows how it works.
[![PyTorch: What is Distributed Data Parallel (DDP)](https://img.youtube.com/vi/Cvdhwx-OBBo/0.jpg)](https://www.youtube.com/watch?v=Cvdhwx-OBBo)

In order to exploit data parallelization over multiple machines, the script that wraps the model into the DistributedDataParallel object, must be launched on each machine. 

Usually in distributed training we would like our jobs to satisfy two important requirements: **fault-tolerance** and **elasticity**. *Elasticity* refers to the ability of the training job to scale up or down as needed while tolerating certain levels of membership changes or failures.
To achieve these two properties we need to fill an abstraction gap. Fortunatly, PyTorch provides us with the `torchrun` utility:

```bash
torchrun
    --nnodes=1:4
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```
If a training script fails, `torchrun` tries to restart it (max 3 times in the example above) using the last saved snapshot that also includes the optimizer state and any other stateful attributes that are needed to restart the training job.

In addition, `torchrun` adapts dynamically the training distribution to changes in the number of nodes or processes thanks to the elasticity feature. The range $1:4$ allows the distributed training to scale from a minimum of 1 node to a maximum of 4 nodes. 

Note that there's no need to pass the `rank` and `world_size` information since `torchrun` autodiscover the available resources. 

In order to distribute the training to multiple machines, the torchrun command must be executed on each machine with identical rendezvous arguments. For this purpose, a workload such as Slurm can be used. In this scenario, we need to take care to the inter-node communication latencies. [This page](https://pytorch.org/docs/stable/elastic/train_script.html#elastic-train-script) and [this page](https://pytorch.org/docs/stable/elastic/run.html#important-notices) provide additional suggestions on how to write the training script. 




### Model parallelism
The size of the training data is not the only problem in distributed training. Indeed during a training run:
* the model's weights are stored in the GPU;
* the activations gradients are computed;
* the optimizer state is also mantained in memory.

For large models, one GPU is not enough to contain all this data. What can we do in this situation? 

We can adopt the second paradigm of distributed training: **model parallelism**. The model is splitted and distributed over multiple GPUs (and potentially on multiple nodes). The bigger the model, the hungrier it is, so typically, data parallelism must also be exploited when model parallelism is necessary. 

The two ideas of distributing the data and the model is clear but it's not straightfoward to implement it: when a model is segmented over multiple GPUs, also gradients and optimizer states are distributed as well, not to mention all the activations for each layer. The naive way, as mention before, is to replicate all of them multiple times for each GPU, but this is not a scalable approach.

#### Model parallelism with ZeRO
The [*Zero Redundancy Optimizer*](https://deepspeed.readthedocs.io/en/latest/zero3.html) (*ZeRO*) removes the memory redundancies across data-parallel processes by partitioning the three model states (optimizer states, gradients, and parameters) across data-parallel processes instead of replicating them. Each GPU has its own exclusive piece of model's weights, gradients and optimizer's state as well as its own subset of training data. This is nice but it's not enough because we need to orchestrate the forward pass and compute the loss. *ZeRO* implements the following procedure: let's say model $M$ and the training data $D$ are splitted in four pieces over four GPUs: 
* $M_0 + D_0 : GPU_0$ 
* $M_1 + D_1 : GPU_1$
* $M_2 + D_2 : GPU_2$
* $M_3 + D_3 : GPU_3$
 
The forward pass works like this ([watch the video](https://www.microsoft.com/en-us/research/uploads/prod/2020/02/Turing-Animation.mp4)):
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
  
Now each GPU has its own accumulated gradients from all dataset subsets. Finally, we can perform the parameters update: the optimization step can execute in parallel on each GPU, which results in a new set of model weights. 

Just to recap: rather than copying the complete model parameters, gradients, and optimizer states, each GPU exclusively retains a specific portion. Subsequently, during execution, when the entire layer parameters are required for a specific layer, all GPUs harmonize to exchange the absent segments amongst themselves. This video gives a visual representation of this process.

ZeRO is one of the many optimizations included in the [DeepSpeed](https://github.com/microsoft/DeepSpeed) open-source project by Microsoft.

### Model parallelism in PyTorch

PyTorch provides a technique inspired by ZeRO, abstracted in a class called [*FullyShardedDataParallel*](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel) (FSDP). This class is another wrapper for our PyTorch model that enables both data and model parallelism. Check out [this blog post](https://medium.com/pytorch/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff) that demonstrates the feasibility of training 175B and 1T-parameter models using FullyShardedDataParallel (FSDP) on AWS.

Advanced techniques for distributed training
============================================

In recent times, new advanced paradigms have been develop in order to push the parallelization capabilities to its limit such as Tensor and Pipeline parallelism. In addition new optimizations such as mixed-precision and quantization have become very important for training very large models.

#### Pipeline and Tensor Parallelism
In order to achieve more fine-grained parallelism, we need to go deeper. Pipeline Parallelism aims to split the computations between layers while Tensor Parallelism reaches the lowe-level tensor multiplications layer. In Tensor Parallelism (TP) each GPU processes only a slice of a tensor and only aggregates the full tensor for operations that require the whole thing. 

the next step is to parallelize the tensor computations. 


To cover
========

https://github.com/Lightning-AI/lightning

https://huggingface.co/docs/accelerate/index 

https://huggingface.co/blog/bloom-megatron-deepspeed

https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/

https://www.mishalaskin.com/posts/tensor_parallel