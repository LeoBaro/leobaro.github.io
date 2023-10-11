---
layout: post
title:  "A gentle introduction to distributed training of large deep learning models"
date:   2023-10-11 21:00:00 +0200
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

<p>Estimated reading time: 15 minutes.</p>

# Introduction

In recent times it has been shown that larger model training improve model quality. The development of neural architectures based on [Transformers](https://leobaro.github.io/deep-learning/2023/06/29/transformers.html) let the number of parameters to drammaticaly increase, from 340M of BERT to 530 billion (!) of the largest (at the time of writing) Megatron-Turing NLG model [D. Narayanan et al. (2021)](https://arxiv.org/abs/2104.04473).
| ![number of parameters](/assets/2023-10-11-distributed-training/model_size.jpg)| 
|:--:|                 
| *Figure 1*: the dramatic increase in the model size over recent years. Credits: [Large Language Models: A New Moore's Law?](https://huggingface.co/blog/large-language-models).|

Training such big models is challenging for two main reasons: first, a large model does not fit in GPU memory even if a multiple-GPU server node is available and second, the computation complexity of the training process is too high, requiring too much time to make the model reaching the desired accuracy. The only way to cope with this is to enable a **large-scale training infrastructure**. What is needed? A cluster of machines of course, but also software libraries that implement the core optimizations on computing, communication, memory, I/O and hide the complexity of running a distributed training. 

# Two distribution models

Two not-exclusive approaches to achieve distributed training are **data parallelism** and **model parallelism**.

## Data parallelism

The first approach is based on splitting the training data over multiple GPUs (same or different nodes) while the model is replicated. Each replica will use separate batches of training samples. However, each replica's weights can not be updated by its own computed gradients if we don't want the replicas to diverge. Hence a synchronization step is performed to recall all the computed gradients, combine them, and backpropagate them similarly for all the replicas. In the context of HPC this paradigm is called *Single Program Multiple Data* or *SPMD* since the same application runs on all machines but each one operates on different portions of the training dataset. 

There are multiple ways of mapping processes to nodes since each node can run multiple processes and each process can use multiple GPUs. To achieve a good I/O, we can assume the following heuristic: each process uses only one GPU and each node will run $G$ processes, where $G$ is equal to the number of GPUs. 

There's a terminology that allows us to express the mapping between processes and nodes/GPUs. The *World Size* (*W*) is the total number of application processes running across all the nodes at one time and the *Local World Size* (L) the number of processes running on each node. Each application has a Global Rank $[0, W-1]$ and a Local Rank $[0, G-1]$, as shown in the image below. 

| ![distributed topology](/assets/2023-10-11-distributed-training/topologies.png)| 
|:--:|                 
| *Figure 2*: the mapping of processes to nodes. Credits: [PyTorch docs](https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md).|

Pytorch provides an abstraction to implement the paradigm explained above, called [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html). It can be used to distribute the training over multiple GPUs, in single or multiple machines. It replicates the model and the optimizer only once during the initialization and it achieves data parallelism using a [DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) that sends training batches to GPUs through a backend distributed communication library (supporting *gloo*, *mpi* and *nccl*). 

```python
model = DistributedDataParallel(model, device_ids=[device_id])
```

The following video shows how it works.
[![PyTorch: What is Distributed Data Parallel (DDP)](https://img.youtube.com/vi/Cvdhwx-OBBo/0.jpg)](https://www.youtube.com/watch?v=Cvdhwx-OBBo)

To exploit data parallelization over multiple machines, the script that wraps the model into the DistributedDataParallel object must be launched on each machine. 

Usually in distributed training, we would like our jobs to satisfy two important requirements: **fault-tolerance** and **elasticity**. *Elasticity* refers to the ability of the training job to scale up or down as needed while tolerating certain levels of membership changes or failures.
To achieve these two properties we need to fill an abstraction gap. Fortunately, PyTorch provides us with the `torchrun` utility:

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

To distribute the training to multiple machines, the torchrun command must be executed on each machine with identical rendezvous arguments. For this purpose, a workload such as Slurm can be used. In this scenario, we need to take care of the inter-node communication latencies. [This page](https://pytorch.org/docs/stable/elastic/train_script.html#elastic-train-script) and [this page](https://pytorch.org/docs/stable/elastic/run.html#important-notices) provide additional suggestions on how to write the training script. 




## Model parallelism
The size of the training data is not the only problem in distributed training. Indeed during a training run:
* The model's weights are stored in the GPU;
* The activations gradients are computed;
* the optimizer state is also maintained in memory.

For large models, one GPU is not enough to contain all this data. On the extreme, to train a trillion-parameters model 16TB of memory is required to store the weights, gradients, and optimizers. Not to mention the activations that consume additional memory. What can we do in this situation? 

We can adopt the second paradigm of distributed training: **model parallelism**. The model is split and distributed over multiple GPUs (and potentially on multiple nodes). The bigger the model, the hungrier it is, so typically, data parallelism must also be exploited when model parallelism is necessary. 

The two ideas of distributing the data and the model are clear but it's not straightforward to implement it: when a model is segmented over multiple GPUs, gradients and optimizer states are distributed as well, not to mention all the activations for each layer. The naive way, as mentioned before, is to replicate all of them multiple times for each GPU, but this is not a scalable approach.

### Model parallelism with ZeRO
The [*Zero Redundancy Optimizer*](https://deepspeed.readthedocs.io/en/latest/zero3.html) (*ZeRO*) removes the memory redundancies across data-parallel processes by partitioning the three model states (optimizer states, gradients, and parameters) across data-parallel processes instead of replicating them. Each GPU has its exclusive piece of the model's weights, gradients, and optimizer's state as well as its subset of training data. This is shown by the image below.

| ![model parallel](/assets/2023-10-11-distributed-training/zero_model_parallel.png)| 
|:--:|                 
| *Figure 3*: Memory savings and communication volume for the three stages of ZeRO compared with standard data parallel baseline. Credits: [Microsoft blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/).|

This is nice but it's not enough because we need to orchestrate the forward pass and compute the loss. *ZeRO* implements the following procedure: let's say model $M$ and the training data $D$ are split in four pieces over four GPUs: 
* $M_0 + D_0 : GPU_0$ 
* $M_1 + D_1 : GPU_1$
* $M_2 + D_2 : GPU_2$
* $M_3 + D_3 : GPU_3$
 
The forward pass works like this ([watch the video](https://www.microsoft.com/en-us/research/uploads/prod/2020/02/Turing-Animation.mp4)):
* $GPU_0$ broadcasts $M_0$ to $GPU_1, GPU_2$ and $GPU_3$. 
* Each GPU makes the forward pass on $M_0$ using its subset of training data.
* The activations are saved in a temporary buffer.
* Each GPU (except $GPU_0$) deletes its copy of $M_0$.
This procedure is repeated for each GPU and the loss is computed on each GPU for its dataset subset. Next, backpropagation needs to do its magic. 
* $GPU_3$ broadcasts $M_3$ to $GPU_0, GPU_1$ and $GPU_2$.
* Each GPU computes the gradients using its dataset subset.
* $GPU_0, GPU_1$ and $GPU_2$ propagate the computed gradients to $GPU_3$ which accumulates them all. 
* $GPU_0, GPU_1$ and $GPU_2$ can delete their own temporary copy of $M_3$ and gradients. 
* Each GPU can now delete all $M_3$ activations computed during the forward step.
  
Now each GPU has its own accumulated gradients from all dataset subsets. Finally, we can perform the parameters update: the optimization step can execute in parallel on each GPU, which results in a new set of model weights. 

Just to recap: rather than copying the complete model parameters, gradients, and optimizer states, each GPU exclusively retains a specific portion. Subsequently, during execution, when the entire layer parameters are required for a specific layer, all GPUs harmonize to exchange the absent segments amongst themselves. This video gives a visual representation of this process.

ZeRO is one of the many optimizations included in the [DeepSpeed](https://github.com/microsoft/DeepSpeed) open-source project by Microsoft.



Advanced techniques for distributed training
============================================
Large memory requirement is not the only issue we face when training large models. The other issue is **compute efficiency**. As reported by [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361), a trillion-parameters model training would require something around 5000 zettaflops i.e. 4000 NVIDIA A100 GPUs running at 50% compute efficiency about 100 days. 

This number does not allow for the democratization of deep learning, only large data centers possess this amount of computing. However, this is not enough. The more GPUs you use, the more communication overhead you have (as highlighted in the previous paragraphs), reducing the computation efficiency. To increase the computation efficiency we could increase the batch size, letting each GPU crunch as many calculations as possible before the I/O step. The problem is that the batch size cannot grow indefinitely, beyond a certain value, the training convergence deteriorates quickly. If we use data parallelism, the more GPUs we use, the more the batch size is split between them, limiting the scalability. 

Fortunately, new advanced paradigms have been developing to push the computing efficiency to its limit, such as *tensor parallelism*, *mixed-precision* and *quantization*.

### Pipeline parallelism vs Tensor parallelism
We understood that in model parallelism the model is split over multiple GPUs. But how? One naive way to do it is called *pipeline parallelism*. Using this technique, the model is split into multiple stages. For example, each layer of an 8-layers feed-forward network would be assigned to a different GPU and the output of one stage would be passed as input to the next stage. The main problem with that is idle time: GPUs must wait for the input coming from the previous stages. How can we parallelize the model training more efficiently?

*Tensor parallelism* (or *model sharding*) splits the model into shards (a smaller chunk of a tensor) that are distributed across multiple devices and executed in parallel. The main difference between a shard and a layer is that a shard is a piece of a tensor, while a layer is a computational unit that operates on tensors. For example, a linear layer takes two tensors as input (the weights and the activations) and produces a single tensor as output. The image below is an example of how tensor parallelism can be implemented for a simple two-layer feed-forward network. The operation $Z=AB$ can be split as $Z=AB=A_1B_1+A_2B_2$.

| ![tensor parallelism](/assets/2023-10-11-distributed-training/tensor_parallelism_example.png)| 
|:--:|                 
| *Figure 4*: a simple example of tensor parallelism. The two computations $f(XA_1)B_1$ and $f(XA_2)B_2$ are executed on different devices and the results are aggregated with an *allreduce* operation. Credits: [mishalaskin.com](https://www.mishalaskin.com/posts/tensor_parallel).|

### 3D parallelism
The techniques mentioned above can be composed and the system topology can be exploited to optimize computational efficiency. [This blog post](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/) from Microsoft, shows a way to compose pipeline, tensor, and data parallelism in a 3d-topology aware fashion. The model's layers are split among four pipeline stages (resulting in 8 layers) per stage. Each stage is deployed in a different GPU. The layers inside a stage are split with tensor parallelism over 4 workers (purple, green, orange, blue). In this way, the tensor parallelism which is the technique requiring the most comunication overhead, can exploit the large intra-node bandwidth. Pipeline parallelism has the lowest communication volume, hence, the pipeline stages communicated across nodes without being limited by the communication bandwidth. Finally, each pipeline is replicated across two data parallel instances, and ZeRO partitions the optimizer states across the data parallel replicas. 

| ![3d parallelism](/assets/2023-10-11-distributed-training/3d_parallelism_1.png)| 
|:--:|                 
| *Figure 5*: Example 3D parallelism with 32 workers. Layers of the neural network are divided among four pipeline stages. Layers within each pipeline stage are further partitioned among four model parallel workers. Lastly, each pipeline is replicated across two data parallel instances, and ZeRO partitions the optimizer states across the data parallel replicas. Credits: [Microsoft's blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/).|

| ![3d parallelism](/assets/2023-10-11-distributed-training/3d_parallelism_2.png)| 
|:--:|                 
| *Figure 6*: Mapping of workers in Figure 1 to GPUs on a system with eight nodes, each with four GPUs. Coloring denotes GPUs on the same node. Credits: [Microsoft's blog](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/).|

### Mixed precision
The model training can be done in *mixed precision* to reduce the memory requirements.

However, the use of *float16* makes the computations more susceptible to overflow leading to diverging gradient issues, disrupting the training process.

*bfloat16* uses 8 bits for the exponent (same as *float32*) and 7 bits for the significand, while *float16* uses 5 bits for the exponent and 11 bits for the significand. This means that *bfloat16* has a larger dynamic range (i.e., it can represent a wider range of values) than float16 and it doesn't suffer from overflow too much, at the cost of reduced precision.

This technique is called *mixed* because the 16-bit formats are only used for the computation of the activations as the weights will be stored always in full precision (and they are cast to 16-bits before the next iteration by the optimizer).

### Quantization
If *mixed precision* is not enough, floating-point numbers can be casted to lower-precision numbers, such as integers or half-precision floating-point numbers. This technique is call *quantization* can further reduce the memory and computational requirements of the model, but it can also lead to accuracy loss.

# Conclusions
Distributed training is a broad set of techniques for training large language models (LLMs) and other complex deep learning models. 

Distributed training is becoming more accessible to everyone, thanks to the development of software libraries and tools, as well as the increasing availability of cloud computing resources. This is leading to a more democratized AI landscape, where more people are able to develop and use large and complex AI models.

The democratization of AI has the potential to revolutionize many industries and aspects of our lives. It is important to use this technology responsibly and ethically, so that it benefits everyone.