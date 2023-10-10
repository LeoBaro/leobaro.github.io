---
layout: post
title:  "Multimodal Foundation Models"
date:   2023-09-25 10:00:00 +0200
categories: multi-modal deep-learning
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

<p>Estimated reading time: 20 minutes.</p>

Introduction
============
We are witnessing a rapid evolution in the field of deep learning in recent years, particularly with the development of foundation models that led to a paradigm shift in AI. Those models have been trained with large-scale data, generally in a self-supervised fashion. This massive scale training and transfer learning led these models to obtain emergent capabilities: zero-shot learning is the ability to perform tasks without any explicit training on those tasks. For example, a foundation model may be able to translate a sentence from English to Spanish, even though it has never been trained on parallel text in those two languages. Few-shot learning is the ability to learn new tasks with very few examples. For example, a foundation model may be able to learn to classify a new type of image after seeing only a few examples of that image type. Moreover, foundation models have opened up the opportunity to build more powerful architectures on top of them.

In this article, I'm going to talk about the current state of the research academy to develop multimodal foundation models and general purpose multimodal agents. The main reference for this article is the review paper on multimodal foundation models of [Li et al. (2023)](https://arxiv.org/pdf/2309.10020.pdf).

Evolution of AI
===============

The evolution of AI in terms of the number of the learnable parameters of the models has been remarkable in the past decade. In 2010, the state-of-the-art deep learning models had only a few million parameters. Since 2018, growth has accelerated for language and multimodal models, with model size reaching billions of parameters to 2023. This has been possible thanks to the availability of more powerful computing resources, the development of new training techniques, and the collection of larger and larger datasets. 

| ![Parameters of Machine Learning systems over time](/assets/2023-09-25-multimodal-foundation-models/parameters.png)| 
|:--:|                 
| *Figure 1*: Model size over time, separated by domain. Most systems above the red line gap are language or multimodal models. 
Credits to: [Villalobos et al. (2022)](https://arxiv.org/abs/2207.02852)|

But in which direction the research is going? What are we trying to achieve?

Back to 2010: Task-Specific Models
----------------------------------
A decade ago, the trend was to train models from scratch, each one capable to solve a particular task. The AlexNet paper was published in 2012 in one of the most influential research papers in the history of computer vision, cited over 140000 times as of 2023 opening the way for the extensive use of deep learning in computer vision. This Convolutional Neural Network architecure was developed to push the ability to perform image classification. In the NLP world, [Sutskever et al. (2014)](https://arxiv.org/abs/1409.3215) proposed an architecture based on Long-Short Term Memory networks to improve that was a breakthrough in the neural machine translation field. Few time after, [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) proposed a new approach to machine translation, introducing the *attention* mechanism. In the breakthrough paper titled *Attention is all you need* by [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) a new simple network architecture was proposed. The *Transformer* was based solely on attention mechanisms, no recurrence nor convolutions, with training parallelism and superior results in machine translation.

Step 2: Pre-Trained Models
--------------------------
The *Transformer* architecture revolutionized the field of NLP, in particular with two works: *"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"* by [Devilin et al. (2018)](https://arxiv.org/abs/1810.04805) and *"Improving language understanding by generative pre-training"* (GPT-1) by [Radford et al. (2018)](https://www.mikecaptain.com/resources/pdf/GPT-1.pdf).
This moment is considered the inception of the foundation model era. These architectures can be **pre-trained at scale** over large textual dataset to automatically learn the sintax and the semantic of the language from raw data, in a **self-supervised** way. The learned representation of the language is general and rich enough to give the possibiliy to fine-tune the same model (BERT or GPT) and to use it to address a variety of tasks. GPT-2 was created as a *"direct scale-up of GPT-1 with a ten-fold increase in both its parameter count and the size of its training dataset"* by [Radford et al. (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

The success of pre-train and self-supervision was so great that the computer vision community started to adopt the same principles to address computer vision tasks. 

Masked image modeling (MIM) is a self-supervised learning technique for training deep learning models for visual tasks. It works by randomly masking out a portion of an input image and then training the model to predict the missing pixels. This forces the model to learn the relationships between different parts of the image in order to accurately reconstruct it. Exploting this training strategy, *BERT* has been adapted to work with images (Beit: BERT Pre-Training of Image Transformers) in [Bao et al. (2022)](https://arxiv.org/pdf/2106.08254.pdf). 

| ![BEIT: BERT Pre-Training of Image Transformers](/assets/2023-09-25-multimodal-foundation-models/beit.png)| 
|:--:|                 
| *Figure x*: Overview of BEIT pre-training. Before pre-training, we learn an “image tokenizer” via autoencoding-style reconstruction, where an image is tokenized into discrete visual tokens according to the learned vocabulary. During pre-training, each image has two views, i.e., image patches, and visual tokens. We randomly mask some proportion of image patches (gray patches in the figure) and replace them with a special mask embedding [M]. Then the patches are fed to a backbone vision Transformer. The pre-training task aims at predicting the visual tokens of the original image based on the encoding vectors of the corrupted image.
Credits to: [Bao et al. (2022)](https://arxiv.org/pdf/2106.08254.pdf).|

Another example of MIM training is the work of [He et al. (2022)](https://arxiv.org/pdf/2111.06377.pdf) proving that Masked AutoEncoders (MAE) are scalable self-supervised learners. 

Another form of self-supervision is called *self-distillation*. In traditional *knowledge distillation*, a teacher model is trained on a given dataset, and then a student model is trained to learn the teacher model's outputs. In *self-distillation*, the teacher model and the student model are the same model. *Self-distillation* works by training the model to predict its own outputs from a previous epoch. This forces the model to learn to represent the data in a more robust and informative way. 



The same principles started to empowered the multi-modal field too, in particular in the subfield of vision-language modalities. Two important models were developed: *CLIP* by [Radford et al. 2021](https://arxiv.org/abs/2103.00020) and *DALL-E* by [Ramesh et al. 2021](https://arxiv.org/abs/2102.12092). *CLIP* exploited contrastive learning to align the representation spaces of a vision encoder and of a textual encoder in the way, an image and the caption describing the image are two simular vector embeddings. In contrast, *DALLE-E* is a 12-billion parameter version of GPT-3 trained to generate images from text descriptions, using a dataset of text–image pairs. 



  
Step 3: Foundation Models
-------------------------
It turned out that increasing the number of parameters of these networks and performing a web-scale training, the models acquired emergent capabilities such as zero-shot and few-shot learning. In particular, [Brown et al. (2020)](https://arxiv.org/abs/2005.14165) demonstrated the scaling capability of the GPT architecture (model and data sizes), giving birth to GPT-3. From the paper:

> *"Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic."*

In 2023, *MetaAI* release a collection of foundation language models ranging from 7B to 65B parameters, called LLama ([Touvron et al. 2023](https://arxiv.org/abs/2302.13971)) trained using publicly available datasets only.

In the vision and multimodal side, it's worth to mention two works: the first is *Flamingo* by [Alayrac et al. 2022](https://arxiv.org/abs/2204.14198) a model that can be *"rapidly adapted to novel tasks using only a handful of annotated examples"* i.e. a few-shot learner that is also capable of handling *"sequences of arbitrarily interleaved visual and textual data, and seamlessly ingest images or videos as inputs"*. 

| ![Flamingo: a Visual Language Model for Few-Shot Learning](/assets/2023-09-25-multimodal-foundation-models/flamingo.png)| 
|:--:|                 
| *Figure x*: . Flamingo is a few-shot learner that can adapt to various image/video understanding and it's also capable of multi-image visual dialogue.
Credits to: [Alayrac et al. 2022](https://arxiv.org/abs/2204.14198).|

The second is PaLM-E by [Driess et al. 2023](https://arxiv.org/pdf/2303.03378.pdf), a *"single general-purpose multimodal language model*". It can handle multimodal sentences of arbitrary text-interleaved modalities (e.g. images inserted alongside text tokens) to perform *"embodied reasoning tasks, visual-language tasks, and language tasks"*. 

| ![PaLM-E: An Embodied Multimodal Language Model](/assets/2023-09-25-multimodal-foundation-models/palm-e.png)| 
|:--:|                 
| *Figure x*: PaLM-E transfers knowledge from visual-language domains into embodied reasoning to answering questions about the observable world. 
Credits to: [Driess et al. 2023](https://arxiv.org/pdf/2303.03378.pdf).|



Step 4: General-purpose assistants
----------------------------------
The aforementioned foundation models unifies several capabilities in single models and provides a flexible interface to interact with them. So, what's missing? It turned out that such models are not very good to follow human intent and interactions in a multi-message environment such as a chat. In the NLP world, GPT-3 has been aligned to perform better in this context, achieving the capability to perform a wide range of language tasks *in the wild*. The alignment was done with Reinforcement Learning with Human Feedback (RLHF). It works by using human feedback as a loss to align a language model trained on a general corpus of text data to that of complex human values. This "fine-tuning" led to the generation of a general-purpose AI agent, called [ChatGPT](https://openai.com/blog/chatgpt).  


Visual understanding
====================

