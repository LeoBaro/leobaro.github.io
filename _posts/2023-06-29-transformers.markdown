---
layout: post
title:  "Hello transformers!"
date:   2023-06-29 14:50:00 +0200
categories: deep-learning 
---

The *Transformer* architecture defined a new standard for modern neural networks design, leading to the development of the current state-of-the-art models such as *GPT*, *BERT*,  *CLIP* and so on and enabling the training of powerful multi-modal architectures. 

This post starts covering the limitations of previous models that lead researchers to the development of the Transformer architecture. 


 and then, it explains the building block of the transformer architecture called *Attention*. Finally, the vision Transformer architecture is introduced.

One categorization of deep learning models considers the nature of the input and the corresponding output of a neural network architecture. Both the input or output can be single vector or a sequence of vectors. Let's give some examples: a single RGB image is represented by a tensor with shape (height,weight,channels), while a single word in sentence can be represented with a tensor of shape (v) where v is the vocabulary size. Both of these examples are single vectors, not sequences. On the other hand, a video is a sequences of images and a sentence is a sequence of word. *Figure 1* discriminates all the possible combinations. Let's start from the left-most panel: it shows a *one to one* architecture, a single tensor goes in, a single tensor comes out. This is the case of *Image Classification*, in which a single image is passed to the network to produce a vector of probabilities (one for each class). In the diagram of the second panel, a vector goes in and a sequences comes out. *Image Captioning* is one task which agrees to this representation of input and output: an single image is given as input and the model generates a sentence (sequence of words) describing the image. Viceversa, *many to one* is a task addressed by generative models which accept a natural language description and they generate an image that follows that description. Finally, the *many to many* approach is used in Machine Translation where a sentence in a given language is translated to another language. The last panel, still describes the *many to many* case but with synched input and output, for example in Video Classification, in which each frame is associated with a vector of probabilities predicting the frame's label. 

| ![dl-architectures](/assets/2023-06-29-transformers/dl-architectures.jpeg)| 
|:--:|                 
| *Figure 1*: from [Andrej Karpathy's blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). Each rectangle represent a vector. Red vectors are the input blue vectors are the output and green vectors represent holds the neural network states. |

Let's focus on *many to many* models and Machine Translation task, since the most important bulding block of the Transformer architecture, the attention mechanism, stemmed from the research in this field. Before Attention and Transformer, Recurrent Neural Network (RNN) and then, Long-Short Memory Netorks (LSTM) [1] have been used to build models that achieved the state of the art scores on benchmark datasets. These models were based on the Encoder-Decoder architecture. 

Encoder-Decoder architecture for sequences
------------------------------------------




| ![dl-architectures](/assets/2023-06-29-transformers/dl-architectures.jpeg)| 
|:--:|                 
| *Figure 1*: from [Andrej Karpathy's blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). Each rectangle represent a vector. Red vectors are the input blue vectors are the output and green vectors represent holds the neural network states. |





References
----------

[1] Sepp Hochreiter and Jürgen Schmidhuber. "Long short-term memory". Neural computation, 9(8):1735–1780, 1997.
[2] Cho, K. et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", 2014. doi:10.48550/arXiv.1406.1078.