---
layout: post
title:  "Hello transformers!"
date:   2023-06-29 14:50:00 +0200
categories: deep-learning 
---

The *Transformer* architecture defined a new standard for modern neural networks design, leading to the development of the current state-of-the-art models such as *GPT*, *BERT*, *CLIP* and enabling the training of powerful multi-modal architectures. 

Prerequisites
=============
- RNN

Outline 
=======
TODO

Sequence to sequence modeling
=============================

One categorization of deep learning models considers the nature of the input and the corresponding output of a neural network architecture. Both the input or output can be single vector or a sequence of vectors. Let's give some examples: a single RGB image is represented by a tensor with shape (height,weight,channels), while a single word in sentence can be represented with a tensor of shape (v) where v is the vocabulary size. Both of these examples are single vectors, not sequences. On the other hand, a video is a sequence of images and a sentence is a sequence of word. *Figure 1* discriminates all the possible combinations. Let's start from the left-most panel: it shows a *one to one* architecture, a single tensor goes in, a single tensor comes out. This is the case of *Image Classification*, in which a single image is passed to the network to produce a vector of probabilities (one for each class). In the diagram of the second panel, a vector goes in and a sequences comes out. *Image Captioning* is one task which agrees to this representation of input and output: an single image is given as input and the model generates a sentence (sequence of words) describing the image. Viceversa, *many to one* is a task addressed by generative models which accept a natural language description and they generate an image that follows that description. Finally, the *many to many* approach is used in Machine Translation where a sentence in a given language is translated to another language. The last panel, still describes the *many to many* case but with synched input and output, for example in Video Classification, in which each frame is associated with a vector of probabilities predicting the frame's label. 

| ![dl-architectures](/assets/2023-06-29-transformers/dl-architectures.jpg)| 
|:--:|                 
| *Figure 1*: from [Andrej Karpathy's blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). Each rectangle represent a vector. Red vectors are the input blue vectors are the output and green vectors represent holds the neural network states. |

Let's focus on *many to many* models and Machine Translation task, since the most important bulding block of the Transformer architecture, the attention mechanism, stemmed from the research in this field. Before Attention and Transformer, Recurrent Neural Network (RNN) and then, Long-Short Memory Netorks (LSTM) [1] have been used to build models that achieved the state of the art scores on benchmark datasets. These models were based on the Encoder-Decoder architecture. 

Encoder-Decoder architecture for sequences
==========================================
Neural network Encoder-Decoder architectures are designed in order to have an hidden layer that acts like a bottleneck forcing the Encoder part to learn how to compress the information. The Decoder part of the network starts from this representation in order to solve a particular task. These kind of architecture is widely used in several fields, from Machine Vision to Natural Language Processing. 
- Example with vanilla?
- Example with conv layers?
- Advantage of learning latent compressed representation (TODO)
- Embeddings - how to represent words ?
  
Both Encoder and Decoder can be implemented with different types of layers regarding the data type and the representation learning capabilities we want to give to the network. Tipycally, in Machine Vision the Encoder and Decoder's layers are convolutional, in order to exploit the locality features of an image. If the input is a sequence, recurrent layers give the AutoEncoder the capability of processing sequences of arbitray length and exploit temporality. This is the case of Natural Language Processing, in which sentences are a sequence of words. 

- Embeddings - how to represent words ?

The diagram in *Figure 2* shows a simple Autoencoder architecture composed of one recurrent layer for the Encoder and one recurrent layer for the Decoder. Each RNN layer is in reality, composed of a solitary rolled RNN cell that unrolls in accordance with the number of time steps. [This article](https://towardsdatascience.com/all-you-need-to-know-about-rnns-e514f0b00c7c) explain well how the RNN layer processes an input sequence. Basically, at each time step, the Encoder RNN cell gets a new sequence element and the internal state computed in the previous time step (the initial hidden state is typically a vector of zeros) and outputs a new element for the output sequence and the updated internal state. We are interested in the internal state of the last time step, obtained with the processing of input `x_3`. This fixed-lenght *encoder vector* as shown in *Figure 2* encodes all the information of the input sequence. The vector is then used to initialize the internal state of the Decoder network, that uses its own outputs to unroll and give the final output sequence `y_1`, `y_2`. 


| <img src="/assets/2023-06-29-transformers/encoder-decoder.jpg" alt="encoder-decoder" width="500"/>| 
|:--:|                 
| *Figure 2*: the diagram shows a simple Autoencoder architecture composed of one recurrent layer for the Encoder and one recurrent layer for the Decoder. Credits to [towardsdatascience.com](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346). |

Limitations
===========
As shown by [Bengio, et al. (1994)](https://www.researchgate.net/publication/5583935_Learning_long-term_dependencies_with_gradient_descent_is_difficult) the main issue of this architecture is that RNNs encounter difficulties in learning to establish long-term dependencies. This capability is crucial in NLP, as sentences can be pretty long and remembering information given at the beginning of the sentence can have an huge impact to solve, for example, a prediction task. 

A variant of the RNN model, called Long-Short Memory Network (LSTM) has been introduced by  [Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) designed for learning long-term dependencies. The [Colah's blog post on LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs) is very useful to understand the inner mechanism of LSTM cells. Autoencoders based on LSTM layers set a new standard for sequence to sequence learning, as shown in the work of [I. Sutskever, et al. (2014)](https://arxiv.org/abs/1409.3215). 

Altought LSTMs mitigated the problem of learning long-term dependencies they have not completely addressed it. For very long sequences they still struggle. This problem is inherently related to the recursion nature of the architecture: information in RNNs and LSTMs is retained thanks to previously computed hidden states but they are updated at each time steps, decreasing the influence of past words. Furthermore, in the Autoencoder architecture, only the last hidden state is actually passed to the decoder network, and researchers realized that this fixed-lenght vector was a bottleneck for these types of models. 

In addition, LSTMs have another important limitation: the processing of the input sequences is sequential and can not be parallelized, hence, LSTMs can't be trained in parallel.  

Transformers
============
Transformers, introduced in a paper titled *"Attention is all you need"* ([A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf)), made their debut in machine translation to tackle two main objectives:

1. Minimize performance degradation caused by long-term dependencies. 
2. Eliminate recursion for parallel computation resulting in reduced training time. 

Rather than relying on past hidden states to capture dependencies with previous words, Transformers take a different approach by processing an entire sentence as a whole. This is the main reason why Transformers do not suffer from long-term dependency issues. In addition, processing an entire sentence as a whole means that they do not require sequential processing and can capture dependencies in a parallel manner. 

Two innovations make all of this possible: *Self-Attention* and *Positional Embeddings*. 


#### Attention and Self-Attention

*Self-Attention* was a novel contribution of [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf), but the original *Attention* technique was introduced before by [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) and [Luong et al. (2015)](https://arxiv.org/abs/1508.04025). Since *Self-Attention* is based on the *Attention* technique, let's understand it first. 

As said before, in the Encoder-Decoder architecture only the last hidden state is actually passed to the decoder network, and researchers realized that this fixed-lenght vector was a bottleneck for these types of models. The *Attention* technique has been introduced to solve this problem. Let's understand how it works. In neural machine translation (NMT), the encoder processes the source sentence and produces a sequence of hidden states, each representing a different word or position in the source sentence. On the other hand, the decoder generates the target sentence word by word based on the context of the source sentence. In the Encoder-Decoder architecture with *Attention*, the encoder passes to the decoder all the hidden states computed at each time step. The decoder network, in order to cope with this amount of data, uses the *Attention* technique to focus on the parts of the input that are relevant to the i-th decoding time step. In other words, the *Attention* technique aims to address the challenge of aligning the relevant parts of the source sentence with the words being generated by the decoder. *Figure 3* shows this exact process. 

| <img src="/assets/2023-06-29-transformers/attention.gif" alt="attention" />| 
|:--:|                 
| *Figure 3*: the diagram shows the Attention technique. Took inspiration from the [jalammar's blog](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/). |

Let's describe all the steps, and remember that the decoder network has access to every hidden state (`h_1`, `h_2`, `h_3`) generated by the encoder network. The decoder starts its processing when it's fed with the `<END>` token. As a normal decoder, it starts with an initial hidden state `h_init` and generates a new hidden state `h_4` and the first output word ("I" in this case). The *Attention* technique is applied to generate the output word "I" in the following way:
1. The decoder gathers its hidden state (`h_4`) and the encoder's hidden states (`h_1`, `h_2`, `h_3`). 
2. A scoring function measures the relevance (or alignment) between the decoder's current hidden state and each of the encoder's hidden states. In other words, these similarity scores allow the *Attention* mechanism to dynamically focus on different parts of the source sentence during the decoding process. There's several ways to compute this score: as the dot product between the two vectors `score(h_t, h_i) = h_t * h_i`, or in the *general scoring* approach: `score(h_t, h_i) = h_t * W * h_i`, where `W` is a learnable weight matrix that transforms the hidden state vectors before taking their dot product. This allows the model to learn more complex alignments between the encoder and decoder hidden states. 
3. Once the scores are computed for each encoder hidden state, they are usually passed through a softmax function to obtain a probability distribution over the hidden states. The softmax function ensures that the scores sum up to 1, representing the importance or **attention weights** assigned to each hidden state. 
4. These attention weights are then used to compute a weighted sum of the encoder hidden states (`C_4`), amplifying hidden states with high scores, and drowning out hidden states with low scores.  
5. The weighted sum (`C_4`) is concatenated with the decoder's hidden state (`h_4`) and this new vector is passed through a feedforward neural network (trained jointly with the model). 
6. The output of the feedforward neural networks indicates the output word of this time step.
7. This process is repeated for each time step.

*Figure 4* shows the powerful effectiveness of the *Attention* technique considering the translation of a sentence from French to English. We can see from the attention matrix that when the model outputs *"european economic area"* it paids attention to the correct input words that in this case are reversed (*"zone économique européenne"*).  This alignment between French and English is learned from the training phase. 

| <img src="/assets/2023-06-29-transformers/attention_sentence.jpg" alt="attention-sentence" />| 
|:--:|                 
| *Figure 4*: . Credits: [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf). |




#### Positional Embeddings








References
----------

- [1] Sepp Hochreiter and Jürgen Schmidhuber. "Long short-term memory". Neural computation, 9(8):1735–1780, 1997.
- [2] Cho, K. et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", 2014. doi:10.48550/arXiv.1406.1078.