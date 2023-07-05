---
layout: post
title:  "Yet another transformers post!"
date:   2023-06-29 14:50:00 +0200
categories: deep-learning 
toc: true
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

Introduction
============
The *Transformer* architecture defined a new standard for modern neural network design, leading to the development of the current state-of-the-art models such as *GPT*, *BERT*, *CLIP* and enabling the training of robust multi-modal architectures. This post assumes a basic understanding of the AutoEncoder architecture and recurrent models.

Sequence-to-sequence modeling
=============================

One categorization of deep learning models considers the nature of the input and the corresponding output of a neural network architecture. The input or output can be a single vector or a sequence of vectors. Let's give some examples: a single RGB image is represented by a tensor with shape (height, weight, channels), while a single word in a sentence can be defined with a tensor of shape (v) where v is the vocabulary size. Both of these examples are single vectors, not sequences. On the other hand, a video is a sequence of images, and a sentence is a sequence of words. *Figure 1* discriminates all the possible combinations. Let's start from the leftmost panel: it shows a *one-to-one* architecture; a single tensor goes in, and a single tensor comes out. The latter is the case of *Image Classification*, in which a single image is passed to the network to produce a vector of probabilities (one for each class). In the diagram of the second panel, a vector goes in, and a sequence comes out. *Image Captioning* is one task that agrees to this representation of input and output: a single image is given as input, and the model generates a sentence (sequence of words) describing the image. Vice versa, *many-to-one* is a task addressed by generative models which accept a natural language description and generate an image that follows that description. Finally, the *many-to-many* approach is used in Machine Translation where a sentence in a given language is translated to another. The last panel still describes the *many-to -many* scenario, but this time with synched input and output, for example, in Video Classification, in which each frame is associated with a vector of probabilities predicting the frame's label. 

| ![dl-architectures](/assets/2023-06-29-transformers/dl-architectures.jpg)| 
|:--:|                 
| *Figure 1*: from [Andrej Karpathy's blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/). Each rectangle represent a vector. Red vectors are the input blue vectors are the output and green vectors represent holds the neural network states. |

Let's focus on *many-to-many* models and *machine translation* tasks since the essential building block of the Transformer architecture, the attention mechanism, stemmed from the research in this field. Before Attention and Transformer, Recurrent Neural Networks (RNN) and then Long-Short Memory Networks (LSTM) ([S. Hochreiter, J. Schmidhuber (1997)](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext)) were used to build models that achieved the state of the art scores on benchmark datasets. These models were based on the Encoder-Decoder architecture. 

Encoder-Decoder architecture for sequences
==========================================
Neural network Encoder-Decoder architectures are designed to have a hidden layer that acts like a bottleneck forcing the encoder part to learn how to compress the information. The decoder part of the network starts from this representation to solve a particular task. These kinds of architecture are widely used in several fields, from Machine Vision to Natural Language Processing. Both encoder and decoder can be implemented with different types of layers regarding the data type and the representation learning capabilities we want to give to the network. Typically, in Machine Vision, the encoder and decoder's layers are convolutional to exploit the locality features of an image. If the input is a sequence, recurrent layers give the Auto-Encoder the capability of processing sequences of arbitrary length and exploit temporality. This is the case of Natural Language Processing, in which sentences are a sequence of words. 

But how can we input words into a neural network? Before diving into the Encoder-Decoder architecture for sequences, let's understand how to convert text into a more suitable format for neural networks. In a nutshell, each sentence is the input of a *Tokenizer* whose purpose is to apply tokenization and numericalization. Finally, an *Embedder* transforms each token into a vector representation. Let's break these steps down with an example (using the HuggingFace library):

{% highlight python %}
from transformers import DistilBertTokenizerFast, DistilBertModel

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

tokens = tokenizer.encode('Run, rabbit run. Dig that hole, forget the sun.', return_tensors='pt')

print(tokens)
#=> prints ['[CLS]', 'run', ',', 'rabbit', 'run', '.', 'dig', 'that', 'hole', ',', 'forget', 'the', 'sun', '.', '[SEP]']
{% endhighlight %}
*Tokenization* is a way of separating a piece of text into smaller units called tokens. 


{% highlight python %}
decoded_tokens = tokenizer.convert_ids_to_tokens(enc.input_ids)
print(decoded_tokens)
#=> prints tensor([[  101,  2448,  1010, 10442,  2448,  1012, 10667,  2008,  4920,  1010, 5293,  1996,  3103,  1012,   102]])
{% endhighlight %}
*Numericalization* converts each unique token to an unique number.


{% highlight python %}
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

print(model.embeddings.word_embeddings(tokens))

#=> prints 
  tensor([[[ 0.0390, -0.0123, -0.0208,  ...,  0.0607,  0.0230,  0.0238],
           [ 0.0449, -0.0244, -0.0157,  ..., -0.0461, -0.0771, -0.0006],
           [-0.0111, -0.0141, -0.0029,  ...,  0.0023,  0.0018,  0.0093],
           ...,
           [-0.0097, -0.0020,  0.0337,  ..., -0.0339, -0.0130, -0.0300],
           [-0.0244, -0.0138, -0.0078,  ...,  0.0069,  0.0057, -0.0016],
           [-0.0199, -0.0095, -0.0099,  ..., -0.0235,  0.0071, -0.0071]]],
        grad_fn=<EmbeddingBackward0>)

print(model.embeddings.word_embeddings(tokens)[0][0].shape)
#=> prints torch.Size([768])

{% endhighlight %}
Finally, the *Encoding* step converts numbers into vectors.

The tokenization step is crucial since tokens are the building blocks of NLP. It can be applied to single characters, single words, or subwords. Character tokenization and word tokenization have drawbacks, so nowadays, the most common way to tokenize a sentence is to use subwords tokenization algorithms, such as WordPiece by [M. Schuster, K. Nakajima (2012)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf). The subword tokenization is learned from the pretraining corpus with statistical rules and algorithms. If you want to know more, check out [this blog post](https://towardsdatascience.com/tokenization-for-natural-language-processing-a179a891bad4). Finally, an embedding is a real-value representation of a word that encodes the semantics and syntactic of the word so that words closer in the vector space are expected to be similar in meaning. The *word2vec* algorithm uses a neural network model to learn word associations from a large corpus of text, as shown by [T. Mikolov et al. (2013)](https://arxiv.org/pdf/1301.3781.pdf). If you want to know more, check out [this blog post](https://towardsdatascience.com/word2vec-research-paper-explained-205cb7eecc30).



Now we're ready to explore the Encoder-Decoder architecture for machine translation (introduced by [K. Cho, et al. (2014)](https://arxiv.org/abs/1406.1078)), shown in *Figure 2*. This Encoder-Decoder architecture comprises one recurrent layer for the encoder and one for the decoder. Each RNN layer is, in reality, composed of a solitary rolled RNN cell that unrolls following the number of time steps. [This article](https://towardsdatascience.com/all-you-need-to-know-about-rnns-e514f0b00c7c) explains well how the RNN layer processes an input sequence. At each time step, the encoder RNN cell gets a new sequence element and the internal state computed in the previous time step (the initial hidden state is typically a vector of zeros) and outputs a new component for the output sequence and the updated internal state. We are interested in the internal state of the last time step, obtained with the processing of input $x_3$. This fixed-length *encoder vector*, as shown in *Figure 2*, encodes all the information of the input sequence. The vector is then used to initialize the decoder network's internal state, which uses its outputs to unroll and give the final output sequence $y_1$, $y_2$. 


| <img src="/assets/2023-06-29-transformers/encoder-decoder.jpg" alt="encoder-decoder" width="500"/>| 
|:--:|                 
| *Figure 2*: the diagram shows a simple Autoencoder architecture composed of one recurrent layer for the encoder and one recurrent layer for the decoder. Credits to [towardsdatascience.com](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346). |

Limitations
===========
As shown by [Bengio, et al. (1994)](https://www.researchgate.net/publication/5583935_Learning_long-term_dependencies_with_gradient_descent_is_difficult) the central issue of this architecture is that RNNs encounter difficulties in learning to establish long-term dependencies. This capability is crucial in NLP, as sentences can be pretty long, and remembering information given at the beginning of the sentence can significantly impact solving, for example, a prediction task. 

A variant of the RNN model, called Long-Short Memory Network (LSTM), was introduced by [Hochreiter & Schmidhuber ([1997)](https://www.bioinf.jku.at/publications/older/2604.pdf), and it was designed for learning long-term dependencies. [Colah's blog post on LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs) explains the inner mechanism of LSTM cells. Autoencoders based on LSTM layers set a new standard for sequence-to-sequence learning, as shown in the work of [I. Sutskever, et al. (2014)](https://arxiv.org/abs/1409.3215). 

Before introducing the Transformer architecture, let's highlight the two main limitations of recurrent-based models:

1. Although LSTMs mitigated the problem of learning long-term dependencies, they have not entirely addressed it. For very long sequences, they still struggle. This problem is inherently related to the recursion nature of the architecture: information in RNNs and LSTMs is retained thanks to previously computed hidden states, but they are updated at each time step, decreasing the influence of past words. Furthermore, in the Encoder-Decoder architecture, only the last hidden state is passed to the decoder network, and researchers realized that this fixed-length vector was a bottleneck for these models. 
2. In recurrent models, the processing of the input sequences is sequential and can not be parallelized. Hence, LSTMs can't be trained in parallel.  

Transformers
============
The *Transformer* architecture, introduced in a paper titled *"Attention is all you need"* by [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf), made its debut in machine translation to tackle two main objectives:

1. Minimize performance degradation caused by long-term dependencies. 
2. Eliminate recursion for parallel computation resulting in reduced training time. 

Rather than relying on past hidden states to capture dependencies with previous words, transformers take a different approach by processing an entire sentence. The latter is the main reason why transformers do not suffer from long-term dependency issues. In addition, processing an entire sentence as a whole means that they do not require sequential processing and can capture dependencies in a parallel manner. 

Two innovations make all of this possible: *Self-Attention* and *Positional Embeddings*. 


#### Attention and Self-Attention

*Self-Attention* was a novel contribution of [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf), but the original *Attention* technique was introduced before by [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) and [Luong et al. (2015)](https://arxiv.org/abs/1508.04025). Since *Self-Attention* is based on the *Attention* technique, let's understand it first in the context of *neural machine translation*. 

As said before, in the Encoder-Decoder architecture with recurrent layers, only the last hidden state is passed to the decoder network, and researchers realized that this fixed-length vector was a bottleneck for these models. The *Attention* technique has been introduced to solve this problem. Let's understand how it works. In the Encoder-Decoder architecture with *Attention*, the encoder passes to the decoder all the hidden states computed at each time step. To cope with this amount of data, the decoder network uses the *Attention* technique to focus on the parts of the input that are relevant to the ith decoding time step. In other words, the *Attention* technique addresses the challenge of aligning the source sentence's relevant parts with the words generated by the decoder. *Figure 3* shows this exact process. 

| <img src="/assets/2023-06-29-transformers/attention.gif" alt="attention" />| 
|:--:|                 
| *Figure 3*: the diagram shows the *Attention* technique. Took inspiration from the [jalammar's blog](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/). |

Let's describe all the steps, and remember that the decoder network has access to every hidden state ($h_1$, $h_2$, $h_3$) generated by the encoder network. The decoder starts processing when it gets the `<END>` token. As a standard decoder, it begins with an initial hidden state $h_{init}$ and generates a new hidden state $h_4$. The *Attention* technique is applied to create the first output word $w_4$ in the following way:
1. The decoder gathers its hidden state ($h_4$) and the encoder's hidden states ($h_1$, $h_2$, $h_3$). 
2. A scoring function measures the relevance (or alignment) between the decoder's current hidden state and each of the encoder's hidden states. In other words, these similarity scores allow the *Attention* mechanism to focus dynamically on different parts of the source sentence during the decoding process. There are several ways to compute this score: as the dot product between the two vectors $score(h_t, h_i) = h_t * h_i$, or in the *general scoring* approach: $score(h_t, h_i) = h_t * W * h_i$ (as in *Figure 3*), where $W$ is a learnable weight matrix that transforms the hidden state vectors before taking their dot product. The latter allows the model to learn more complex alignments between the encoder and decoder hidden states. 
3. Once the scores are computed for each encoder hidden state, they are usually passed through a softmax function to obtain a probability distribution over the hidden states. The softmax function ensures that the scores sum up to 1, representing the importance or **attention weights** assigned to each hidden state. 
4. These attention weights are then used to compute a weighted sum of the encoder hidden states ($C_4$), amplifying hidden states with high scores and drowning out hidden states with low scores.  
5. The weighted sum ($C_4$) is concatenated with the decoder's hidden state ($h_4$), and this new vector is passed through a feedforward neural network (trained jointly with the model). 
6. The output of the feedforward neural networks indicates the output word of this time step.
7. This process is repeated for each time step.

*Figure 4* shows the *Attention* technique's powerful effectiveness when translating a sentence from French to English. We can see from the attention matrix that when the model outputs *"european economic area"* it pays attention to the correct input words that, in this case, are reversed (*"zone économique européenne"*). This alignment between French and English is learned from the training phase. 

| <img src="/assets/2023-06-29-transformers/attention_sentence.jpg" alt="attention-sentence"  width="400"/>| 
|:--:|                 
| *Figure 4*: *Attention* effectiveness for Machine Translation. Credits: [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf). |

Now let's focus on *Self-Attention*. These two techniques share the same core concepts but are applied on slightly different inputs. *Attention* is used in the Encoder-Decoder architecture with RNNs layers, applied to the encoder's hidden states and decoder's hidden states that are obtained sequentially. On the other hand, the *Transformer* architecture (in the original paper, still composed by an encoder and a decoder) processes an entire sentence as a whole and applies *Attention* to the input embeddings only (that's why it's called *Self-Attention*). The main idea is to combine the embedding of the input vectors (one for each token of the sequence) in a linear combination fashion $x_i' = \sum_{j=1}^{n} w_{ij}x_j$ where $w_{ij}$ are the attention weights and ${x_j}$ the input embeddings, in order to generate new contextualized embeddings ${x_i'}$ that are enriched with the context of the sentence. This is crucial to address the ambiguity of the language! Let's think about the word *watch*. The only way to be sure of its meaning is to look at the context. In the sentence:

> He received a watch as a gift for his birthday. 

We understand that *watch* is a noun representing the time-tracker device on your wrist, not the verb *to watch*. So, how can we add context to the input embeddings? 

There're several ways, but the most common is the *scaled dot-product attention* introduced in [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf) and shown in *Figure 5*. 

| <img src="/assets/2023-06-29-transformers/self_attention.gif" alt="self-attention" />| 
|:--:|                 
| *Figure 5*: the diagram shows the structure of a *Self-Attention* head. |

Let's break it down: 
- We start by projecting the input embeddings using three separate linear projections with learnable weights. These linear projections transform the input embeddings into different spaces, enabling the model to effectively attend to different parts of the sequence. 
- The projected Query (Q) and Key (K) vectors are multiplied together (K is transposed to allow the matrix multiplication). The output is a $n*n$ matrix of attention scores where $n$ is the number of tokens in the input sequence. The dot products of the matrix multiplications can be seen as a similarity function: a small attention score means that the two embeddings don't share much in common.  
- Since dot products can, in general, produce arbitrarily large numbers and undermine the training process, the output is first scaled by a factor $sqrt(768)$ (where 768 is the embedding dimension in this example) to normalize the variance and then a normalization is applied with the softmax function. The output matrix is called *attention weights*.  
- Finally, the projected V embeddings are mixed together, computing a linear combination using the attention weights as coefficients: $v_i' = \sum_{j=1}^{n} w_{ij}v_j$  

*Figure 6* visualizes individual neurons in the query and key vectors and shows how they are used to compute the attention weights for the word *watch*.

| <img src="/assets/2023-06-29-transformers/bertviz.jpg" alt="attention-weights-visualization" />| 
|:--:|                 
| *Figure 6*: The attention weights for the word *watch*. Generated with: [Bertviz](https://github.com/jessevig/bertviz). |


Excellent, now we understand how to build a Self-Attention module. But something has not been explained yet. It has been said that the linear layers project the input embeddings into three different spaces. The dimension of these spaces is computed as $head\\_dim=768/att\\_heads\\_num$. The variable $att\\_heads\\_num$ represents the number of Self-Attention heads. It turned out to be beneficial to have multiple Self-Attention heads. Having multiple heads enables the model to concentrate simultaneously on different aspects. For instance, one head can emphasize the interaction between the subject and verb, while another identifies nearby adjectives. *Figure 7* shows the structure of a multi-attention head.  


| <img src="/assets/2023-06-29-transformers/multi_head.gif" alt="multi-attention-head" />| 
|:--:|                 
| *Figure 7*: The multi-attention head |

As we can see from the diagram above, each contextualized embedding with shape $[batch\\_size, seq\\_len, head\\_dim]$ is concatenated on the last dimension. The concatenation has the original dimension of the input embeddings, i.e., 
$[batch\\_size, seq\\_len, head\\_dim * att\\_heads\\_num]$. So, if we choose to have $12$ attention heads, the linear layers will project the input embeddings of dimension $768$ into $768/12=64$ dimensions (this is done to maintain constant the computation across each head). The dimension of the contextualized embeddings after the concatenation will be $batch\\_size, seq\\_len, 12*64]=[batch\\_size, seq\\_len, 768]$. Finally, the concatenation of the contextualized embeddings is passed to a linear layer to provide additional flexibility and non-linearity to the model's output. 






#### Positional Embeddings
