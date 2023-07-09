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

The *Transformer* architecture defined a new standard for modern neural network design, leading to the development of the current state-of-the-art models such as *GPT*, *BERT*, *CLIP* and enabling the training of robust multi-modal architectures. This post assumes a basic understanding of the Auto-Encoder architecture and recurrent models.

<a target="_blank" href="https://www.amazon.it/Natural-Language-Processing-Transformers-Applications/dp/1098136799/ref=sr_1_1?keywords=natural+language+processing+with+transformers&amp;qid=1688565977&amp;sprefix=Natural+lan%252Caps%252C113&amp;sr=8-1&_encoding=UTF8&tag=leobaro-21&linkCode=ur2&linkId=036ca3124da4daab7b40fe2a7c8e21de&camp=3414&creative=21718">🔥🔥🔥 One of the best book on Transformers, NLP and 🤗Hugging Face 🔥🔥🔥</a>


Sequence-to-sequence modeling
=============================

One categorization of deep learning models considers the nature of the input and the corresponding output of a neural network architecture. The input or output can be a single vector (with $N$ dimensions i.e. *features*) or a sequence of vectors. Let's give some examples: a single RGB image is represented by a tensor with shape $(height, weight, channels)$, while a single word in a sentence can be defined with a tensor of shape $(v)$ where $v$ is the vocabulary size. Both of these examples are not sequences. On the other hand, a video is a sequence of images, and a sentence is a sequence of words. *Figure 1* discriminates all combinations of neural architectures. Let's start from the leftmost panel: it shows a *one-to-one* architecture; a single tensor goes in, and a single tensor comes out. The latter is the case of *Image Classification*, in which a single image is passed to the network to produce a vector of probabilities (one for each class). In the diagram of the second panel, a vector goes in, and a sequence comes out. *Image Captioning* is one task that agrees to this representation of input and output: a single image is given as input, and the model generates a sentence (sequence of words) describing the image. Vice versa, *many-to-one* is a task addressed by generative models which accept a natural language description and generate an image that follows that description. Finally, the *many-to-many* approach is used in *Machine Translation* where a sentence in a given language is translated to another. The last panel still describes the *many-to -many* scenario, but this time with synched input and output, for example, in *Video Classification*, in which each frame is associated with a vector of probabilities predicting the frame's label. 

| ![dl-architectures](/assets/2023-06-29-transformers/dl-architectures.jpg)| 
|:--:|                 
| *Figure 1*: Each rectangle represent a vector. Red vectors are the input blue vectors are the output and green vectors represent holds the neural network states. Credits: [Andrej Karpathy's blog](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).|

> :bulb: **Let's focus on *many-to-many* architectures and the *Machine Translation* task since the essential building block of the Transformer architecture, the attention mechanism, stemmed from the research in this field.**

Before Attention and Transformer, Recurrent Neural Networks (RNN) and then Long-Short Memory Networks (LSTM), introduced by [S. Hochreiter, J. Schmidhuber (1997)](https://direct.mit.edu/neco/article-abstract/9/8/1735/6109/Long-Short-Term-Memory?redirectedFrom=fulltext), were used to build models that achieved the state of the art scores on benchmark datasets. These models were based on the Encoder-Decoder architecture. 

Encoder-Decoder architecture for sequences
==========================================
Neural network Encoder-Decoder architectures are designed to have a hidden layer that acts like a bottleneck, forcing the encoder part to learn how to compress the information. The decoder part of the network starts from this representation to solve a particular task. These kinds of architecture are widely used in several fields, from *Machine Vision* to *Natural Language Processing* (NLP). Both encoder and decoder networks can be implemented with different types of layers regarding the input data type and the representation learning capabilities we want to give to the network. Typically, in *Machine Vision*, the encoder and decoder's layers are convolutional to reduce the number of parameters and to exploit the locality features of an image. If the input is a sequence, recurrent layers give the Auto-Encoder the capability of processing sequences of arbitrary length and exploit temporality. This is the case of NLP, in which sentences are a sequence of words. 

But how can we input words represented as strings into a neural network? Before diving into the Encoder-Decoder architecture for sequences, let's understand how to convert text into a more suitable format for neural networks. In a nutshell, each sentence is the input of a *Tokenizer* whose purpose is to apply tokenization and numericalization. Finally, an *Embedder* transforms each token into a vector representation. Let's break these steps down with an example (using the HuggingFace library).

*Tokenization* is a way of separating a piece of text into smaller units called tokens:
{% highlight python %}
from transformers import DistilBertTokenizerFast, DistilBertModel

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

tokens = tokenizer.encode('Run, rabbit run. Dig that hole, forget the sun.', return_tensors='pt')

print(tokens)
#=> prints ['[CLS]', 'run', ',', 'rabbit', 'run', '.', 'dig', 'that', 'hole', ',', 'forget', 'the', 'sun', '.', '[SEP]']
{% endhighlight %}

*Numericalization* converts each unique token to an unique number:
{% highlight python %}
decoded_tokens = tokenizer.convert_ids_to_tokens(enc.input_ids)
print(decoded_tokens)
#=> prints tensor([[101,  2448,  1010, 10442,  2448,  1012, 10667,  2008,  4920,  1010, 5293,  1996,  3103,  1012,   102]])
{% endhighlight %}

Finally, the *Encoding* step converts numbers into vectors:
{% highlight python %}
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

print(model.embeddings.word_embeddings(tokens))

#=> prints 
#=>  tensor([[[ 0.0390, -0.0123, -0.0208,  ...,  0.0607,  0.0230,  0.0238],
#=>           [ 0.0449, -0.0244, -0.0157,  ..., -0.0461, -0.0771, -0.0006],
#=>           [-0.0111, -0.0141, -0.0029,  ...,  0.0023,  0.0018,  0.0093],
#=>           ...,
#=>           [-0.0097, -0.0020,  0.0337,  ..., -0.0339, -0.0130, -0.0300],
#=>           [-0.0244, -0.0138, -0.0078,  ...,  0.0069,  0.0057, -0.0016],
#=>           [-0.0199, -0.0095, -0.0099,  ..., -0.0235,  0.0071, -0.0071]]],
#=>       grad_fn=<EmbeddingBackward0>)

print(model.embeddings.word_embeddings(tokens)[0][0].shape)
#=> prints torch.Size([768])

{% endhighlight %}

The tokenization step is crucial since tokens are the building blocks of NLP. It can be applied to single characters, single words, or subwords. Character tokenization and word tokenization have drawbacks, so nowadays, the most common way to tokenize a sentence is to use subwords tokenization algorithms, such as WordPiece by [M. Schuster, K. Nakajima (2012)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf). The subword tokenization is learned from the pretraining corpus with statistical rules and algorithms. If you want to know more, check out [this blog post](https://towardsdatascience.com/tokenization-for-natural-language-processing-a179a891bad4). Finally, an embedding is a real-value representation of a word that encodes the semantics and syntactic of the word so that words closer in the vector space are expected to be similar in meaning. The *word2vec* algorithm uses a neural network model to learn word associations from a large corpus of text, as shown by [T. Mikolov et al. (2013)](https://arxiv.org/pdf/1301.3781.pdf). If you want to know more, check out [this blog post](https://towardsdatascience.com/word2vec-research-paper-explained-205cb7eecc30).

Now we're ready to explore the Encoder-Decoder architecture, introduced by [K. Cho, et al. (2014)](https://arxiv.org/abs/1406.1078), to translate a sentence in another language. The Encoder-Decoder architecture comprises one recurrent layer for the encoder and one for the decoder (*Figure 2*). Each RNN layer is, in reality, composed of a solitary rolled RNN cell that unrolls following the number of time steps. [This article](https://towardsdatascience.com/all-you-need-to-know-about-rnns-e514f0b00c7c) explains well how the RNN layer processes an input sequence. In a nutshell, at each time step the encoder RNN cell gets a new sequence element and the internal state computed in the previous time step (the initial hidden state is typically a vector of zeros) and outputs a new component for the output sequence and the updated internal state. We are interested in the internal state of the last time step, obtained with the processing of input $x_3$. This fixed-length *encoder vector* encodes all the information of the input sequence. The vector is then used to initialize the decoder network's internal state, which uses its outputs to unroll and give the final output sequence $y_1$, $y_2$. 


| <img src="/assets/2023-06-29-transformers/encoder-decoder.jpg" alt="encoder-decoder" width="800"/>| 
|:--:|                 
| *Figure 2*: the diagram shows a simple Autoencoder architecture composed of one recurrent layer for the encoder and one recurrent layer for the decoder. Credits to [towardsdatascience.com](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346). |

Limitations
===========
As shown by [Bengio, et al. (1994)](https://www.researchgate.net/publication/5583935_Learning_long-term_dependencies_with_gradient_descent_is_difficult) the central issue of this architecture is that RNNs encounter difficulties in learning long-term dependencies. This capability is crucial in NLP, as sentences can be pretty long, and remembering information given at the beginning of the sentence can significantly impact solving, for example, a prediction task. 

A variant of the RNN model, called Long-Short Memory Network (LSTM), introduced by [Hochreiter & Schmidhuber, (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf), was designed for learning long-term dependencies. [Colah's blog post on LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs) explains the inner mechanism of LSTM cells. Autoencoders based on LSTM layers set a new standard for sequence-to-sequence learning, as shown in the work of [I. Sutskever, et al. (2014)](https://arxiv.org/abs/1409.3215). 

Before introducing the *Transformer* architecture, let's highlight the two main limitations of recurrent-based models:

1. Although LSTMs mitigated the problem of learning long-term dependencies, they have not entirely addressed it. For very long sequences, they still struggle. This problem is inherently related to the recursion nature of the architecture: information in RNNs and LSTMs is retained thanks to previously computed hidden states, but they are updated at each time step, decreasing the influence of past words. Furthermore, in Encoder-Decoder architectures, only the last hidden state is passed to the decoder network, and researchers realized that this fixed-length vector was a bottleneck for these models. 
2. In recurrent models, the processing of the input sequences is sequential and can not be parallelized. Hence, LSTMs can't be trained in parallel.  

Transformers
============
The *Transformer* architecture, introduced in a paper titled *"Attention is all you need"* by [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf), made its debut in machine translation to tackle two main objectives:

1. Minimize performance degradation caused by long-term dependencies. 
2. Eliminate recursion for parallel computation resulting in reduced training time. 

> :bulb: **Rather than relying on past hidden states to capture dependencies with previous words, transformers take a different approach by processing an entire sentence as a whole. Hence, they do not suffer from long-term dependency issues nor require sequential processing and they capture dependencies between words in a parallel manner.**

Two innovations make all of this possible: **self-attention** and **positional embeddings**.


#### Attention and Self-Attention

*Self-attention* was a novel contribution of [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf), but the original *Attention* technique was introduced before by [Bahdanau et al. (2014)](https://arxiv.org/abs/1409.0473) and [Luong et al. (2015)](https://arxiv.org/abs/1508.04025). Since *self-attention* is based on the *Attention* technique, let's understand it first in the context of *neural machine translation*. 

As said before, in the Encoder-Decoder architecture with recurrent layers, only the last hidden state is passed to the decoder network, and researchers realized that this fixed-length vector was a bottleneck for these models. The *Attention* technique has been introduced to solve this problem. Let's understand how it works. In the Encoder-Decoder architecture with *Attention*, the encoder passes to the decoder all the hidden states computed at each time step, and not only the last state. To cope with this amount of data, the decoder network uses the *Attention* technique to focus on the parts of the input that are relevant to the *ith* decoding time step. In other words, the *Attention* technique addresses the challenge of aligning the source sentence's relevant parts with the words generated by the decoder. *Figure 3* shows this exact process. 

| <img src="/assets/2023-06-29-transformers/attention.gif" alt="attention" />| 
|:--:|                 
| *Figure 3*: the diagram shows the *Attention* technique. Took inspiration from the [jalammar's blog](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/). |

Let's describe all the steps, and remember that the decoder has access to every hidden state ($h_1$, $h_2$, $h_3$) generated by the encoder. The decoder starts when it receives the `<END>` token. It begins with an initial hidden state $h_{init}$ and generates a new hidden state $h_4$. The first output word $w_4$ is generated in the following way using attention:
1. The decoder gathers its hidden state ($h_4$) and the all the encoder's hidden states ($h_1$, $h_2$, $h_3$). 
2. A scoring function measures the relevance (or alignment) between the decoder's current hidden state and each of the encoder's hidden states. In other words, these similarity scores allow the *Attention* mechanism to focus dynamically on different parts of the source sentence during the decoding process. There are several ways to compute this score: as the dot product between the two vectors $score(h_t, h_i) = h_t * h_i$, or in the *general scoring* approach: $score(h_t, h_i) = h_t * W * h_i$ (as in *Figure 3*), where $W$ is a learnable weight matrix that transforms the hidden state vectors before taking their dot product. The latter allows the model to learn more complex alignments between the encoder and decoder hidden states. 
3. Once the scores are computed for each encoder hidden state, they are usually passed through a softmax function to obtain a probability distribution over the hidden states. The softmax function ensures that the scores sum up to 1, representing the importance or **attention weights** assigned to each hidden state. 
4. These attention weights are then used to compute a weighted sum of the encoder hidden states ($C_4$), amplifying hidden states with high scores and drowning out hidden states with low scores.  
5. The weighted sum ($C_4$) is concatenated with the decoder's hidden state ($h_4$), and this new vector is passed through a feedforward neural network (trained jointly with the model). 
6. The output of the feedforward neural networks indicates the output word of this time step.
7. This process is repeated for each time step.

*Figure 4* shows the *Attention* technique's powerful effectiveness when translating a sentence from French to English. The attention matrix shows that when the model outputs *"european economic area"* it pays attention to the correct input words that, in this case, are reversed (*"zone économique européenne"*). This alignment between French and English is learned from the training phase. 

| <img src="/assets/2023-06-29-transformers/attention_sentence.jpg" alt="attention-sentence"  width="400"/>| 
|:--:|                 
| *Figure 4*: *Attention* effectiveness for Machine Translation. Credits: [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf). |

Now let's focus on *self-attention*. These two techniques share the same core concepts but are applied on slightly different inputs. *Attention* is used in the Encoder-Decoder architecture with recurrent layers, and it's applied to the encoder's hidden states with the decoder's hidden states that are obtained sequentially. On the other hand, the *Transformer* architecture (in the original paper, still composed by an encoder and a decoder) processes an entire sentence as a whole and applies *Attention* to the encoder's input embeddings only (or decoder's ones). That's why it's called **self**-attention. The main idea is to combine the embeddings of the input vectors (one for each token of the sequence) in a linear combination fashion $x_i' = \sum_{j=1}^{n} w_{ij}x_j$ where $w_{ij}$ are the attention weights and ${x_j}$ the input embeddings, in order to generate new contextualized embeddings ${x_i'}$ that are enriched with the context of the sentence. This is crucial to address the ambiguity of the language! Let's think about the word *watch*. The only way to be sure of its meaning is to look at the context. In the sentence:

> He received a watch as a gift for his birthday. 

We understand that *watch* is a noun representing the time-tracker device on your wrist, not the verb *to watch*. So, how can we add context to the input embeddings? 

There're several ways, but the most common is the *scaled dot-product attention* introduced in [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf) and shown in *Figure 5*. 

| <img src="/assets/2023-06-29-transformers/self_attention.gif" alt="self-attention" />| 
|:--:|                 
| *Figure 5*: the diagram shows the structure of a *self-attention* head. |

Let's break it down: 
- It starts by projecting the input embeddings using three separate linear projections with learnable weights. These linear projections transform the input embeddings into different spaces, called Query ($Q$), Key ($K$) and Value ($V$) vectors.
- The projected $Q$ and $K$ vectors are multiplied together ($K$ is transposed to allow the matrix multiplication). The output is a $n*n$ matrix of attention scores where $n$ is the number of tokens in the input sequence. The dot products of the matrix multiplications can be seen as a similarity function: a small attention score means that the two embeddings don't share much in common.  
- Since dot products can, in general, produce arbitrarily large numbers and undermine the training process, the output is first scaled by a factor $sqrt(embed\\_dim)$ (the embedding dimension, 768 in this example) to normalize the variance and then a normalization is applied with the softmax function. The output matrix is called *attention weights*.  
- Finally, the projected Value ($V$) embeddings are mixed together, computing a linear combination using the attention weights as coefficients: $v_i' = \sum_{j=1}^{n} w_{ij}v_j$  

*Figure 6* visualizes individual neurons in the $Q$ and $K$ vectors and shows how they are used to compute the attention weights for the word *watch*.

| <img src="/assets/2023-06-29-transformers/bertviz.jpg" alt="attention-weights-visualization" />| 
|:--:|                 
| *Figure 6*: The attention weights for the word *watch*. Generated with: [Bertviz](https://github.com/jessevig/bertviz). |

A way to explain the Key/Value/Query mechanism is to think about retrieval systems. When you search for videos on Youtube for example, $Q$ is the text in the search bar and $K$ represent the metadata for every candidate video (title, description, tags and so on). The search engine will compute the similarity between $Q$ and $K$ and give you $V$, i.e. the list of the best matched videos.

**Excellent!** Now we understand how to build a *self-attention* module. But something has not been explained yet. It has been said that the linear layers project the input embeddings into three different spaces. The dimension of these spaces is computed as $head\\_dim=embed\\_dim/att\\_heads\\_num$. The variable $att\\_heads\\_num$ represents the number of *self-attention* heads. It turned out to be beneficial to have multiple *self-attention* heads. 
> :bulb: **Having multiple self-attention heads enables the model to concentrate simultaneously on different aspects.**

For instance, one head can emphasize the interaction between the subject and verb, while another identifies nearby adjectives. *Figure 7* shows the structure of a multi-attention head.  


| <img src="/assets/2023-06-29-transformers/multi_head.gif" alt="multi-attention-head" />| 
|:--:|                 
| *Figure 7*: The multi self-attention head |

As we can see from the diagram above, each contextualized embedding with shape $[batch\\_size, seq\\_len, head\\_dim]$ is concatenated on the last dimension. The concatenation has the original dimension of the input embeddings, i.e., 
$[batch\\_size, seq\\_len, head\\_dim * att\\_heads\\_num]$. So, if we choose to have $12$ attention heads, the linear layers will project the input embeddings of dimension $768$ into $768/12=64$ dimensions (this is done to maintain constant the computation across each head). The dimension of the contextualized embeddings after the concatenation will be $batch\\_size, seq\\_len, 12*64]=[batch\\_size, seq\\_len, 768]$. Finally, the concatenation of the contextualized embeddings is passed to a linear layer to provide additional flexibility and non-linearity to the model's output. 

Finally, *Figure 8* shows another visualization to explain self-attention.


| <img src="/assets/2023-06-29-transformers/self_attention_google.gif" alt="self-attention" width="400"/>| 
|:--:|                 
| *Figure 8*: The input embeddings are represented by the unfilled circles. Self-attention aggregates information from all of the other words, generating a new representation per word (filled balls) informed by the entire context. This step is then repeated multiple times in parallel for all words, successively generating new representations. The decoder operates similarly, but generates one word at a time, from left to right. It attends not only to the other previously generated words, but also to the final representations generated by the encoder. Credits: [https://ai.googleblog.com](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) |

#### Positional Embeddings

The multi-head self-attention module treats tokens as interchangeable, regardless of their positions in the sequence. In simpler terms, a transformer model would understand the same meaning from the sentences "I love pizza." and "love I pizza,", because it focuses solely on the content of the words. It doesn't rely on the specific order or position of the words to comprehend the sentence, and this property is called *permuation equivariant*.
> :bulb: **Let's enrich the tokens embedding with positional information!**

To account for the order of the words, the most popular way is to add another embedding layer with trainable parameters and sum the token embeddings with the positional embedding before passing the result to the multi-head self-attention module. This new layer uses the token position index instead of the token ID. During the model pre-training, it learns how to encode the position of the tokens. 

| <img src="/assets/2023-06-29-transformers/embeds.jpg" alt="embeddings" />| 
|:--:|                 
| *Figure 9*: Positional embeddings |

While this approach is straightforward to implement, there are two main variants:
1. *Absolute positional representations*: the position of the tokens can be encoded with static patterns consisting of modulated sine and cosine signals.
2. *Relative positional representations*: to give more importance to surrounding tokens when computing an embedding. [P. He, et al. (2020)](https://arxiv.org/abs/2006.03654) implemented it modifying of the self-attention mechanism.   

The *Transformer* architecture
==============================
Until now, we understood the core components of the *Transformer* model, but the goal of this blog post is to explain the whole architecture. *Figure 10*, taken from [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf) shows the *Transformer* architecture.

| <img src="/assets/2023-06-29-transformers/transformer_arch.jpg" alt="Attention-is-all-you-need" />| 
|:--:|                 
| *Figure 10*: the *Transformer* architecture from [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf). Annotations: [lena-voita.github.io](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) |

Let's build the encoder and decoder modules separately.


#### The Encoder
The _Transformer_ architecture **comprises a stack (N=6) of encoders modules**. We must introduce three more components to build our first encoder module: the feed-forward network, layer normalization, and skip connections. 

| <img src="/assets/2023-06-29-transformers/encoder.jpg" alt="encoder" width="200"/>| 
|:--:|             
*Figure 11*: The encoder module. Credits: [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf) |


##### **Position-Wise Feed-Forward Network**

Let's start with the position-wise feed-forward network: **it processes each contextualized embedding independently**. It comprises two fully-connected layers, called *position-wise feed-forward layers*, whose hidden size is four times the size of the embeddings, and it uses the _GELU_ activation, followed by a _Dropout_ layer. It performs a convolution-like operation with a kernel size of 1 (i.e., applying a linear transformation to the attended features). This operation helps capture complex patterns and interactions between different attended features, enhancing the model's ability to learn and represent intricate relationships within the data.

| <img src="/assets/2023-06-29-transformers/feed_forward.jpg" alt="feed-forward" width="700" />| 
|:--:|                 
| *Figure 12*: this illustration highlights how the position-wise property of the feed-forward layer.  Credits: [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/) |

##### **Layer normalization and Skip connections**
***Layer normalization*** is a technique introduced by [J.L. Ba, et al. (2016)](https://arxiv.org/abs/1607.06450) to reduce the training time by normalizing the activities of the neurons. Unlike *batch normalization* ([S. Ioffe, C. Szegedy, (2015)](https://arxiv.org/abs/1502.03167)), it directly estimates the normalization statistics from the summed inputs to the neurons within a hidden layer without introducing new dependencies between training cases. 

$$\mu = \frac{1}{H} \sum_{i=1}^{H} x_i$$

$$\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2$$

where $x_i$ represents the input activations for a particular layer, $H$ represents the number of hidden units in that layer, $\mu$ represents the mean, and $\sigma^2$ represents the variance. These statistics are used to normalize the input activations by subtracting the mean $\mu$ and dividing by the square root of the variance $\sigma$. 

***Skip connections*** (or *residual connections*, introduced in [K. He, et al. (2016)](https://arxiv.org/pdf/1512.03385v1.pdf)) are connections that allow the direct flow of information from one layer to another, bypassing certain intermediate layers in a neural network, facilitating the flow of gradients during training. 

As we can see from *Figure 11* the skip connections and the layer normalization are applied twice, right after the multi-head self-attention module and right after the position-wise feed-forward module. This configuration is called *Post layer normalization*, the same configuration introduced in [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf). The most common arrangment is called *Pre layer normalization*, shown in the bottom panel of *Figure 13*, and it tends to be more stable during training with respect of the original one. 

 | <img src="/assets/2023-06-29-transformers/norm.jpg" alt="Post-layer normalization and Pre-layer normalization" width="500"/>| 
|:--:|                 
| *Figure 13*: Post layer normalization and Pre layer normalization. Credits: [Natural Language Processing with Transformers: Building Language Applications with Hugging Face (2022)](https://www.amazon.it/Natural-Language-Processing-Transformers-Applications/dp/1098136799/ref=sr_1_1?keywords=natural+language+processing+with+transformers&amp;qid=1688565977&amp;sprefix=Natural+lan%252Caps%252C113&amp;sr=8-1&_encoding=UTF8&tag=leobaro-21&linkCode=ur2&linkId=036ca3124da4daab7b40fe2a7c8e21de&camp=3414&creative=21718) |


#### The Decoder

We're close to the end of this journey! We need to introduce the decoder network, but fortunatly, it reuses almost the same components we discussed. 

| <img src="/assets/2023-06-29-transformers/decoder.jpg" alt="decoder" width="200"/>| 
|:--:| 
| *Figure 14*: Decoder network. Credits: [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf) |

The objective of the decoder network is to predict the next word in a sentence given the words it has already predicted and the contextualized embedding from each encoder layer. The decoder is also composed of a stack of $N=6$ identical layers. The main difference from the encoder network is the presence of two slightly different attention layers: a *masked multi-head self-attention layer* and an *encoder-decoder attention layer*.

##### **What is masking, and why do we need it?**
Masking means hiding certain parts of the input embeddings before the self-attention module processes them. This pre-processing step is not needed during inference, but it's needed during the training phase. Let's understand why. 

We understood that the encoder network can process the input embeddings in parallel, but what about the decoder? During inference (the correct output sequence is not known) the decoder works in sequential mode, like a RNN:
- it receives from the encoder, the contextualized embeddings;
- it looks at all the previous outputs;
- it predicts the next word until it generates the `<eos>` token (short for *end of sentence*).

As highlighted from [artoby's post on StackOverflow](https://stackoverflow.com/questions/58127059/how-to-understand-masked-multi-head-attention-in-transformer), during training, since the correct output sequence is known, we could train the decoder in parallel: we just need to give as input all the correct embeddings but masked appropriately considering the time step. For example, if we want to translate `I love you` to German, we could execute in parallel the following decoder's steps:
- Input decoding step 0:  `---    -----  ----`
- Input decoding step 1:  `Ich    -----  ----`
- Input decoding step 2:  `Ich    liebe  ----`

If we don't mask the input, the decoder's self-attention module could cheat, attending to subsequent words! By masking and shifting the output embeddings by one position, two factors work in conjunction to maintain the auto-regressive nature of the decoder. Firstly, the masking prevents the predictions for a given position, denoted as $i$, from relying on any future positions beyond $i$. Secondly, offsetting the output embeddings guarantees that the predictions at position $i$ only depend on the known outputs at positions preceding $i$.


##### **Encoder-Decoder attention**
The animation in *Figure 15* shows how the decoding stage works. In the *Encoder-Decoder attention* layers, the $Q$ matrix originates from the preceding decoder layer, while the $K$ and $V$ matrices are extracted from the Encoder's output. As a result, each position in the decoder can focus on every position within the input sequence.

| <img src="/assets/2023-06-29-transformers/decoding.gif" alt="transformer decoding step" />| 
|:--:| 
| *Figure 15*: The decoding step of the decoder network. Credits: [Alammar, J (2018). The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer) |



#### The final head

*Figure 16* shows the two final layers that process the Decoder network's output and predict the next word. The linear layer is a fully connected neural network whose size equals vocabulary size. The softmax layer assigns a probability for each possible word predicted by the linear layer. The word with the highest probability is chosen to be the following word in the sentence for the current time step.

| <img src="/assets/2023-06-29-transformers/final_head.jpg" alt="linear and softmax layers" width="200"/>| 
|:--:| 
| *Figure 16*: The final layers of the *Transformer* architecture: linear and softmax. Credits: [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf) |


# Conclusion

I hope you liked the post! 

Now, it's time to get your hands dirty! Let the implementation begins 🤓

- [Karpathy's minGPT](https://github.com/karpathy/minGPT) and [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Harvard's implementation](https://nlp.seas.harvard.edu/annotated-transformer/#full-model) of [A. Vaswani, et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf) 
