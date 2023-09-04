---
layout: post
title:  "An introduction on multimodal learning"
date:   2023-08-01 10:00:00 +0200
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
This blog post gives an introduction to the field of Multimodal Machine Learning. In practice, these are my notes from the course held by Carnegie Mellon University: [11-777 Multimodal Machine Learning, 2022 Fall](https://cmu-multicomp-lab.github.io/mmml-course/fall2022/) and the review paper of [P. Liang, et al. (2023)](https://arxiv.org/abs/2209.03430).


Towards multimodality
=====================
Let's start by defining *modality*:
> :bulb: **A modality is a way something is expressed or perceived**

A multimodal dataset contains information spread over multiple modalities. Let's think about a recording of a public speech, for example. Utterances (audio modality) carry most of the information. Still, the speaker's body language or facial expressions (video modality) complement and enrich the audio information to convey the final message. Can machine learning models extract knowledge from multiple modalities?

As shown in *Figure 1*, multimodal machine learning has gained particular attention in the scientific community over the last ten years. 

| <img src="/assets/2023-08-01-multimodal/trend.jpg" alt="multimodal trend" width="800"/> | 
|:--:|                 
| *Figure 1*: The number of publications in the multimodal machine learning field over recent years. Credits: [Dimensions](https://app.dimensions.ai).|

In 2010, researchers started to use deep learning to exploit multimodal data. In the beginning with Deep Boltzmann Machines ([N. Srivastava, R. Salakhutdinov, 2012](https://papers.nips.cc/paper_files/paper/2012/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)) and then, [K. Xu et al. 2016](https://arxiv.org/abs/1502.03044) exploited the [attention mechanism](https://leobaro.github.io/deep-learning/2023/06/29/transformers.html) to bring multimodal research under the umbrella of computer vision and natural language processing for image captioning. After that moment, research in the multimodal modal domain exploded thanks to several key factors: the availability of new large-scale multimodal datasets and processing hardware but also advances in computer vision and language processing models enabled the representation of heterogeneous data (images, text), into homogeneous embedded vectors with meaningful features. As research on transformers, model optimization, and distributed training continued, as well as the availability of new multimodal datasets increased, multimodal deep learning spread over multiple fields. Let's think about robotics, which aims to develop intelligent autonomous agents capable of integrating and learning from different modalities. But also other real-world tasks such as multimodal sentiment and emotion recognition, multimodal QA, multimodal dialog, event recognition, multimedia information retrieval, etc.

The properties of multimodal data
=====================================
We're used to the unimodal data source, whose spectrum goes from raw (close to the sensor that captured it) to more abstract representations. For example, a microphone records speech (sound waves) that can be translated into written language (tokens) and analyzed to extract high-order descriptions such as sentiment. The representation spectrum is also in other modalities, such as images, and nothing stops us from integrating all this information, regardless of the abstraction level. Integrating different modalities closer to the raw spectrum is more challenging because they tend to be more heterogeneous than higher-level ones. **Heterogeneity** is a crucial concept in multimodal learning, and it's manifested along the following dimensions:
* **Representation** refers to the representation used in the sample space. Examples: set of characters (text) vs. matrixes (images) vs. nodes and edges (graphs). 
* **Distribution**: Different modalities have different distributions and frequencies of samples (the number of objects per image generally has a lower frequency than the number of words in a sentence).
* **Structure**: Images have spatial structure, while sentences have sequential structure. In addition, the latent space has a structure that can differ across modalities. 
* **Information**: the total information that can be extracted from a modality. It depends on the abstraction level: image data can be exploited from raw pixels to object categories.
* **Noise**: occlusion in images, typos in NLP, missing data, and so on.
* **Relevance**: to a specific task we want to address. 

So, although different modalities can be very different and they can bring their unique information, they can be *connected* by sharing complementary information. **Connection** is another crucial concept in multimodal learning. What is the form of these connections? We can see it from a bottom-up (statistical) or top-down view (semantic). Statistically speaking, elements that co-occur (cor)relate to each other, called *association*. On the semantic side, a *correspondence* is the presence of the same element (with the same semantic) in both elements of different modalities. The statistical *dependency* is a stronger connection type, which identifies causal relations. The equivalent in the semantic world is called *relationship*. TODO: examples. These two properties can help to define multimodal as:

> :bulb: **multimodal: the science of heterogeneous and interconnected data**

In particular, multimodal learning aims to exploit the complementary information from multiple connected modalities through their **Interaction**, the third key concept. It comes into play when the model integrates various modalities to perform inference. The result of the interaction could be **new information** (*Response* in *Figure 2*) that can help the model to make better predictions.  

| ![multimodal trend](/assets/2023-08-01-multimodal/inference.jpg)| 
|:--:|                 
| *Figure 2*: |

Regarding how the response changes, we can distinguish several types of interactions. We talk about **redundancy** if the modalities give similar answers. In this case, the multimodal response can be **enhanced** (higher confidence) or **equivalent** to the unimodal one (no interaction). What if the two modalities give different responses? For example, let's ask the question:

> **Is this a dangerous animal?**

🤔🤔🤔

| ![shark fin](/assets/2023-08-01-multimodal/interactions.jpg)| 
|:--:|                 
| *Figure 3*: non-redundant interactions.|

*Figure 3* shows a disagreement between the modalities. We have **dominance** if one modality response takes over the other (e.g., the multimodal response is "*yes*".). We have **independence** if the responses do not interact. **Modulation** is when one modality response enhances or reduces the other. Finally, the best scenario is when the **emergence** property gives birth to new information. These properties can overlap: typically, a dominant modality is modulated by another non-dominant modality.


Multimodal Machine Learning challenges
======================================
The core technical challenges of multimodal machine learning are summarized in *Figure 4*. 

| ![multimodal representation learning challenges](/assets/2023-08-01-multimodal/multimodal-challenges.jpg)| 
|:--:|                 
| *Figure 4*: Representation and Alignment are at the core of every multimodal problem, and they are mandatory to perform Reasoning. Reasoning can give the final answer, or we could be interested in learning Generation or Transference. Finally, we want Quantification to understand and improve the multimodal models. Credits to [P. Liang, et al. (2023)](https://arxiv.org/abs/2209.03430). |

## Representation
The first challenge is the building block for the most multimodal problem: learning a multimodal **representation** that reflects cross-modal interactions between individual elements across different modalities. There are three main approaches (and sub-challenges) for generating a representation of multimodal data, as shown in Fig.5.

| ![multimodal representations](/assets/2023-08-01-multimodal/representations.jpg)| 
|:--:|                 
| *Figure 5*:  |

### Representation by Fusion
Information from multiple modalities is integrated to reduce the number of separate representations. Most research approaches fuse homogeneous modalities (same structure), while fusion with raw and heterogeneous modalities remains an active research topic. One common trick to make modalities homogeneous is to use an unimodal Encoder, as shown in Fig.6. 

| ![Unimodal Encoders](/assets/2023-08-01-multimodal/fusion_naive.jpg)| 
|:--:|                 
| *Figure 6*: Unimodal encoders can be used to obtain homogeneous modalities to ease the fusion process. They can be jointly learned with a fusion network or pretrained. |

 Two early fusion approaches were concatenating the vectors of the X modalities and then making a prediction or predicting X results independently (one for each modality) and then fusing the predictions using majority votes or a model. With representation learning, these approaches have been deprecated in favor of more intelligent fusion techniques.   

 The linear regression model is an example of a fusion mechanism. Let's say we have two modalities, A and B, and two samples $x_A$ and $x_B$:

 $$z=w_0 + w_1x_A + w_2x_B + w_3(x_A\times x_B)+\epsilon$$

The multiplicative term models the interaction between $x_A$ and $x_B$, while $\epsilon$ represents the error residual, which captures everything that it's not possible to model using the additive and (first order) multiplicative terms. Let's give an example:

* $z$: Final exam scores of students in a course.
* $x_A$: Number of hours spent studying.
* $x_B$: Participation in group study sessions.

The model could examine how individual study hours and group study interactions impact exam performance.

The next paragraphs will explain the following fusion techniques:
| ![fusion tehcniques](/assets/2023-08-01-multimodal/fusions.jpg)| 

#### Additive, Multiplicative, and Tensor Fusion

We can define some ways of fusing multivariate samples using the previous idea. 

| ![additive fusion](/assets/2023-08-01-multimodal/additive_fusion.jpg)| 
|:--:|                 
| *Figure 7*: Additive fusion $z=w_1x_A + w_2x_B$|

| ![multiplicative fusion](/assets/2023-08-01-multimodal/multiplication_fusion.jpg)| 
|:--:|                 
| *Figure 8*: Multiplicative fusion $z=w(x_A \times x_B)$|

Additive and multiplicative fusion can be used only when each dimension of the first modality is aligned (has a meaningful equivalence) to the corresponding dimension of the second modality. Let's say $x_A$ was obtained using word2vec, while $x_B$ was obtained using a CNN separately. Since these two models were trained independently, there's no mapping between the first dimension of $x_A$ and the first dimension of $x_B$. 

We can use the bilinear fusion if we don't want to fine-tune the models to produce a coordinated space.
| ![bilinear fusion](/assets/2023-08-01-multimodal/bilinear_fusion.jpg)| 
|:--:|                 
| *Figure 9*: Bilinear fusion $z=w(x_A^T \times x_B)$|

The output is a combination of each dimension, modeling all possible interactions. An extension to that is called Tensor Fusion by [Zadeh et al., 2017](https://arxiv.org/abs/1707.07250), which models both unimodal (additive) and bimodal (multiplicative) interactions.


| ![tensor fusion](/assets/2023-08-01-multimodal/bilinear_fusion_trick.jpg)| 
|:--:|                 
| *Figure 10*: Tensor fusion $z=w([x_A\;1]^T \cdot [x_B\;1])$|

#### Low-rank decomposition

The previous techniques are also extendible to $\mathbb{R}^n$, but it becomes pretty computationally intensive, especially if more than two modalities are involved. Luckily, [Liu et al., 2018](https://arxiv.org/abs/1806.00064) found a way to decompose the computation into an equivalent low-rank representation, drastically increasing the efficiency.  

#### High-Order Polynomial Fusion

The previous decomposition also makes it feasible to increase the order of the interaction polynomial. Until now, we limited ourselves to second-order bimodal terms such as $w(x_A\times x_B)$, which fails to unleash the complete expressive power of multilinear fusion with restricted orders of interactions. [Hou et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f56d8183992b6c54c92c16a8519a6e2b-Abstract.html) show a procedure to create a P-order tensor product for integrating multimodal features by considering high-order moments, shown in Fig.11. In this way, we can introduce interactions of type $w(x_A^2\times x_B^2)$ or $w(x_A^5\times x_B)$ and so on. 

| ![polynomial tensor pooling](/assets/2023-08-01-multimodal/polynomial_tensor_pooling.png)| 
|:--:|                 
| *Figure 11*: The scheme of 5-order polynomial tensor pooling block for fusing $z_1$ and $z_2$. Credits to [Hou et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f56d8183992b6c54c92c16a8519a6e2b-Abstract.html)|


#### Gated Fusion

Gated fusion follows a different approach. *Gating* is a term that means *"do not propagate unwanted signals"* or, in a positive fashion, *"select preferable signals to move forward"*. Hence, we understand that *Gating* and *Attention* want to achieve the same goal. Gated fusion works as follows:
* a gate function $g_A(x_A, x_B)$ produces a vector of weights $v_A$. 
* a gate function $g_B(x_A, x_B)$ produces a vector of weights $v_B$. 
* The input modalities are multiplied by the vectors of weights and then fused (using, for example, additive fusion): 
  $$z=g_A(x_A, x_B)*x_A + g_B(x_A, x_B)*x_B$$

This process is similar to *Attention*, and the output of $g_A$ and $g_B$ can be seen as attention scores. [Arevalo et al., 2017](https://arxiv.org/abs/1702.01992) shows the internal mechanism of the gate unit.

#### Modality-Shifting Fusion

The gate mechanism can be employed in different ways. For example, let's say we have three modalities, but only the first modality is the primary one (containing most of the signal). The other two modalities can be integrated to perform more robust predictions. Citing [Wang et al. (2018)](https://arxiv.org/abs/1811.09362):

> "*Humans convey their intentions through the usage of both verbal and nonverbal behaviors during face-to-face communication. Speaker intentions often vary dynamically depending on different nonverbal contexts, such as vocal patterns and facial expressions. As a result, when modeling human language, it is essential to not only consider the literal meaning of the words but also the nonverbal contexts in which these words appear"* 

In Wang's work, the primary modality is the written language, while video and audio can help the model to disambiguate the semantics of the words. This process is shown in Fig.12: a Gated Modality-mixing Network module integrates acoustic and visual data in a *nonverbal* shift-vector. This integration is done with an Attention Gate as a weighted average over the visual and acoustic embedding based on the original word embedding. Then, the multimodal-shifted word representation is generated by integrating the nonverbal shift vector into the initial word embedding. 

| ![recurrent attended variation embedding network wang 2018](/assets/2023-08-01-multimodal/wang_2018.jpg)| 
|:--:|                 
| *Figure 12*: The Recurrent Attended Variation Embedding Network (RAVEN). The modality shifting can be seen as a modulation interaction: the secondary modalities modulate the primary one. Credits to [Wang et al. (2018)](https://arxiv.org/abs/1811.09362)|

The same approach has been used by [Rahman et al., 2020](https://arxiv.org/abs/1908.05787) to create a Multimodal Adaptation Gate (MAG) to allow BERT and XLNet to accept multimodal nonverbal data during fine-tuning. The MAG generates a shift to the internal representation of BERT and XLNet, conditioned on the visual and acoustic modalities. 

#### Dynamic Fusion
Even if the previous technique works well, they are applicable in specific domains. [Xu et al. (2021)](https://arxiv.org/abs/2102.02340) proposed a fusion technique applicable when the multimodal data is very heterogeneous, such as electronic health records data, that *"contains a mixture of structured (codes) and unstructured (free-text) data with sparse and irregular longitudinal features – all of which doctors utilize when making decisions"*. They do not propose a particular architecture but a neural architecture search method called *Multimodal Fusion Architecture Search* to *"simultaneously search across multimodal fusion strategies and modality-specific architectures"*.

#### Nonlinear Fusion
Nonlinear fusion adopts a nonlinear model to perform the fusion after the concatenation of the modalities. This can be seen as an early fusion approach (Fig.13) with the following:

$y=f(x_A,x_B)\in \mathbb{R}^d$ 

where $f$ could be a multi-layer perceptron or any nonlinear model.

| ![early and late fusion](/assets/2023-08-01-multimodal/early_vs_late_fusion.jpg)| 
|:--:|                 
| *Figure 13*: Early fusion concatenates original or extracted features at the input level. Late fusion aggregates predictions at the decision level. Credits to: [Huang et al. (2020)](https://www.nature.com/articles/s41746-020-00341-z)|

This kind of fusion is powerful but needs to be more interpretable. How can we be sure that the model learns cross-modal interactions? Hessel and Lee (2020)(https://arxiv.org/abs/2010.06572) showed that *"sometimes high-performing black-box algorithms turn out to be mostly exploiting unimodal signals in the data"*, i.e., the black-box model is equivalent to two unimodal encoders and an additive fusion:

$y=f_A(x_A) + f_B(x_B)$ 

The authors created a diagnostic tool called *Empirical Multimodally-Additive function Projection* (EMAP) to isolate whether cross-modal interactions improve performance for a given model on a given task. They found that in many cases for seven image+text classification tasks, removing cross-modal interactions results in little to no performance degradation. 

Another discriminating factor is whether the model is trained to solve a downstream task. In the latter case, the learned representation will be optimized to solve the task the model is trained for. Hence, it will not contain the complete multimodal information but only the one that matters. On the other hand, [Ngiam1 et al. (2011)](https://dl.acm.org/doi/10.5555/3104482.3104569) proposed one of the first deep models to learn a shared representation between modalities to "*demonstrate cross modality feature learning, where better features for one modality (e.g., video) can be learned if multiple modalities (e.g., audio and video) are present at feature learning time*". They a bimodal deep autoencoder shown in Fig.14 on audio and video data in a denoising
fashion, using an augmented dataset with examples that require the network to reconstruct both modalities gave only one. 

| ![bimodal deep autoencoder](/assets/2023-08-01-multimodal/ngiam_2011.jpg)| 
|:--:|                 
| *Figure 14*:  Credits to: [Ngiam et al. (2011)](https://dl.acm.org/doi/10.5555/3104482.3104569)|

In addition, they demonstrated the zero-shot cross-modal adaptation of this model. They trained a linear classifier on audio samples only (transformed via the previously learned shared representation) and then tested it with video samples, as shown in Fig.15. From the paper:

> *"In essence, we are telling the supervised learner how the digits "1", "2", etc. sound, while asking it to distinguish them based on how they are visually spoken –
hearing to see".*

| ![bimodal deep autoencoder](/assets/2023-08-01-multimodal/ngiam_2.jpg)| 
|:--:|                 
| *Figure 15*:  Credits to: [Ngiam et al. (2011)](https://dl.acm.org/doi/10.5555/3104482.3104569)|


Finally, one last topic that is currently an open research question. We discussed early fusion, in which different modalities are encoded in a homogeneous vector and concatenated (Fig.6, Fig.13): fusion happens after significant independent processing. [Barnum et al. 2022](https://arxiv.org/abs/2011.07191) claim that "*the brain performs multimodal processing almost immediately [..] neuroscience suggests that a detailed study of early multimodal fusion could improve artificial multimodal representations. [..] primary sensory cortices may not be unimodal (Budinger et al., 2006). This may partly be because of individual neurons' abilities to be modulated by multiple modalities (Meredith & Allman, 2009). In a striking discovery, Allman and Meredith (2007) found that 16% of visual neurons in the posterolateral lateral suprasylvian that were previously believed to be only visually responsive were significantly facilitated by auditory stimuli."*.  

Very Early Fusion is an active research field, [Barnum et al. 2022](https://arxiv.org/abs/2011.07191) proposed a convolutional-LSTM model applied directly to raw visual and acoustic modalities using local patches to search for correspondences, demonstrating that *"immediate fusion of audio and visual inputs in the initial C-LSTM layer results in higher performing networks that are more robust to the addition of white noise in both audio and visual inputs"*.


### Representation by Coordination
Fusion is not always the right approach to force different modalities into the same representation space because there is only sometimes a one-to-one mapping between two modalities. Coordination aims to keep the representation spaces separated but allows linking them with different degrees (of coordination). This technique works by contextualizing the representation spaces in order to incorporate information from multiple modalities. 

There are two leading families of approaches: Coordination functions, Gated functions, and Contrastive functions.

#### Coordination function

Two unimodal encoders $f$ and $h$ create an homogeneous representations ($Z_A$ and $Z_B$). The model must learn a coordination function $g(f(Z_A), h(Z_B))$ that contextualizes a representation space by telling, for each sample, how similar it is to the samples of the other representation space.

In order to learn that function, we need a **dataset of positive pairs** (that are supposed to mean the same) and a **similarity function**. There are several techniques to compute similarity: cosine similarity (equal to Pearson's correlation coefficient if the samples are normalized), kernel similarity (the same used by the SVM model), and canonical correlation analysis (CCA). The latter learns $U$ and $V$ projection over $Z_A$ and $Z_B$ to make them as correlated as possible:

$$\argmax_{V,U,f_A,f_B} corr(Z_A, Z_B)$$ 

Data can be correlated in different ways: orthogonal to each other, and canonical correlation forces the embeddings to be correlated in multiple ways. [Wang et al. 2016](https://arxiv.org/pdf/1602.01024.pdf) achieved a correlation-based representation learning model using deep canonically correlated autoencoders, shown in Fig.16. 

| ![dcca](/assets/2023-08-01-multimodal/dcca.jpg)| 
|:--:|                 
| *Figure 16*: Credits to: [CMU Multimodal Machine Learning course, Fall 2022](https://www.youtube.com/watch?v=65xxHVyHKi0&list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA&index=9&t=905s)|


#### Gated function
Gates are used in the same way as gated fusion, but this time, two different embeddings are generated:

$$Z_A = g_A(X_A,X_B)X_A + g_B(X_A,X_B)X_B$$
$$Z_B = g_B(X_A,X_B)X_B + g_A(X_A,X_B)X_A$$

| ![gated coordination](/assets/2023-08-01-multimodal/gated_coordination.jpg)| 
|:--:|                 
| *Figure 17*: Gated coordination. Credits to: [CMU Multimodal Machine Learning course, Fall 2022](https://www.youtube.com/watch?v=65xxHVyHKi0&list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA&index=9&t=905s)|

#### Contrastive function

Contrastive learning is one of the most popular techniques for multimodal coordination. It can be used in self-supervised ([Chen et al. 2020](https://arxiv.org/abs/2002.05709), [Jaiswal et al. 2021](https://arxiv.org/abs/2011.00362)) and supervised settings ([Khosla et al. 2020](https://arxiv.org/abs/2004.11362)). 
Citing the *[Extending Contrastive Learning to the Supervised Setting
](https://blog.research.google/2021/06/extending-contrastive-learning-to.html)* blog post by Google:

> "These contrastive learning approaches typically teach a model to pull together the representations of a target image (a.k.a., the “anchor”) and a matching (“positive”) image in embedding space, while also pushing apart the anchor from many non-matching (“negative”) images."*

[Kiros et al. 2014](https://arxiv.org/abs/1411.2539) brought this idea to the multimodal field, proposing a symmetric contrastive loss:

$$\max\{0, \alpha + sim(\textbf{Z}_L, \textbf{Z}^+_V) - sim(\textbf{Z}_L, \textbf{Z}^-_V)\} + \max\{0, \alpha + sim(\textbf{Z}_V, \textbf{Z}^+_L) - sim(\textbf{Z}_V, \textbf{Z}^-_L)\}$$

In the above formula, the similarity function is applied to positive samples of language and visual samples $sim(\textbf{Z}_L, \textbf{Z}^+_V)$ and to negative samples of language and visual samples $sim(\textbf{Z}_L, \textbf{Z}^-_V)$, and the other way around.   

| ![coordination contrastive loss](/assets/2023-08-01-multimodal/coordination_contrastive_loss.jpg)| 
|:--:|                 
| *Figure 18*: Coordination with a contrastive loss.|

The two separated autoencoders are trained and coordinated together through the contrastive loss. A fascinating property emerged from the training: the ability to do arithmetics between these two different but coordinated embeddings. At test time, the researchers encoded an image of a red car and the words *"red"* and *"blue"*. 

| ![coordination contrastive loss](/assets/2023-08-01-multimodal/contrastive_loss_showcase.jpg)| 
|:--:|                 
| *Figure 19*: Multimodal vector space arithmetic. Credits: [Kiros et al. 2014](https://arxiv.org/abs/1411.2539)|


Seven years after, the *Contrastive Language-Image Pre-training (CLIP)* by [Radford et al. 2021](https://arxiv.org/abs/2103.00020) (OpenAI), has been released. It shares the same principles as [Kiros et al. 2014](https://arxiv.org/abs/1411.2539) with a slightly different loss. CLIP was a breakthrough proving that the large-scale weak-supervision contrastive training paradigm can also be applied for vision and not only text (such as GPT). The pretrain dataset comprises 400 million (image, text) pairs, and the learning objective is to predict the captions that go with a particular image. During pretraining, CLIP learns to associate images and text by contrasting positive pairs (an image and its corresponding text) against negative pairs (an image with randomly sampled text and vice versa). The model is trained to make the positive pair's similarity higher (matrix diagonal in Fig.20) than the negative pair's similarity.

| ![clip](/assets/2023-08-01-multimodal/clip.jpg)| 
|:--:|                 
| *Figure 20*: CLIP usage workflow. Credits to: [OpenAI blog](https://openai.com/research/clip).|

This method gave rise to a powerful zero-shot multimodal model. From [OpenAI's blog post on CLIP](https://openai.com/research/clip):

> *"Because they learn a wide range of visual concepts directly from natural language, CLIP models are significantly more flexible and general than existing ImageNet models. We find they are able to zero-shot perform many different tasks. To validate this we have measured CLIP’s zero-shot performance on over 30 different datasets including tasks such as fine-grained object classification, geo-localization, action recognition in videos, and OCR."*

To discover more about CLIP, I suggest two blog posts: [A. Chadha, V. Jain Paper Reviews, 2022](https://www.vinija.ai) and [OpenAI's blog post on CLIP](https://openai.com/research/clip). 

Contrastive representation learning is a vast field of study. I will recommend the [Lil'Log blog post]((https://lilianweng.github.io/posts/2021-05-31-contrastive)) which gives a lovely overview of the contrastive losses and critical principles.
 

### Representation by Fission
In this case, the number of representations is greater than the number of modalities. Modalities can interact with each other in more than one way. Let's think about language and vision. A word can have different types of interactions: it can directly correspond to the name of an object, or it can have a different type of relation with that object (e.g., parent, use by, etc.). Fission is a way to model all these types of relationships with different representations. In this sense, we could interchange the name *fission* with *factorization*. As we can see from Fig.21, fission representation aims to learn factorized representations, i.e., representations that capture the information specific to particular modalities and representations that capture the information that is in common between modalities. 

| ![types of fissions](/assets/2023-08-01-multimodal/fission.jpg)| 
|:--:|                 
| *Figure 21*: Sub-challenge of Fission. Credits to: [CMU Multimodal Machine Learning course, Fall 2022](https://www.youtube.com/watch?v=65xxHVyHKi0&list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA&index=9&t=905s) |

#### Modality-level fission

A naive way to build a system capable of learning modality-level factorized representations is to train three different encoders:
* The first gets the first modality A as input
* The second gets the second modality B as input
* The third gets both the first and second modalities as input
We could use their output to make predictions. The problem with this solution is that there is no guarantee that the first and second encoders will learn modality-specific features. 

[Tsai et al. 2019](https://arxiv.org/pdf/1806.06176.pdf) proposed a joint generative-discriminative objective to factorize representations into two sets of independent factors: multimodal discriminative and modality-specific generative factors. This architecture is shown in Fig.22. The paper explains that:
> *Multimodal discriminative factors are shared across all modalities and contain joint multimodal features required for discriminative tasks such as sentiment prediction. Modality-specific generative factors are unique for each modality and contain the information required for generating data.*

The objectives that generate factorization are three losses:
- a discriminative loss for the multimodal autoencoder  
- a generative loss for both unimodal autoencoder
- a "no-overlap" loss enforced between $Q_{Z_{y}}$ and $Q_{Z_{a1}}$, $Q_{Z_{y}}$ and $Q_{Z_{a2}}$ and $Q_{Z_{y}}$ and $Q_{Z_{a3}}$.

| ![Learning Factorized Multimodal Representations](/assets/2023-08-01-multimodal/mfm.jpg)| 
|:--:|                 
| *Figure 22*: The Multimodal Factorization Model with three modalities.   Credits to: [Tsai et al. 2019](https://arxiv.org/pdf/1806.06176.pdf).|

---

Another approach to achieve modality-level fission is based on *Information Theory* by [Shannon, 1948](https://ieeexplore.ieee.org/document/6773024) that defines the *Information Content I* of a sample $x$ as:

$$I(x)\sim\frac{1}{p(x)}$$

The less probable $x$ is, the less information it carries. The quantity above does not have a superior limit, so let's take the $log$:

$$I(x)=-\log(p(x))$$ 

Given the definition above, we can compute the (discrete) information content of a modality called *Entropy* as:

$$H(X)=\mathbb{E}[I(X)]=\mathbb{E}[-\log(p(X))]=-\sum_{n=1}^{\infty}p(X)\log(p(X)) = 1$$ 

In the above formula, $X$ is a discrete random variable.  

If two modalities are independent, $H$ will describe their entropy. However, in real-world scenarios, modalities are interconnected. If we want to factorize which features overlap and which are modality-specific, we must introduce two other definitions: *conditional entropy* and *mutual information*.

Conditional entropy is defined as:

$$H(A|B)=-\mathbb{E}_{A,B}[\log\frac{p(a,b)}{p(a)}]$$

Mutual information is defined as:

$$I(A;B)=H(A)-H(A|B)=\mathbb{E}_{A,B}[\log\frac{P_{AB}(a,b)}{P_A(a)P_B(B)}]$$

| ![information gain](/assets/2023-08-01-multimodal/ig.jpg)| 
|:--:|                 
| *Figure 23*: Conditional Entropy and Mutual Information. Credits to: [CMU Multimodal Machine Learning course, Fall 2022](https://www.youtube.com/watch?v=65xxHVyHKi0&list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA&index=9&t=905s). |


[Colombo et al. 2021](https://arxiv.org/abs/2109.00922) developed a Fusion approach by defining a loss based on MI: two input modalities are fused by minimizing the loss relative to the downstream task and by minimizing the loss associated with their mutual information score. However, this approach will create a representation containing only the commonalities between the modality and not modality-specific features, and it is acceptable under the assumption that the information present in both modalities is the most important for the downstream task.

[Tsai et al. 2020](https://arxiv.org/abs/2006.05576) showed a link between Information Theory and Self-Supervised learning with data augmentation: different views of the same data are generated using shifts, adding noise, etc. We can think of them as different modalities grounded in the source space: high mutual information low conditional entropy.

#### Fine-grained fission

The right panel of Fig.21 shows a fine-grained clustering of different interactions. This sub-field is still an active research area. [Hu et al. 2019](https://arxiv.org/pdf/1807.03094.pdf) proposed a contribution in this context with their *Deep Multimodal Clustering* algorithm, which aim is to find the multimodal correspondences between visual features from images and audio features from audio clips (associated to the images), in an unsupervised way. Two unimodal networks (CNNs) are used to extract the feature maps. A clustering network puts together the visual feature maps with the corresponding audio feature maps. They trained the multimodal clustering network end-to-end using pair visual-audio data with a max-margin loss. 

| ![deep multimodal clustering](/assets/2023-08-01-multimodal/deep_multimodal_clustering.jpg)| 
|:--:|                 
| *Figure 24*:  Credits to: [Hu et al. 2019](https://arxiv.org/pdf/1807.03094.pdf)|

The Co-Clustering Module (shown in Fig.24) tries to find the relations and correspondences between different objects and different sounds. As the paper explained:
> *"The aforementioned clusters indicate a kind of soft assignment (segmentation) over the input image or spectrogram, where each cluster mostly corresponds to certain content (e.g., baby face and drum in image, voice and drumbeat in sound), hence they can be viewed as the distributed representations of each modality."*
