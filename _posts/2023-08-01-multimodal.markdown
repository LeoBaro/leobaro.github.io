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
This blog post gives an introduction of the field of Multimodal Machine Learning. In practice these are my notes from the course held by the Carnegie Mellon University: [11-777 Multimodal Machine Learning, 2022 Fall](https://cmu-multicomp-lab.github.io/mmml-course/fall2022/) and the review paper of [P. Liang, et al. (2023)](https://arxiv.org/abs/2209.03430).


Towards multimodality
=====================
Let's start by giving a definition of *modality*:
> :bulb: **A modality is a way something is expressed or perceived**

A multimodal dataset contains information spread over multiple modalities. Let's think about a recording of a public speech for example. Utterances (audio modality) carry most of the information but also the body language of the speaker or facial expressions (video modality) complement and enrich the audio information to convey the final message. Can machine learning models extract knowledge from multiple modalities?

As shown by *Figure 1*, multimodal machine learning has gained particular attention in the scientific community over the last 10 years. 

| <img src="/assets/2023-08-01-multimodal/trend.jpg" alt="multimodal trend" width="800"/> | 
|:--:|                 
| *Figure 1*: the number of publications in the multimodal machine learning field over recent yers. Credits: [Dimensions](https://app.dimensions.ai).|

From 2010, researchers start to use deep learning to exploit multimodal data. In the beginning with Deep Boltzmann Machines ([N. Srivastava, R. Salakhutdinov, 2012](https://papers.nips.cc/paper_files/paper/2012/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)) and then, [K. Xu et al. 2016](https://arxiv.org/abs/1502.03044) exploited the [attention mechanism](https://leobaro.github.io/deep-learning/2023/06/29/transformers.html) to bring multimodal research under the umbrella of computer vision and natural language processing for image captioning. After that moment, research in the multimodal modal domain exploded thanks to several key factors: the availability of new large-scale multimodal datasets and processing hardware but also, advances in computer vision and language processing models enabled the representation of etherogenous data (images, text),into homogeneous embedded vectors with meaningful features. As research on transformers, models optimization and  distributed training continued as well as the availability of new multimodal datasets increased, multimodal deep learning spreaded over multiple fields. Let's just think about robotics, which aim is to develop intelligent autonomous agents capable of integrating and learning from different modalities. But also other real world tasks such as multimodal sentiment and emotion recognition, multimodal QA, multimodal dialog, event recognition, multimedia information retrieval and so on.

The properties of multimodal data
=====================================

We're used to unimodal source of data, which spectrum goes from raw (close to the sensor that captured it) to more abstract representations. For example, a microphone records speech (sound waves) that can be translated to written language (tokens), and analyzed to extract high orders descriptions such as sentiment. This is true also for other modalities such as images, and nothing stop us to integrate all these information togheter, regardless of the abstraction level. Integrating different modalities that are closer to the raw part of the spectrum is more challenging because they tend to be more heteregeneous, with respect to higher level ones. **Heterogeneity** is a key concept in multimodal learning, and it's manifested along the following dimensions:
* **Representation**: it refers on the representation used in the samples space. Examples: set of characters (text) vs matrixes (images) vs nodes and edges (graphs). 
* **Distribution**: different modalities have different distributions and frequencies of samples (the number of objects per image has in general a lower frequency than the number of words in a sentence).
* **Structure**: images have spatial structure while sentences have sequential structure. In addition, the latent space has its own structure that can be different across modalities. 
* **Information**: the total information that can be extracted from a modality. It depends on the abstraction level: an image data can be exploited from raw pixels to object categories.
* **Noise**: occlusion in images, typos in NLP, missing data, and so on.
* **Relevance**: to a specific task that we want to address. 

So, altought different modalities can be very different and they can bring its own unique information, they can be *connected* by sharing complementary information. **Connection** is another key concept in multimodal learning. Which is the form of these connections? We can see it from a bottom-up (statistical) or top-down views (semantic). Statistically speaking, elements that co-occur (cor)relate to each other and this is called *association*. On the semantic side, a *corrispondence* is the presence of the same element (with the same semantic) in both element of different modalities. A stronger connection type is the statistical *dependency*, which identifies casual relations. The equivalent in the semantic world is called *relationship*. TODO: examples. This two properties can help to define multimodal as:

> :bulb: **multimodal: the science of heterogeneous and interconnected data**

In particular, the goal of multimodal learning is to exploit the complementary information coming from multiple connected modalities, through their **Interaction**, the third key concept. It comes into play when the model integrate multiple modalities to perform inference. The result of the interaction could be **new information** (*Response* in *Figure 2*) that can help the model to make better predictions.  

| ![multimodal trend](/assets/2023-08-01-multimodal/inference.jpg)| 
|:--:|                 
| *Figure 2*: |

Regarding on how the response change we can distinguish several types of interactions. We talk about **redundancy** if the modalities give similar answers. In this case the multimodal response can be **enhanced** (higher confidence) or it can be **equivalent** to the unimodal one (no interaction). What if the two modalities give different responses? For example, let's ask the question:

> **is this a dangerous animal?**

🤔🤔🤔

| ![shark fin](/assets/2023-08-01-multimodal/interactions.jpg)| 
|:--:|                 
| *Figure 3*: non-redundant interactions.|

*Figure 3* shows a disagreement between the modalities. We have **dominance** if one modality response takes over the other (e.g. the multimodal response is "*yes*".). We have **independence** if the responses do not interact. **Modulation** is when one modality response enhance or reduce the other one. Finally, the best scenario is when the **emergence** property give birth to new information. These property can overlap: typically, a dominant modality is also modulated by another non-dominant modality.



Multimodal Machine Learning challenges
======================================

The core technical challenges of multimodal machine learning are summarized in *Figure 4*. 

| ![multimodal representation learning challenges](/assets/2023-08-01-multimodal/multimodal-challenges.jpg)| 
|:--:|                 
| *Figure 4*: Representation and Alignment are at the core of every multimodal problem and they are mandatory to perform Reasoning. Reasoning can give the final answer or maybe we could be interested in learning Generation or Transference. Finally, we want Quantification to understand and improve the multimodal models. Credits to [P. Liang, et al. (2023)](https://arxiv.org/abs/2209.03430). |

## Representation
The first challenge is the building block for most multimodul problem: learning a multimodal **representation** that reflect cross-modal interactions between individual elements across different modalities. There are three main approaches (and sub-challenges) for generating a representation for multimodal data, as shown in Fig.5.

| ![multimodal representations](/assets/2023-08-01-multimodal/representations.jpg)| 
|:--:|                 
| *Figure 5*:  |

### Representation by Fusion
Information from multiple modalities is integrated to reduce the number of separate representations. Most research approches fuse homogeneous modalities (same structure) while fusion with raw and etherogeneous modalities is still an active reaserch topic. One common trick to make modalities homogeneous is to use an unimodal Encoder, as shown in Fig.6. 

| ![Unimodal Encoders](/assets/2023-08-01-multimodal/fusion_naive.jpg)| 
|:--:|                 
| *Figure 6*: unimodal Encoders can be used to obtain homogeneous modalities to ease the fusion process. They can be jointly learned with fusion network or pretrained. |

 Two early approach of fusion were to concatenate the vectors of the X modalities and then make a prediction or predict X results independetly (one for each modality) and then fusion the predictions using majority votes or a model. With representation learning these approaches have been deprecated in favor of more smart fusion techniques.   

 The linear regression model is an example of fusion mechanism. Let's say we have two modalities A and B, and two samples $x_A$ and $x_B$:

 $$z=w_0 + w_1x_A + w_2x_B + w_3(x_A\times x_B)+\epsilon$$

The multiplicative term models the interaction between $x_A$ and $x_B$, while $\epsilon$ represents the error residual, that captures everything that it's not possible to model using the additive and (first order) multiplicative terms. Let's give an example:

* $z$: Final exam scores of students in a course.
* $x_A$: Number of hours spent studying.
* $x_B$: Participation in group study sessions.

The model could examine how individual study hours and group study interactions impact exam performance.

The next paragraphs will explain the following fusion techniques:
| ![fusion tehcniques](/assets/2023-08-01-multimodal/fusions.jpg)| 

#### Additive, Multiplicative and Tensor Fusion

Taking the previous idea we can define some ways of fusing multivariate samples. 

| ![additive fusion](/assets/2023-08-01-multimodal/additive_fusion.jpg)| 
|:--:|                 
| *Figure 7*: Additive fusion $z=w_1x_A + w_2x_B$|

| ![multiplicative fusion](/assets/2023-08-01-multimodal/multiplication_fusion.jpg)| 
|:--:|                 
| *Figure 8*: Multiplicative fusion $z=w(x_A \times x_B)$|

Additive and multiplicative fusion can be used only when each dimension of the first modality is aligned (have a meaningful equivalence) to the corresponding dimension of the second modality. Let's say $x_A$ was obtained using word2vec, while $x_B$ was obtained using a CNN, seperately. Since these two models were trained indipendently there's no mapping between the first dimension of $x_A$ and the first dimension of $x_B$. 

If we don't want to fine-tune the models to produce a coordinated space, a workaround is to use the bilinear fusion.
| ![bilinear fusion](/assets/2023-08-01-multimodal/bilinear_fusion.jpg)| 
|:--:|                 
| *Figure 9*: Bilinear fusion $z=w(x_A^T \times x_B)$|

The output is a combination of each dimension, modeling all possible interactions. An extension to that is called Tensor Fusion by [Zadeh et al., 2017](https://arxiv.org/abs/1707.07250) that models both unimodal (additive) and bimodal (multiplicative) interactions.


| ![tensor fusion](/assets/2023-08-01-multimodal/bilinear_fusion_trick.jpg)| 
|:--:|                 
| *Figure 10*: Tensor fusion $z=w([x_A\;1]^T \cdot [x_B\;1])$|

#### Low-rank decomposition

The previous techiniques are also extendible to $\mathbb{R}^n$ but it becomes quite computational intensive especially if more than two modalities are involved. Luckily, [Liu et al., 2018](https://arxiv.org/abs/1806.00064) found a way to decompose the computation into an equivalent low-rank representation, drastically increasing the efficiency.  

#### High-Order Polynomial Fusion

The previous decomposition makes also feasible to increase the order of the interaction polynomial. Until now we limited ourselves to second-order bimodal terms such as $w(x_A\times x_B)$, which fails to unleash the complete expressive power of multilinear fusion with restricted orders of interactions. [Hou et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f56d8183992b6c54c92c16a8519a6e2b-Abstract.html) show a procedure to create a P-order tensor product for integrating multimodal features by considering high-order moments, shown in Fig.11. In this way, we can introduce interactions of type $w(x_A^2\times x_B^2)$ or $w(x_A^5\times x_B)$ and so on. 

| ![polynomial tensor pooling](/assets/2023-08-01-multimodal/polynomial_tensor_pooling.png)| 
|:--:|                 
| *Figure 11*: The scheme of 5-order polynomial tensor pooling block for fusing $z_1$ and $z_2$. Credits to [Hou et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f56d8183992b6c54c92c16a8519a6e2b-Abstract.html)|


#### Gated Fusion

Gated fusion follows a different approach. *Gating* is a term that means *"do not propagate unwanted signals"* or, in a positive fashion, *"select preferable signals to move forward"*. Hence, we understand that *Gating* and *Attention* want to achieve the same goal. Gated fusion works as follows:
* a gate function $g_A(x_A, x_B)$ produces a vector of weights $v_A$. 
* a gate function $g_B(x_A, x_B)$ produces a vector of weights $v_B$. 
* the input modalities are multiplied by the vectors of weights and then fused together (using for example additive fusion): 
  $$z=g_A(x_A, x_B)*x_A + g_B(x_A, x_B)*x_B$$

This is really similar to *Attention*, and the output of $g_A$ and $g_B$ can be seen as attention scores. [Arevalo et al., 2017](https://arxiv.org/abs/1702.01992) shows the internal mechanism of the gate unit.

#### Modality-Shifting Fusion

The gate mechanism can be employed in different ways. For example, let's say we have three modalities but only the first modality is the primary one (containing most of the signal). The other two modalities can be integrated to perform more robust predictions. Citing [Wang et al. (2018)](https://arxiv.org/abs/1811.09362):

> "*Humans convey their intentions through the usage of both verbal and nonverbal behaviors during face-to-face communication. Speaker intentions often vary dynamically depending on different nonverbal contexts, such as vocal patterns and facial expressions. As a result, when modeling human language, it is essential to not only consider the literal meaning of the words but also the nonverbal contexts in which these words appear"* 

In Wang's work, the primary modality is the written language while video and audio can help the model to disambiguate the semantics of the words. This process is shown in Fig.12: a Gated Modality-mixing Network module integrate acustic and visual data in a *nonverbal* shift vector. This integration is done with an Attention Gate as a weighted average over the visual and acoustic embedding based on the original word embedding. Then, the multimodal-shifted word representation is generated by integrating the nonverbal shift vector to the original word embedding. 

| ![recurrent attended variation embedding network wang 2018](/assets/2023-08-01-multimodal/wang_2018.jpg)| 
|:--:|                 
| *Figure 12*: the Recurrent Attended Variation Embedding Network (RAVEN). The modality shifting can be seen as a modulation interaction: the secondary modalities modulates the primary one. Credits to [Wang et al. (2018)](https://arxiv.org/abs/1811.09362)|

The same approach has been used by [Rahman et al., 2020](https://arxiv.org/abs/1908.05787) to create a Multimodal Adaptation Gate (MAG) to allow BERT and XLNet to accept multimodal nonververbal data during fine-tuning. The MAG generates a shift to internal representation of BERT and XLNet, conditioned on the visual and acoustic modalities. 

#### Dynamic Fusion
Even if the previous technique works well, they are applicable in specific domains. [Xu et al. (2021)](https://arxiv.org/abs/2102.02340) proposed a fusion technique to applicable when the multimodal data is very heterogeneous, such as electronic health records data, that *"contains a mixture of structured (codes) and unstructured (free-text) data with sparse and irregular longitudinal features – all of which doctors utilize when making decisions"*. Actually, they don't propose a particular architecture but a neural architecture search method called *MUltimodal Fusion Architecture SeArch* (MUFASA) to *"simultaneously search across multimodal fusion strategies and modality-specific architectures"*.

#### Nonlinear Fusion
Nonlinear fusion adopts a nonlinear model to perform the fusion after the concatenation of the modalities. This can be seen as an early fusion approach (Fig.13) with:

$y=f(x_A,x_B)\in \mathbb{R}^d$ 

where $f$ could be a multi-layer perceptron or any nonlinear model.

| ![early and late fusion](/assets/2023-08-01-multimodal/early_vs_late_fusion.jpg)| 
|:--:|                 
| *Figure 13*: Early fusion concatenates original or extracted features at the input level. Late fusion aggregates predictions at the decision level. Credits to: [Huang et al. (2020)](https://www.nature.com/articles/s41746-020-00341-z)|

This kind of fusion is powerful but lacks interpretability. How can we be sure that cross-modal interactions are learnt by the model? Hessel and Lee (2020)(https://arxiv.org/abs/2010.06572) showed that *"sometimes high-performing black-box algorithms turn out to be mostly exploiting unimodal signals in the data"* i.e. the black-box model is equivalent to two unimodal encoders and an additive fusion:

$y=f_A(x_A) + f_B(x_B)$ 

The authors created a diagnostic tool called *Empirical Multimodally-Additive function Projection* (EMAP), for isolating whether or not cross-modal interactions improve performance for a given model on a given task. They found that, in many cases for seven image+text classification tasks, removing cross-modal interactions results in little to no performance degradation. 

Another discriminating factor is the following: is the model trained to solve a particular downstream task? In the latter case, the learned representation will be optimized to solve the task the model is trained for. Hence, it will not contain the complete multimodal information, but only the one that matters. On the other hand, [Ngiam1 et al. (2011)](https://dl.acm.org/doi/10.5555/3104482.3104569) proposed one of the first deep models to learn a shared representation between modalities to "*demonstrate cross modality feature learning, where better features for one modality (e.g., video) can be learned if multiple modalities (e.g., audio and video) are present at feature learning time*". They a bimodal deep autoencoder shown in Fig.14 on audio and video data in a denoising
fashion, using an augmented dataset with examples that require the network to reconstruct both
modalities given only one. 

| ![bimodal deep autoencoder](/assets/2023-08-01-multimodal/ngiam_2011.jpg)| 
|:--:|                 
| *Figure 14*:  Credits to: [Ngiam et al. (2011)](https://dl.acm.org/doi/10.5555/3104482.3104569)|

In addition they demonstrated the zero-shot cross-modal adaptation of this model. They trained a linear classifier on audio samples only (transformed via the previously-learned shared representation) and then tested it with video samples, as shown in Fig.15. From the paper:

> *"In essence, we are telling the supervised learner how the digits "1", "2", etc. sound, while asking it to distinguish them based on how they are visually spoken –
hearing to see".*

| ![bimodal deep autoencoder](/assets/2023-08-01-multimodal/ngiam_2.jpg)| 
|:--:|                 
| *Figure 15*:  Credits to: [Ngiam et al. (2011)](https://dl.acm.org/doi/10.5555/3104482.3104569)|


Finally, one last topic that is currently an open research question. We talked about early fusion in which different modalities are encoded in a homogeneous vector and then concatenated (Fig.6,Fig.13): fusion happens after significant independent processing. [Barnum et al. 2022](https://arxiv.org/abs/2011.07191) claims that "*the brain performs multimodal processing almost immediately [..] neuroscience suggests that a detailed study of early multimodal fusion could improve artificial multimodal representations. [..] primary sensory cortices may not be unimodal at all (Budinger et al., 2006). This may
in part be because of individual neuron’s abilities to be modulated by multiple modalities (Meredith & Allman, 2009). In a striking discovery, Allman & Meredith (2007) found that 16% of visual
neurons in the posterolateral lateral suprasylvian that were previously believed to be only visually
responsive were significantly facilitated by auditory stimuli."*.  

Very Early Fusion is an active research field, [Barnum et al. 2022](https://arxiv.org/abs/2011.07191) proposed a convolutional-LSTM model applied directly to raw visual and acustic modalities using local patches to search for correspondences, demonstrating that *"immediate fusion of audio and visual inputs in the initial C-LSTM layer results in higher performing networks that are more robust to the addition of white noise in both audio and visual inputs".*




### Representation by Coordination
Fusion is not always the right approach to force different modalities into the same representation space because there's not always a one-to-one mapping between two modalities. Coordination aims to keep the representation spaces separated but allows to link them with different degrees (of coordination). This technique works by contextualizing the representation spaces in order to incorporate information from multiple modalities. 

There're two main families of approaches: Coordination functions, Gated functions and Contrastive functions.

#### Coordination function

Two unimodal encoders $f$ and $h$ create an homogeneous representations ($Z_A$ and $Z_B$). The model must learn a coordination function $g(f(Z_A), h(Z_B))$ that contextualize a representation space by telling, for each sample, how similar it is to the samples of the other representation space.

In order to learn that function, we need a **dataset of positive pairs** (that are supposed to mean the same) and a **similarity function**. There're several techniques to compute similarity: cosine similarity (equal to the Pearson's correlation coefficient if the samples are normalized), kernel similarity (the same used by the SVM model) and canonical correlation analysis (CCA). The latter learns $U$ and $V$ projection over $Z_A$ and $Z_B$ to make them as correlated as possible:

$$\argmax_{V,U,f_A,f_B} corr(Z_A, Z_B)$$ 

There's different ways data can be correlated, orthogonal to each other and canonical correlation forces the embeddings to be correlated in multiple ways. [Wang et al. 2016](https://arxiv.org/pdf/1602.01024.pdf) managed to achieve a correlation-based representation learning model using deep canonically correlated autoencoders, shown in Fig.16. 

| ![dcca](/assets/2023-08-01-multimodal/dcca.jpg)| 
|:--:|                 
| *Figure 16*: Credits to: [CMU Multimodal Machine Learning course, Fall 2022](https://www.youtube.com/watch?v=65xxHVyHKi0&list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA&index=9&t=905s)|


#### Gated function
Gates are used in the same way as gated fusion but this time two different embeddings are generated:

$$Z_A = g_A(X_A,X_B)X_A + g_B(X_A,X_B)X_B$$
$$Z_B = g_B(X_A,X_B)X_B + g_A(X_A,X_B)X_A$$

| ![gated coordination](/assets/2023-08-01-multimodal/gated_coordination.jpg)| 
|:--:|                 
| *Figure 17*: Gated coordination. Credits to: [CMU Multimodal Machine Learning course, Fall 2022](https://www.youtube.com/watch?v=65xxHVyHKi0&list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA&index=9&t=905s)|




#### Contrastive function

Contrastive learning is one of the most popular techniques for multimodal coordination. It can be used in self-supervised ([Chen et al. 2020](https://arxiv.org/abs/2002.05709), [Jaiswal et al. 2021](https://arxiv.org/abs/2011.00362)) and supervised settings ([Khosla et al. 2020](https://arxiv.org/abs/2004.11362)). 
Citing the *[Extending Contrastive Learning to the Supervised Setting
](https://blog.research.google/2021/06/extending-contrastive-learning-to.html)* blog post by Google:

> *"These contrastive learning approaches typically teach a model to pull together the representations of a target image (a.k.a., the “anchor”) and a matching (“positive”) image in embedding space, while also pushing apart the anchor from many non-matching (“negative”) images."*

[Kiros et al. 2014](https://arxiv.org/abs/1411.2539) brought this idea to the multimodal field, proposing a simmetric contrastive loss:

$$\max\{0, \alpha + sim(\textbf{Z}_L, \textbf{Z}^+_V) - sim(\textbf{Z}_L, \textbf{Z}^-_V)\} + \max\{0, \alpha + sim(\textbf{Z}_V, \textbf{Z}^+_L) - sim(\textbf{Z}_V, \textbf{Z}^-_L)\}$$

Where similarity function is applied to positive samples of language and visual samples $sim(\textbf{Z}_L, \textbf{Z}^+_V)$ and to negative samples of language and visual samples $sim(\textbf{Z}_L, \textbf{Z}^-_V)$, and the other way around.   

| ![coordination contrastive loss](/assets/2023-08-01-multimodal/coordination_contrastive_loss.jpg)| 
|:--:|                 
| *Figure 18*: Coordination with a contrastive loss.|

The two separated autoencoders are trained and coordinated together trough the contrastive loss. A fashinating property emerged from the training: the ability to do aritmetics between these two different but coordinated embeddings. At test time, the researchers encoded an image of a red car, then they encoded the words *"red"* and *"blue"*. 

| ![coordination contrastive loss](/assets/2023-08-01-multimodal/contrastive_loss_showcase.jpg)| 
|:--:|                 
| *Figure 19*: Multimodal vector space arithmetic. Credits: [Kiros et al. 2014](https://arxiv.org/abs/1411.2539)|


Seven years after, the *Contrastive Language-Image Pre-training (CLIP)* by [Radford et al. 2021](https://arxiv.org/abs/2103.00020) (OpenAI), has been released. It shares the same principles of [Kiros et al. 2014](https://arxiv.org/abs/1411.2539) with a slightly different loss. CLIP was a breakthrough proving that the large-scale weak-supervision contrastive training paradigm can be applied also for vision and not only for text (such as GPT). The pretrain dataset comprises of 400 million (image, text) pairs and the learning objective is to predict the captions that go with a particular image. During pretraining, CLIP learns to associate images and text by contrasting positive pairs (an image and its corresponding text) against negative pairs (an image with randomly sampled text and vice versa). The model is trained to make the positive pair's similarity higher (matrix diagonal in Fig.20) than the negative pair's similarity.

| ![clip](/assets/2023-08-01-multimodal/clip.jpg)| 
|:--:|                 
| *Figure 20*: CLIP usage workflow. Credits to: [OpenAI blog](https://openai.com/research/clip).|

This method gave rise to a powerful zero-shot multimodal model. From [OpenAI's blog post on CLIP](https://openai.com/research/clip):

> *"Because they learn a wide range of visual concepts directly from natural language, CLIP models are significantly more flexible and general than existing ImageNet models. We find they are able to zero-shot perform many different tasks. To validate this we have measured CLIP’s zero-shot performance on over 30 different datasets including tasks such as fine-grained object classification, geo-localization, action recognition in videos, and OCR."*

To discover more about CLIP I suggest two blog posts: [A. Chadha, V. Jain Paper Reviews, 2022](https://www.vinija.ai) and [OpenAI's blog post on CLIP](https://openai.com/research/clip). 

Contrastive representation learning is a very wide field of study. I'll recommend the [Lil'Log blog post]((https://lilianweng.github.io/posts/2021-05-31-contrastive)) that gives a very nice overview of the contrastive losses and key principles.
 

### Representation by Fission
In this case, the number of representations is greater then the number of modalities. Modalities can interact with each other in more than one way. Let's think about language and vision. A word can have different types interactions: it can directly correspond to a name of an object or it can have a different type of relation with that object (e.g. parent, use by, ..). Fission is a way to model all these types of relationships with different representations. In this sense we could interchange the name *fission* with *factorization*.  

| ![types of fissions](/assets/2023-08-01-multimodal/fission.jpg)| 
|:--:|                 
| *Figure 21*: Sub-challenge of Fission. Credits to: [CMU Multimodal Machine Learning course, Fall 2022](https://www.youtube.com/watch?v=65xxHVyHKi0&list=PL-Fhd_vrvisNM7pbbevXKAbT_Xmub37fA&index=9&t=905s) |


> **Work in progress....**