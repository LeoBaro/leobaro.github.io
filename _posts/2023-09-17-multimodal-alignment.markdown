
## Alignment
The second big research topic in multimodal learning is Alignment. The primary goal of Alignment is to bridge the semantic gap between different modalities. It makes it possible for the model to understand and correlate information from different modalities by mapping them into a shared feature space.



Right after BERT, VisualBERT by [Harold et al. (2019)](https://arxiv.org/abs/1908.03557), has been proposed as one of the first multi-modal transformer architectures, capable of processing natural language and visual data. In VisualBERT, word tokens and patch image embeddings are concatenated together (early-fusion) in a weakly aligned fashion: the text caption associated with the images is not explicitly related with the patches and the goal of the transformer is to align not only the words but also the visual objects. 

Another approach, ViLBERT by [Jiasen et al. 2019]() uses cross-attention between the image embeddings and the contextualized text embeddings. 

[Sun et al. 2019]() 
[Miech et al. 2020](end-to-end learning of visual representation from uncurated instructional videos)
[Zhu and Yang 2020](ActBert) 

All these approaches use a CNN. A new generation of pure vision transformer 

