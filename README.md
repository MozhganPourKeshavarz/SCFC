# SCFC
Stacked Cross-modal Feature Consolidation Attention Networks for Image Captioning


## Abstract
Recently, the attention-enriched encoder-decoder framework has aroused great interest in image cap- tioning due to its overwhelming progress. Many visual attention models directly leverage meaningful regions to generate image descriptions. However, seeking a direct transition from visual space to text is not enough to generate fine-grained captions. This paper exploits a feature-compounding approach to bring together high-level semantic concepts and visual information regarding the contextual environment fully end-to-end. Thus, we propose a stacked cross-modal feature consolidation (SCFC) attention net- work for image captioning in which we simultaneously consolidate cross-modal features through a novel compounding function in a multi-step reasoning fashion. Besides, we jointly employ spatial information and context-aware attributes (CAA) as the principal components in our proposed compounding func- tion, where our CAA provides a concise context-sensitive semantic representation. To make better use of consolidated features potential, we further propose an SCFC-LSTM as the caption generator, which can leverage discriminative semantic information through the caption generation process. The experi- mental results indicate that our proposed SCFC can outperform various state-of-the-art image captioning benchmarks in terms of popular metrics on the MSCOCO and Flickr30K datasets.


<img width="960" alt="Screen Shot 2023-05-03 at 11 43 23 AM" src="https://user-images.githubusercontent.com/36272225/235967846-c8d777fb-9b13-464d-af35-faa0e451eddd.png">


## Citation

If you use this code, please cite our paper:

```
@article{pourkeshavarz2023stacked,
  title={Stacked Cross-modal Feature Consolidation Attention Networks for Image Captioning},
  author={Pourkeshavarz, Mozhgan and Nabavi, Shahabedin and Moghaddam, Mohsen Ebrahimi and Shamsfard, Mehrnoush},
  journal={arXiv preprint arXiv:2302.04676},
  year={2023}
}
```
