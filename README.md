# UCPM: Uncertainty-Guided Cross-Modal Retrieval with Partially Mismatched Pairs
The official pytorch implementation of [UCPM: Uncertainty-Guided Cross-Modal Retrieval with Partially Mismatched Pairs]() (submitted to IEEE TIP). 

## Introduction
### UCPM framework
<img src="https://github.com/qxzha/UCPM/main/framework.png" >

## DATASETS 
Our directory structure of ```data```.
```
data
├── f30k_precomp # pre-computed BUTD region features for Flickr30K, provided by SCAN
│     ├── train_ids.txt
│     ├── train_caps.txt
│     ├── ......
│
├── coco_precomp # pre-computed BUTD region features for COCO, provided by SCAN
│     ├── train_ids.txt
│     ├── train_caps.txt
│     ├── ......
│
├── cc152k_precomp # pre-computed BUTD region features for cc152k, provided by NCR
│     ├── train_ids.txt
│     ├── train_caps.tsv
│     ├── ......
│
└── vocab  # vocab files provided by SCAN and NCR
      ├── f30k_precomp_vocab.json
      ├── coco_precomp_vocab.json
      └── cc152k_precomp_vocab.json
```
### Downloads
We follow [NCR](https://github.com/XLearning-SCU/2021-NeurIPS-NCR) to obtain image features and vocabularies.
[Download Dataset](https://ncr-paper.cdn.bcebos.com/data/NCR-data.tar)

### Noise Index
if you want to experiment with the same noise index as in this paper, the noise index files can be downloaded from [here](https://drive.google.com/file/d/1JG0-dIS_d8SdaUw-Bbgf00rL8nOkHA5J/view?usp=drive_link).
