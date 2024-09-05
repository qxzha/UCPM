# UCPM: Uncertainty-Guided Cross-Modal Retrieval with Partially Mismatched Pairs
The official pytorch implementation of [UCPM: Uncertainty-Guided Cross-Modal Retrieval with Partially Mismatched Pairs]() (submitted to IEEE TIP). 

## Introduction
### UCPM framework
<img src="https://github.com/qxzha/UCPM/blob/main/framework.png" >

## Requirements
- Python 3.8
- PyTorch 1.20.0
- numpy
- scikit-learn
- Punkt Sentence Tokenizer:

```
import nltk
nltk.download()
> d punkt
```
(Optional) if the above download failed, you can manually download it from [here](https://drive.google.com/file/d/1eY9FnCm1YbnU5PHwiay7agQjuwSPz3_Z/view?usp=drive_link).
The directory structure is:
```
/home/username/
├── nltk_data
│     ├── tokenizers
│          ├── punkt
│               ├── czech.pickle
│               ├── french.pickle
│               ├── polish.pickle
│               ├── ......
```

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
If you want to experiment with the same noise index as in this paper, the noise index files can be downloaded from [here](https://drive.google.com/file/d/1JG0-dIS_d8SdaUw-Bbgf00rL8nOkHA5J/view?usp=drive_link).

## Training and Evaluation
### Training
Coming soon

### Pre-trained models and Evaluation
The pre-trained models are available here:

1. CC152K model [Download](https://drive.google.com/file/d/1pmnNmxZDcO99Jb0li1_vU9kkrz3N7wAO/view?usp=drive_link)
2. F30k 20% MRate model [Download](https://drive.google.com/file/d/1Ut15QxkkaEjpDVIjU4xZb58HrKWuZcNA/view?usp=drive_link)
3. F30k 40% MRate model [Download](https://drive.google.com/file/d/1E83kUnr1gwrvPB0Ry4zFJwA0g3dfLsPp/view?usp=drive_link)
4. F30k 60% MRate model [Download](https://drive.google.com/file/d/10TidbMZ68iO0ERRF9M6_wxSDLXDB0QKv/view?usp=drive_link)
5. F30k 80% MRate model [Download](https://drive.google.com/file/d/1mnG8Nw9ZhpnCEIVYMBcEPgCgOsSd8SlG/view?usp=drive_link)
6. COCO 20% MRate model [Download](https://drive.google.com/file/d/1Ck6bReHF0rQjNeVEwmRKnBn_swBy2ej8/view?usp=drive_link)
7. COCO 40% MRate model [Download](https://drive.google.com/file/d/1nwwR8sHbJlz5fj7yHCrz9Q4dXmX4PLdM/view?usp=drive_link)
8. COCO 60% MRate model [Download](https://drive.google.com/file/d/1WG-GzfljnwdAoj9DlFIKfzw_YN4kXd7j/view?usp=drive_link)
9. COCO 80% MRate model [Download](https://drive.google.com/file/d/1_rbC88LOKthc7fmVxm3YPY4trKKoEFtg/view?usp=drive_link)

Modify the ```data_path```, ```vocab_path```, and ```model_paths``` in the ```eval.py``` file and run it.
```
python eval.py
```

## Experiment Results

### Results on Well-Annotated Datasets
<img src="https://github.com/qxzha/UCPM/blob/main/well_annotated.png" >

### Results on Simulated PMPs
<img src="https://github.com/qxzha/UCPM/blob/main/flickr_mscoco.png" >

### Results on Real-World PMPs
<img src="https://github.com/qxzha/UCPM/blob/main/cc152k.png" >

## License
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
