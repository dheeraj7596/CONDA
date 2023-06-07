# Leveraging QA Datasets to Improve Generative Data Augmentation

This repository contains code to run ConDA-SocialIQA.
We consider SocialIQA dataset as an example QA dataset and the provided code can be easily modified for other QA datasets.

- [Framework](#framework)
- [Training](#training)
	- [Required Inputs](#required-inputs)
    - [Commands](#commands)
	- [Requirements](#requirements)
- [Citation](#citation)

## Framework
![CONWEA-Framework](docs/ConDA-overview.png)

## Training

### Required Inputs
Our framework requires QA datasets and target-task classification datasets.
* All datasets are in `csv` format.
* QA datasets are in `data/qa/` folder. Each sample is in Question-Answer-Context format.
* Target classification datasets are in `data/cls/` folder.
  * Each classification dataset has 3 sub-folders: `train, val, test`. We don't use validation data.
  * `Train` folder have three files corresponding to few-shot supervision obtained from three different random seeds.
  * Note that `train_qac_x.csv` is same as `train/train_x.csv` in QAC format, that is used for domain adaptation step. 


### Commands

The ```scripts/run.sh``` requires three arguments: 
- ```gpu_id``` refers to the id of the gpu. 
- ```tmp_path```, refers to the destination to dump the models.
- ```dataset```, refers to the dataset name. It has to be among ``[sst, agnews, imdb, nyt-coarse, yahoo, yelp]``.

Example command to run:
```shell script
$ sh scripts/run.sh 1 data/tmp agnews
```

Above script performs following thrice for three random seeds (13, 21, 42):
1. QAC fine-tune GPT2-Medium on SocialIQA dataset.
2. Domain Adaptation on target task dataset (AGNews in above example).
3. Generate synthetic training data
4. Train BERT-base classifier.

### Requirements

This project is based on ```python==3.7```. The dependencies are as follows:
```
scikit-learn==0.24.1
torch==1.9.1
argparse==1.1
transformers==4.11.3
datasets==1.12.1
nltk
scipy=1.6.2
numpy==1.20.3
```

## Citation

```
@inproceedings{mekala-etal-2022-leveraging,
    title = "Leveraging {QA} Datasets to Improve Generative Data Augmentation",
    author = "Mekala, Dheeraj  and
      Vu, Tu  and
      Schick, Timo  and
      Shang, Jingbo",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.660",
    pages = "9737--9750",
    abstract = "The ability of generative language models (GLMs) to generate text has improved considerably in the last few years, enabling their use for generative data augmentation. In this work, we propose CONDA, an approach to further improve GLM{'}s ability to generate synthetic data by reformulating data generation as context generation for a given question-answer (QA) pair and leveraging QA datasets for training context generators. Then, we cast downstream tasks into the same question answering format and adapt the fine-tuned context generators to the target task domain. Finally, we use the fine-tuned GLM to generate relevant contexts, which are in turn used as synthetic training data for their corresponding tasks. We perform extensive experiments on multiple classification datasets and demonstrate substantial improvements in performance for both few- and zero-shot settings. Our analysis reveals that QA datasets that require high-level reasoning abilities (e.g., abstractive and common-sense QA datasets) tend to give the best boost in performance in both few-shot and zero-shot settings.",
}
```
