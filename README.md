# Leveraging QA Datasets to Improve Generative Data Augmentation

## Code is being updated!! Expect rapid & frequent changes!
We provide code to run ConDA-SocialIQA on AGNews and SST datasets.

In this repo, we consider SocialIQA dataset as an example and provide code for the same.
The provided code can be easily modified for other QA datasets.

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


### Commands

The ```scripts/run.sh``` requires three arguments: 
- ```gpu_id``` refers to the id of the gpu. 
- ```tmp_path```, refers to the destination to dump the models.
- ```dataset```, refers to the dataset name. It has to be one among ``[sst, agnews]``.

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
scikit-learn==0.21.3
torch==1.9.1
argparse
transformers==4.3.3
nltk
scipy=1.3.1
numpy==1.17.2
```

## Citation

```
@inproceedings{Mekala2022LeveragingQD,
  title={Leveraging QA Datasets to Improve Generative Data Augmentation},
  author={Dheeraj Mekala and Tu Vu and Timo Schick and Jingbo Shang},
  year={2022}
}
```
