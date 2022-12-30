# Leveraging QA Datasets to Improve Generative Data Augmentation

## Code is being updated!! Expect rapid & frequent changes!

- [Training](#training)
	- [Commands](#commands)
	- [Requirements](#requirements)

In this folder, we provide code to run ConDA-SocialIQA on AGNews and SST datasets.

We view QA dataset in Question-Answer-Context (QAC) format and fine-tune GPT2. 
In this repo, we consider SocialIQA as example and provide code for the same.
The provided code can be easily modified for other QA datasets.

### Commands

The ```agnews.sh``` or ```sst.sh``` requires two arguments: 
- ```gpu_id``` refers to the id of the gpu. 
- ```tmp_path```, refers to the destination to dump the models.

Example command to run:
```shell script
$ sh scripts/agnews.sh 1 data/tmp
$ sh scripts/sst.sh 1 data/tmp
```

Above scripts perform:
1. QAC fine-tuning on SocialIQA dataset.
2. 

### Requirements

This project is based on ```python==3.7```. The dependencies are as follow:
```
scikit-learn==0.21.3
torch==1.9.1
argparse
transformers==4.3.3
nltk
scipy=1.3.1
numpy==1.17.2
```
