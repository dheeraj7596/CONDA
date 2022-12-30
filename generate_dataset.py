#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import pickle
import pandas as pd
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
# from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def generate(tokenizer, model, q_str, a_str, strategy, num_tries=5, batch_size=150):
    model.eval()
    start_question_tag = "question :"
    end_question_tag = "\n"
    start_answer_tag = "answer :"
    end_answer_tag = "\n"
    start_context_tag = "context :"

    text = start_question_tag + " " + q_str + " " + end_question_tag + " " + \
           start_answer_tag + " " + a_str + " " + end_answer_tag + " " + start_context_tag

    encoded_dict = tokenizer.encode_plus(text, return_tensors='pt')
    ids = encoded_dict['input_ids'].to(device='cuda')

    assert strategy in ["topk", "topp"]
    if num_tries <= batch_size:
        num_samples_list = [num_tries]
    else:
        its = int(num_tries / batch_size)
        remainder = num_tries % batch_size
        if remainder == 0:
            num_samples_list = [batch_size] * its
        else:
            num_samples_list = [batch_size] * its + [remainder]
    try:
        if strategy == "topk":
            sample_outputs = []
            for n in num_samples_list:
                temp = model.generate(
                    input_ids=ids,
                    do_sample=True,
                    top_k=20,
                    max_length=200,
                    num_return_sequences=n
                )
                sample_outputs += temp
        elif strategy == "topp":
            sample_outputs = []
            for n in num_samples_list:
                temp = model.generate(
                    input_ids=ids,
                    do_sample=True,
                    top_p=0.95,
                    max_length=200,
                    num_return_sequences=n
                )
                sample_outputs += temp
    except Exception as e:
        print(e)
        return []

    ans = []
    for t in sample_outputs:
        whole_part = tokenizer.decode(t)
        ans.append(whole_part)

    return ans


def compare(gen_contexts, answers):
    length = len(gen_contexts)
    assert len(gen_contexts) == len(answers)
    anals = []
    for i in range(length):
        c = gen_contexts[i]
        print(c)
        print("*" * 80)
        a = answers[i]
        context = c.split("context :")[1]
        if a in context:
            anals.append(1)
        else:
            anals.append(0)
    return sum(anals) / length


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        required=True,
        help="Decoding strategy",
    )
    parser.add_argument(
        "--num_tries",
        type=int,
        default=1,
        help="Number of sequences to be generated for each Q&A",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--analyze_file", type=str, default=None, help="A csv or a json file containing the analysis data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--analyze_answers", action="store_true", help="Check answers in generated contexts"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # Sanity checks
    # if args.dataset_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a dataset name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."
    #
    # if args.push_to_hub:
    #     assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def generation_handler(dataset_name, tokenizer, model, strategy, num_tries):
    contexts = []
    labels = []

    assert dataset_name in ["sst", "agnews"]

    if dataset_name == "sst":
        answers = ["positive", "negative"]
        questions = ["is this sentence positive or negative?"] * len(answers)
        for q, a in zip(questions, answers):
            temp_contexts = generate(tokenizer, model, q, a, strategy=strategy, num_tries=num_tries)
            contexts += temp_contexts
            if a == answers[0]:
                labels += ["positive"] * num_tries
            else:
                labels += ["negative"] * num_tries
    elif dataset_name == "agnews":
        answers = ['sports', 'business', 'technology', 'politics']
        questions = ["what is this document about?"] * len(answers)
        for q, a in zip(questions, answers):
            temp_contexts = generate(tokenizer, model, q, a, strategy=args.strategy, num_tries=args.num_tries)
            contexts += temp_contexts
            labels += [a] * args.num_tries

    df = pd.DataFrame.from_dict({"generated_context": contexts, "label": labels})
    return df


if __name__ == "__main__":
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                # repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
                repo_name = ""
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.to(device='cuda')
    model.resize_token_embeddings(len(tokenizer))

    df = generation_handler(args.dataset_name, tokenizer, model, strategy=args.strategy, num_tries=args.num_tries)
    df.to_csv(os.path.join(args.output_dir, "df_gen_" + args.strategy + "_" + str(args.num_tries) + "_sampling.csv"),
              index=False)
    pickle.dump(df, open(
        os.path.join(args.output_dir, "df_gen_" + args.strategy + "_" + str(args.num_tries) + "_sampling.pkl"), "wb"))
