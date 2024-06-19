

import torch
from transformers import AutoTokenizer, DebertaForSequenceClassification 
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, GenerationConfig
from datasets import load_dataset, load_from_disk
import transformers
from transformers.pipelines.pt_utils import KeyDataset

import pandas as pd
from itertools import combinations, permutations
from typing import List

import random
from datasets import Dataset, DatasetDict
from datetime import datetime
import sys
import os
import logging
import random
random.seed(43) #43
from tabulate import tabulate
import openai
import time
from tqdm import tqdm
import anthropic
from safetensors.torch import load_model, save_model 
import shutil
import json
import subprocess

from datasets import Dataset, concatenate_datasets

#################################################

# Parameters

# Mixture of Agents Models
models = ["Qwen/Qwen1.5-72B-Chat", "Qwen/Qwen1.5-110B-Chat", "microsoft/WizardLM-2-8x22B",
          "mistralai/Mixtral-8x22B-Instruct-v0.1", "meta-llama/Llama-3-70b-chat-hf", "databricks/dbrx-instruct"]
MoA_models = models

# Set the models and their configs in:
# - api_config.yaml
# - gen_answer_config.yaml
# - judge_config.yaml 


# Generation Settings
generation_dict = {
    "batch_size": 8,
    "temperatures": [0.7], #0.9 #1.5
    "candidates_per_temp": [1],
    "generation_max_length": 512,
    "dataset_cutoff": 4, #3, None
    #"top_k": 10,
    #"top_p": 0.9
}

#################################################

# Ensembling Parameters
perform_ensembling = False
ranker_config = {
    "ranker_checkpoint": "llm-blender/PairRM",

    "ranker_model": "microsoft/deberta-v3-large",
    "ranker_max_length": 1024, #512, 1024
    "ranker_batch_size": 16, #32
    "source_max_length": 256, # 128, 256
    "candidate_max_length": 256, # 128, 256
    "device": "cuda:0"
}

#################################################

if not perform_ensembling:
    
    for model_name in models:

        print(f"Generating candidates for model: {model_name}")
        
        model_id = model_name.split("/")[1]
        saved_jsonl_path = f"data/arena-hard-v0.1/model_answer/{model_name}.jsonl"
        if not os.path.exists(saved_jsonl_path):
            candidate_generation_command = "python gen_answer.py"

            print("Generation Command: ", candidate_generation_command)
            print("Generating candidates...")
            #generation_result = subprocess.run(candidate_generation_command, shell=True, capture_output=True, text=True)
            with subprocess.Popen(candidate_generation_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
                for line in process.stdout:
                    print(line, end='')  # Print the output in real-time

        else:
            print(f"Model {model_name} already has candidates generated. Already saved to: {saved_jsonl_path}")

        ##########################################

        judgement_command = "python gen_judgment.py"
        print("Generating judgements...")
        #judgement_result = subprocess.run(judgement_command, shell=True, capture_output=True, text=True)
        #breakpoint()
        with subprocess.Popen(judgement_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            for line in process.stdout:
                print(line, end='')  # Print the output in real-time

        #print("------------------------------------------------")
        #print(f"Judgement Results for {model_name}:")
        #for line in judgement_result.stdout.split("\n"):
        #    print(line)
        #print("------------------------------------------------")

    ##########################################

    show_results_command = "python show_result.py"
    print("Showing results...")
    show_results_result = subprocess.run(show_results_command, shell=True, capture_output=True, text=True)

    print("------------------------------------------------")
    print(f"Arena-Hard-Auto Results for {model_name}:")
    for line in show_results_result.stdout.split("\n"):
        print(line)
    print("------------------------------------------------")