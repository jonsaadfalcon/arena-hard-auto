

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
perform_ensembling = True
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
    
    print(f"Generating candidates for models: {models}")
    
    model_name = models[0]
    model_id = model_name.split("/")[1]
    saved_jsonl_path = f"data/arena-hard-v0.1/model_answer/{model_name}.jsonl"
    if not os.path.exists(saved_jsonl_path):
        candidate_generation_command = "python gen_answer.py"

        print("Generation Command: ", candidate_generation_command)
        print("Generating candidates...")
        with subprocess.Popen(candidate_generation_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            for line in process.stdout:
                print(line, end='')  # Print the output in real-time

    else:
        print(f"Model {model_name} already has candidates generated. Already saved to: {saved_jsonl_path}")

    ##########################################

    judgement_command = "python gen_judgment.py"
    print("Generating judgements...")
    with subprocess.Popen(judgement_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            print(line, end='')  # Print the output in real-time

    ##########################################

    show_results_command = "python show_result.py"
    print("Showing results...")
    show_results_result = subprocess.run(show_results_command, shell=True, capture_output=True, text=True)

    print("------------------------------------------------")
    print(f"Arena-Hard-Auto Results for {model_name}:")
    for line in show_results_result.stdout.split("\n"):
        print(line)
    print("------------------------------------------------")

else:

    print("Performing ensembling for MT Bench...")

    # Check if dataset already exists
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ensemble_model_id = f"ensemble_{timestamp}"
    final_dataset_path = f"data/arena-hard-v0.1/model_answer/{ensemble_model_id}.jsonl"

    #################################################

    # Load datasets
    total_datasets = []
    print("Loading Models...")
    for model_name in models:
        model_id = model_name.split("/")[1]
        print(f"Loading model: {model_id}")
        saved_jsonl_path = f"data/arena-hard-v0.1/model_answer/{model_id}.jsonl"
        dataset = pd.read_json(saved_jsonl_path, lines=True)
        total_datasets.append(dataset)

    #################################################

    # Perform ranking over candidates
    import llm_blender
    blender = llm_blender.Blender()
    blender.loadranker(ranker_config['ranker_checkpoint'])

    #################################################

    breakpoint()

    # Score first turn candidates
    assert len(first_turn_instructions) == len(first_turn_candidates)
    print("Performing Ensemble Candidate Ranking for First Turn Candidates with PairRM Ranker...")
    scores = blender.rank(first_turn_instructions, first_turn_candidates, return_scores=True, batch_size=ranker_config['ranker_batch_size'])
    first_ranks = [sorted(range(len(score)), key=lambda i: score[i], reverse=True) for score in scores]
        
    assert len(first_ranks) == len(first_turn_candidates)
    first_turn_top_candidate_texts_from_ranker = [first_turn_candidates[i][first_ranks[i][0]] for i in range(len(first_ranks))]

    #################################################

    # Score second turn candidates
    assert len(second_turn_instructions) == len(second_turn_candidates)
    print("Performing Ensemble Candidate Ranking for Second Turn Candidates with PairRM Ranker...")
    scores = blender.rank(second_turn_instructions, second_turn_candidates, return_scores=True, batch_size=ranker_config['ranker_batch_size'])
    second_ranks = [sorted(range(len(score)), key=lambda i: score[i], reverse=True) for score in scores]

    assert len(second_ranks) == len(second_turn_candidates)
    second_turn_top_candidate_texts_from_ranker = [second_turn_candidates[i][second_ranks[i][0]] for i in range(len(second_ranks))]

    #################################################

    # Create new ensemble JSONL

    ensemble_choices = []
    for idx in range(len(first_turn_top_candidate_texts_from_ranker)):
        ensemble_choices.append([{
            "turns": [first_turn_top_candidate_texts_from_ranker[idx], second_turn_top_candidate_texts_from_ranker[idx]]
        }])

    question_ids = dataset["question_id"].tolist()
    answer_ids = dataset["answer_id"].tolist()
    model_ids = ["ensemble"] * len(question_ids)
    choices = ensemble_choices
    turn_instructions = [""] * len(question_ids)

    os.makedirs(os.path.dirname(final_dataset_path), exist_ok=True)
    with open(os.path.expanduser(final_dataset_path), "a") as fout:
        for question_id, answer_id, model_id, choices, turn_instruction in zip(question_ids, answer_ids, model_ids, choices, turn_instructions):
            ans_json = {
                "question_id": question_id,
                "answer_id": answer_id,
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
                "turn_instruction": turn_instruction,
            }
            fout.write(json.dumps(ans_json) + "\n")

    #################################################

    judgement_command = f"python gen_judgment.py --model-list {ensemble_model_id} --parallel 2 --judge-model gpt-4"
    print("Generating judgements...")
    judgement_result = subprocess.run(judgement_command, shell=True, capture_output=True, text=True)
    #breakpoint()
    #with subprocess.Popen(judgement_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
    #    for line in process.stdout:
    #        print(line, end='')  # Print the output in real-time

    print("------------------------------------------------")
    print(f"Judgement Results for {ensemble_model_id}:")
    for line in judgement_result.stdout.split("\n"):
        print(line)
    print("------------------------------------------------")

    ##########################################

    show_results_command = f"python show_result.py --model-list {ensemble_model_id}"
    print("Showing results...")
    show_results_result = subprocess.run(show_results_command, shell=True, capture_output=True, text=True)

    print("------------------------------------------------")
    print(f"MTBench Results for {ensemble_model_id}:")
    for line in show_results_result.stdout.split("\n"):
        print(line)
    print("------------------------------------------------")