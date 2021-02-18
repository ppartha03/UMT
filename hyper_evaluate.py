import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import submitit
from pathlib import Path
from datetime import datetime
import random
import itertools
import math
import csv
import wandb
import sys
from filelock import FileLock
from nltk.translate.bleu_score import sentence_bleu as bleu_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from bleurt import score

import time

#from comet_ml import OfflineExperiment
import torch
from torch.autograd import Variable
import torch.nn as nn

from infersent_comp.data import get_nli, get_batch, build_vocab, DICO_LABEL
from infersent_comp.mutils import get_optimizer
from infersent_comp.models import NLINet
import pandas as pd
from collections import Counter
import pdb

from perturbations import *

os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = "dryrun"

# commandline arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hyperEvaluate(config):
    ext_language = config['lang']
    perturbation = config['perturb']

    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-" + ext_language)
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-" + ext_language)

    tokenizer_inv = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-" + ext_language + "-en")
    model_inv = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-" + ext_language + "-en")

    translation = pipeline("translation_en_to_outer", model=model, tokenizer=tokenizer)
    translation_inv = pipeline("translation_en_back", model=model_inv, tokenizer=tokenizer_inv)

    perturbations = ['treeMirror', 'verbAtBeginning', 'verbSwaps', 'adverbVerbSwap',
    'nounVerbSwap', 'nounVerbMismatched', 'nounAdjSwap', 'shuffleHalvesFirst', 'shuffleHalvesLast',
    'reversed', 'shuffle', 'rotateAroundRoot','functionalShuffle']

    assert perturbation in perturbations

    test_file = open('test_data/opus-'+ext_language+'-en.test.txt')

    lines = test_file.readlines()

    sys.argv = sys.argv[:1]

    bleurt_ops = score.create_bleurt_ops('/home/pparth2/scratch/UMT/UMT/Results/cached/bleurt-base-128')

    wandb.init(project="UMT-" + config['lang'] + '-' + config['preturb'], reinit = True)
    wandb.run.name = config['lang'] + '-' + config['preturb']

    wandb.config.update(config)

    for i in range(0,len(lines),4):

        other_lang_gold = lines[i]
        eng_gold = lines[i+1]

        eng_perturbed = eval(config['perturb']+'(\"'+eng_gold+'\")')

        other_lang_translated = translation(eng_gold, max_length=400)[0]['translation_text']
        other_lang_translated_p = translation(eng_perturbed, max_length=400)[0]['translation_text']

        eng_back_translated = translation_inv(other_lang_translated, max_length=400)[0]['translation_text']
        eng_back_translated_p = translation_inv(other_lang_translated_p, max_length=400)[0]['translation_text']

        # BLEU-2

        bleu2_metric_eng_p_eng_gold = bleu_score([eng_gold.split()], eng_perturbed.split(), (0.5,0.5))
        bleu2_metric_eng_back_eng_gold = bleu_score([eng_gold.split()], eng_back_translated_p.split(), (0.5,0.5))

        bleu2_metric_eng_p_eng_gold = bleu_score([eng_gold.split()], eng_perturbed.split(), (0.5,0.5))
        bleu2_metric_eng_back_eng_gold = bleu_score([eng_gold.split()], eng_back_translated_p.split(), (0.5,0.5))

        M1_bleu2 = bleu2_metric_eng_eng_gold / bleu2_metric_eng_p_eng_gold
        M2_bleu2 = bleu_score([eng_perturbed.split()], eng_back_translated_p.split(), (0.5,0.5))

        # BLEU-4

        bleu4_metric_eng_p_eng_gold = bleu_score([eng_gold.split()], eng_perturbed.split(), (0.25,0.25,0.25,0.25))
        bleu4_metric_eng_back_eng_gold = bleu_score([eng_gold.split()], eng_back_translated_p.split(), (0.25,0.25,0.25,0.25))

        bleu4_metric_eng_p_eng_gold = bleu_score([eng_gold.split()], eng_perturbed.split(), (0.25,0.25,0.25,0.25))
        bleu4_metric_eng_back_eng_gold = bleu_score([eng_gold.split()], eng_back_translated_p.split(), (0.25,0.25,0.25,0.25))

        M1_bleu4 = bleu4_metric_eng_eng_gold / bleu4_metric_eng_p_eng_gold
        M2_bleu4 = bleu_score([eng_perturbed.split()], eng_back_translated_p.split(), (0.25,0.25,0.25,0.25))

        # BLEURT

        bleurt_out = bleurt_ops([eng_perturbed], [eng_back_translated_p])
        assert bleurt_out["predictions"].shape == (1,)
        M2_bleu_rt = float(bleurt_out["predictions"])

        bleurt_out = bleurt_ops([eng_gold], [eng_perturbed])
        assert bleurt_out["predictions"].shape == (1,)
        bleurt_metric_eng_p_eng_gold = float(bleurt_out["predictions"])

        bleurt_out = bleurt_ops([eng_gold], [eng_back_translated_p])
        assert bleurt_out["predictions"].shape == (1,)
        bleurt_metric_eng_back_eng_gold = float(bleurt_out["predictions"])

        wandb.log({
        "index": i,
        "M1_Bleu2": M1_bleu2,
        "M2_Bleu2": M2_bleu2,
        "bleu2_e_p_vs_e_gold":bleu2_metric_eng_p_eng_gold,
        "bleu2_e_back_vs_e_gold":bleu2_metric_eng_back_eng_gold,
        "M1_Bleu4": M1_bleu4,
        "M2_Bleu4": M2_bleu4,
        "bleu4_e_p_vs_e_gold":bleu4_metric_eng_p_eng_gold,
        "bleu4_e_back_vs_e_gold":bleu4_metric_eng_back_eng_gold,
        "M2_bleurt": M2_bleu_rt,
        "bleurt_e_p_vs_e_gold": bleurt_metric_eng_p_eng_gold,
        "bleurt_e_back_vs_e_gold": bleurt_metric_eng_back_eng_gold
        })

if __name__ == '__main__':

    PARAM_GRID = list(product(
    ['de'],#'fr','ru','jap'], #languages
    ['shuffle']#, 'verbAtBeginning', 'verbSwaps', 'adverbVerbSwap',
    #'nounVerbSwap', 'nounVerbMismatched', 'nounAdjSwap', 'shuffleHalvesFirst', 'shuffleHalvesLast',
    #'reversed', 'treeMirror', 'rotateAroundRoot','functionalShuffle'] #perturbations
    )
    )

    h_param_list = []

    for param_ix in range(this_worker, len(PARAM_GRID), N_WORKERS):

        params = PARAM_GRID[param_ix]


        lang, pert = params
        config = {}
        config['lang'] = lang
        config['perturb'] = pert

        h_param_list.append([config])

    if myargs.is_slurm:
    # run by submitit
        d = datetime.today()
        exp_dir = (
            Path("./dumps/")
            / "projects"
            / "UMT"
            / "dumps"
            / f"{d.strftime('%Y-%m-%d')}_rand_eval}"
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        submitit_logdir = exp_dir / "submitit_logs"
        num_gpus = 1
        workers_per_gpu = 10
        executor = submitit.AutoExecutor(folder=submitit_logdir)
        executor.update_parameters(
            timeout_min=60,
            gpus_per_node=num_gpus,
            slurm_additional_parameters={"account": "rrg-bengioy-ad"},
            tasks_per_node=num_gpus,
            cpus_per_task=workers_per_gpu,
            slurm_mem="32G",#16G
            slurm_array_parallelism=50,
        )
        job = executor.map_array(HyperEvaluate,h_param_list)
        print('Jobs submitted!')


    else:
        print("Don\'t provide the slurm argument")
