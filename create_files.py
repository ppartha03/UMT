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
import spacy
import math
import csv
import wandb
from itertools import product
import sys
from filelock import FileLock
from nltk.translate.bleu_score import sentence_bleu as bleu_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from bleurt import score

import time

#from comet_ml import OfflineExperiment
import torch
from torch.autograd import Variable
import torch.nn as nn

import pandas as pd
from collections import Counter
import pdb

from perturbations import *

os.environ["WANDB_API_KEY"] = '829432a2360cc623158d30f47c37fe11d3e12d57'
os.environ["WANDB_MODE"] = "dryrun"

# commandline arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data > Model > Lang > Perturbation > eng.perturb -
#                                    > lang.perturb -
#                                    > eng.gold -
#                                    > lang.gold -
#                                    > eng.gold.translate
#                                    > eng.perturb.translate

def HyperEvaluate(config):
    ext_language = config['lang']
    perturbation = config['perturb']
    model = config['model']

    nlp_o = spacy.load(ext_language+"_core_news_sm")
    nlp = spacy.load("en_core_web_sm")

    perturbations = [treeMirrorPre, treeMirrorPo, treeMirrorIn, verbAtBeginning, verbSwaps, adverbVerbSwap,
      nounVerbSwap, nounVerbMismatched, nounAdjSwap, shuffleHalvesFirst, shuffleHalvesLast,
      reversed, wordShuffle, rotateAroundRoot,functionalShuffle, nounSwaps, conjunctionShuffle]

    assert perturbation in perturbations

    test_file = open('test_data/opus-'+ext_language+'-en.test.txt')

    lines = test_file.readlines()

    # todo : Save samples in a csv : with metrics, perturbed example and beams
    # todo : ensure the beams have the same seed across runs and so do the perturbation functions. Use the index as seed value.

    target_dir = os.path.join('Data', model, ext_language, perturbation.__name__)
    if not os.path.exists(target_dir):
        os.path.makedirs(target_dir)

    eng_gold_file =  open(os.path.join(target_dir, 'en.gold'), "w")
    eng_perturb_file = open(os.path.join(target_dir, 'en.perturb'), "w")

    o_lang_gold_file = open(os.path.join(target_dir, ext_language + '.gold'), "w")
    o_lang_perturb_file = open(os.path.join(target_dir, ext_language + '.perturb'), "w")

    lock = FileLock(os.path.join(open(os.path.join('target_dir', 'lock.l'))

    counter_file = open(os.path.join('Data', model, ext_language,'stats.txt'),'a')

    this_perturbation = total_sentences = len(lines) //4

    with lock:
        for i in range(0,len(lines),4):
            try:
                other_lang_gold = lines[i].strip()
                eng_gold = lines[i+1].strip()

                eng_perturbed = config['perturb'](eng_gold)
                other_lang_perturbed = config['perturb'](other_lang_gold)

                eng_gold_file.write(eng_gold + '\n')
                o_lang_gold_file.write(other_lang_gold + '\n')

                eng_perturb_file.write(eng_perturbed + '\n')
                o_lang_perturb_file.write(other_lang_perturbed + '\n')

            except AssertionError:
                this_perturbation -= 1
                pass

        counter_file.write(perturbation.__name__ + ' : ' + str(this_perturbation) + '/' + str(len(lines)))

    o_lang_gold_file.close()
    eng_gold_file.close()

    eng_perturb_file.close()
    o_lang_perturb_file.close()

    counter_file.close()


if __name__ == '__main__':

    PARAM_GRID = list(product(
    ['Helsinki-opus'], #model
    ['de'],#'fr','ru','ja'], #languages
    [treeMirrorPre],#, treeMirrorPo, treeMirrorIn, verbAtBeginning, verbSwaps, adverbVerbSwap,
      #nounVerbSwap, nounVerbMismatched, nounAdjSwap, shuffleHalvesFirst, shuffleHalvesLast,
      #reversed, wordShuffle, rotateAroundRoot,functionalShuffle, nounSwaps, conjunctionShuffle],
    )
    )

    h_param_list = []

    for param_ix in range(len(PARAM_GRID)):

        params = PARAM_GRID[param_ix]


        model, lang, pert = params
        config = {}
        config['lang'] = lang
        config['perturb'] = pert
        config['model'] = model

        h_param_list.append(config)

    # run by submitit
    d = datetime.today()
    exp_dir = (
        Path("./dumps/")
        / "projects"
        / "UMT"
        / "dumps"
        / f"{d.strftime('%Y-%m-%d')}_rand_eval"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    submitit_logdir = exp_dir / "submitit_logs"
    num_gpus = 1
    workers_per_gpu = 10
    executor = submitit.AutoExecutor(folder=submitit_logdir)
    executor.update_parameters(
        timeout_min=15,
        gpus_per_node=num_gpus,
        slurm_additional_parameters={"account": "rrg-bengioy-ad"},
        tasks_per_node=num_gpus,
        cpus_per_task=workers_per_gpu,
        slurm_mem="32G",#16G
        slurm_array_parallelism=50,
    )
    job = executor.map_array(HyperEvaluate,h_param_list)
    print('Jobs submitted!')
