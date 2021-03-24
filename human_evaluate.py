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
import json
import wandb
from itertools import product
import sys
from filelock import FileLock
from utils.metrics import *

import time

#from comet_ml import OfflineExperiment

from perturbations import *


# Data > Model > Lang > Perturbation > source.perturb -
#                                    > target.perturb -
#                                    > source.gold -
#                                    > target.gold -
#                                    > source.gold.to.target
#                                    > source.perturb.to.target

def HyperEvaluate(config):
    ext_language = config['lang']
    perturbation = config['perturb']
    model_ = config['model']

    perturbations = [treeMirrorPre, treeMirrorPo, treeMirrorIn, verbAtBeginning, verbSwaps, adverbVerbSwap,
      nounVerbSwap, nounVerbMismatched, nounAdjSwap, shuffleHalvesFirst, shuffleHalvesLast,
      reversed, wordShuffle, rotateAroundRoot,functionalShuffle, nounSwaps, conjunctionShuffle]

    assert perturbation in perturbations

    # todo : Save samples in a csv : with metrics, perturbed example and beams
    # todo : ensure the beams have the same seed across runs and so do the perturbation functions. Use the index as seed value.

    samples_dir = os.path.join('Data', model_, ext_language, perturbation.__name__)
    human_eval_dir = os.path.join('Human', model_)

    if not os.path.exists(human_eval_dir):
        os.makedirs(human_eval_dir)


    perturb_id = perturbations.index(perturbation)

    eng_gold_file =  open(os.path.join(samples_dir, 'en.gold'), "r")
    eng_perturb_file = open(os.path.join(samples_dir, 'en.perturb'), "r")

    gold_human = open(os.path.join(human_eval_dir, 'en' + str(perturb_id) + '-human-gold.csv'), "w")
    eval_human = open(os.path.join(human_eval_dir, 'en' + str(perturb_id) + '-human-eval.csv'), "w")

    eng_gold_sents = eng_gold_file.readlines()
    eng_perturb_sents = eng_perturb_file.readlines()

    fieldnames = ['original_id', 'id', 'perturb_id', 'English Perturbed', 'English Fixed']

    indices = random.sample(range(len(eng_gold_sents)),20)

    # metrics - bleu2, bleu3, bleu4

    assert len(eng_gold_sents) == len(eng_perturb_sents)

    original_id = 0
    id_ = 0
    lock = FileLock(os.path.join(human_eval_dir, 'lock.l'))

    with lock:
        writer_NL_gold = csv.DictWriter(gold_human, fieldnames=fieldnames)
        writer_NL_eval = csv.DictWriter(eval_human, fieldnames=fieldnames)

        writer_NL_gold.writeheader()
        writer_NL_eval.writeheader()
        for ix in indices:
            en_gold = json.loads(eng_gold_sents[ix])
            en_perturb = json.loads(eng_perturb_sents[ix])

            original_id = en_gold['original_id']
            id_ = en_gold['id']

            d_eval = {
                 "original_id": original_id,
                 "id": id_,
                 "perturb_id": perturb_id,
                 "English Perturbed": en_perturb['text'],
                 "English Fixed": ''
                 }

            d_gold = {
                 "original_id": original_id,
                 "id": id_,
                 "perturb_id": perturb_id,
                 "English Perturbed": en_perturb['text'],
                 "English Fixed": en_gold['text']
                 }

            writer_NL_eval.writerow(d_eval)
            writer_NL_gold.writerow(d_gold)


    eng_gold_file.close()
    eng_perturb_file.close()

    gold_human.close()
    eval_human.close()


if __name__ == '__main__':

    PARAM_GRID = list(product(
    ['Helsinki-opus'], #model
    ['de'], #languages
    [treeMirrorPre, treeMirrorPo, treeMirrorIn, verbSwaps, adverbVerbSwap, verbAtBeginning,
      nounVerbSwap, nounVerbMismatched, nounAdjSwap, shuffleHalvesFirst, shuffleHalvesLast,
      reversed, wordShuffle, rotateAroundRoot,functionalShuffle, nounSwaps, conjunctionShuffle]
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

        if config not in h_param_list:
            h_param_list.append(config)

    print(len(h_param_list))
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
        timeout_min=10,
        gpus_per_node=num_gpus,
        slurm_additional_parameters={"account": "rrg-bengioy-ad"},
        tasks_per_node=num_gpus,
        cpus_per_task=workers_per_gpu,
        slurm_mem="16G",#16G
        slurm_array_parallelism=100,
    )
    job = executor.map_array(HyperEvaluate,h_param_list)
    print('Jobs submitted!')
