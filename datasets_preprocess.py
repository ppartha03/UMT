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

import time

#from comet_ml import OfflineExperiment

from perturbations import *

from datasets import load_dataset

# commandline arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data > Model > Lang > Perturbation > source.perturb -
#                                    > target.perturb -
#                                    > source.gold -
#                                    > target.gold -
#                                    > source.gold.to.target
#                                    > source.perturb.to.target

def HyperEvaluate(config):
    ext_language = config['lang']
    perturbation = config['perturb']
    model = config['model']

    try:
        nlp_o = spacy.load(ext_language+"_core_news_sm")
    except:
        nlp_o = spacy.load(ext_language+"_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    perturbations = [treeMirrorPre, treeMirrorPo, treeMirrorIn, verbAtBeginning, verbSwaps, adverbVerbSwap,
      nounVerbSwap, nounVerbMismatched, nounAdjSwap, shuffleHalvesFirst, shuffleHalvesLast,
      reversed, wordShuffle, rotateAroundRoot,functionalShuffle, nounSwaps, conjunctionShuffle]

    assert perturbation in perturbations

    # todo : Save samples in a csv : with metrics, perturbed example and beams
    # todo : ensure the beams have the same seed across runs and so do the perturbation functions. Use the index as seed value.

    target_dir = os.path.join('Data', model, ext_language, perturbation.__name__)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    eng_gold_file =  open(os.path.join(target_dir, 'en.gold'), "w")
    eng_perturb_file = open(os.path.join(target_dir, 'en.perturb'), "w")

    o_lang_gold_file = open(os.path.join(target_dir, ext_language + '.gold'), "w")
    o_lang_perturb_file = open(os.path.join(target_dir, ext_language + '.perturb'), "w")

    lock = FileLock(os.path.join(target_dir, 'lock.l'))

    counter_file = open(os.path.join('Data', model, ext_language,'stats.txt'),'a')

    original_id = 0
    id_ = 0

    dataset = load_dataset(model, ext_language+'-en', split = 'validation')

    this_perturbation = total_sentences = len(dataset)
    with lock:
        for i in range(len(dataset)):
            try:
                other_lang_gold = dataset[i]['translation'][ext_language]
                eng_gold = dataset[i]['translation']['en']

                eng_perturbed = perturbation(eng_gold, nlp = nlp)
                other_lang_perturbed = perturbation(other_lang_gold, nlp = nlp_o)

                d = {"original_id": original_id, "id": id_, "text": eng_gold}
                d_f = json.dumps(d)
                eng_gold_file.write(d_f + '\n')

                d = {"original_id": original_id, "id": id_, "text": other_lang_gold}
                d_f = json.dumps(d)
                o_lang_gold_file.write(d_f + '\n')

                d = {"original_id": original_id, "id": id_, "text": eng_perturbed}
                d_f = json.dumps(d)
                eng_perturb_file.write(d_f + '\n')

                d = {"original_id": original_id, "id": id_, "text": other_lang_perturbed}
                d_f = json.dumps(d)
                o_lang_perturb_file.write(d_f + '\n')

                id_ += 1
            except AssertionError:
                this_perturbation -= 1
                pass
            except:
                this_perturbation -= 1
                pass
            original_id += 1

    counter_lock = FileLock(os.path.join('Data', model, ext_language, 'counter_lock'))

    with counter_lock:
        counter_file = open(os.path.join('Data', model, ext_language,'stats.txt'),'a')
        counter_file.write(perturbation.__name__ + ' : ' + str(this_perturbation) + '/' + str(total_sentences) + '\n')
    counter_file.close()

    o_lang_gold_file.close()
    eng_gold_file.close()

    eng_perturb_file.close()
    o_lang_perturb_file.close()

    counter_file.close()


if __name__ == '__main__':

    PARAM_GRID = list(product(
    ['wmt19'], #model For wmt18 ['de', 'ru', 'zh']
    ['de', 'lt', 'ru', 'zh'], #languages#[verbAtBeginning]
    [treeMirrorPre, treeMirrorPo, treeMirrorIn, verbSwaps, adverbVerbSwap, verbAtBeginning,
      nounVerbSwap, nounVerbMismatched, nounAdjSwap, shuffleHalvesFirst, shuffleHalvesLast,
      reversed, wordShuffle, rotateAroundRoot,functionalShuffle, nounSwaps, conjunctionShuffle],
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
        timeout_min=30,
        gpus_per_node=num_gpus,
        slurm_additional_parameters={"account": "rrg-bengioy-ad"},
        tasks_per_node=num_gpus,
        cpus_per_task=workers_per_gpu,
        slurm_mem="8G",#16G
        slurm_array_parallelism=100,
    )
    job = executor.map_array(HyperEvaluate,h_param_list)
    print('Jobs submitted!')
