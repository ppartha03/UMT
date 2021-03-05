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
from transformers import MarianTokenizer, MarianMTModel
from utils.metrics import *

import time

#from comet_ml import OfflineExperiment

from perturbations import *

os.environ["WANDB_API_KEY"] = '829432a2360cc623158d30f47c37fe11d3e12d57'
os.environ["WANDB_MODE"] = "dryrun"

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
    model_ = config['model']
    metric_dict = {'bleu': bleu,
                    'levenshtein': levenshtein}

    metric = metric_dict[config['metric']]

    weights = None
    metric_name = config['metric']
    if config['metric'] == 'bleu':
        metric_name = config['metric']+'-'+str(config['n'])
        weights = tuple([1./config['n']]*config['n'])

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

    samples_dir = os.path.join('Data', model_, ext_language, perturbation.__name__)
    metrics_dir = os.path.join('Metrics', model_, ext_language, perturbation.__name__)

    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    eng_gold_file =  open(os.path.join(samples_dir, 'en.gold'), "r")
    eng_perturb_file = open(os.path.join(samples_dir, 'en.perturb'), "r")

    en_gold_translate_file = open(os.path.join(samples_dir, 'en.gold.translate'), "r")
    en_perturb_translate_file = open(os.path.join(samples_dir, 'en.perturb.translate'), "r")

    o_lang_gold_file = open(os.path.join(samples_dir, ext_language +'.gold'), "r")
    o_lang_perturb_file = open(os.path.join(samples_dir, ext_language + '.perturb'), "r")

    metrics_file = open(os.path.join(metrics_dir, metric_name + '.scores' ), "w")

    eng_gold_sents = eng_gold_file.readlines()
    eng_perturb_sents = eng_perturb_file.readlines()

    o_lang_gold_sents = o_lang_gold_file.readlines()
    o_lang_perturb_sents = o_lang_perturb_file.readlines()

    en_gold_translate_sents = en_gold_translate_file.readlines()
    en_perturb_translate_sents = en_perturb_translate_file.readlines()

    # metrics - bleu2, bleu3, bleu4

    assert len(eng_gold_sents) == len(eng_perturb_sents)

    original_id = 0
    id_ = 0
    lock = FileLock(os.path.join(metrics_dir, 'lock.l'))

    with lock:
        for i in range(0,len(eng_gold_sents)):
            en_gold = json.loads(eng_gold_sents[i])
            en_perturb = json.loads(eng_perturb_sents[i])

            o_lang_gold = json.loads(o_lang_gold_sents[i])
            o_lang_perturb = json.loads(o_lang_pertub_sents[i])

            en_gold_translate = json.loads(en_gold_translate_sents[i])
            en_perturb_translate = json.loads(en_perturb_translate_sents[i])

            original_id = o_lang_gold['original_id']
            id_ = o_lang_gold['id']

            alpha_e = metric([str(_) for _ in nlp(en_gold['text'])], [str(_) for _ in nlp(en_perturb['text'])], weights = weights)
            alpha_o = metric([str(_) for _ in nlp_o(o_lang_gold['text'])], [str(_) for _ in nlp_o(o_lang_perturb['text'])], weights = weights)
            beta = metric([str(_) for _ in nlp_o(en_gold_translate['text'])], [str(_) for _ in nlp_o(o_lang_gold['text'])], weights = weights)

            beta1 = metric([str(_) for _ in nlp_o(en_perturb_translate['text'])], [str(_) for _ in nlp_o(o_lang_gold['text'])], weights = weights)
            beta2 = metric([str(_) for _ in nlp_o(en_perturb_translate['text'])], [str(_) for _ in nlp_o(o_lang_perturb['text'])], weights = weights)

            en_len = len([str(_) for _ in nlp(en_gold['text'])])
            o_len = len([str(_) for _ in nlp(o_lang_gold['text'])])

            d = {
                 "original_id": original_id,
                 "id": id_,
                 "source_len": en_len,
                 "target_len": o_len,
                 "alpha_e": alpha_e,
                 "alpha_o": alpha_o,
                 "beta": beta,
                 "beta_1": beta1,
                 "beta_2": beta2
                 }

            d_f = json.dumps(d)
            metrics_file.write(d_f + '\n')

    metrics_file.close()
    eng_gold_file.close()
    eng_perturb_file.close()

    en_gold_translate_file.close()
    en_perturb_translate_file.close()

    o_lang_gold_file.close()
    o_lang_perturb_file.close()


if __name__ == '__main__':

    PARAM_GRID = list(product(
    ['Helsinki-opus'], #model
    ['de','fr','ru','ja'], #languages
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
        config['batch_size'] = 128

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
        slurm_mem="16G",#16G
        slurm_array_parallelism=100,
    )
    job = executor.map_array(HyperEvaluate,h_param_list)
    print('Jobs submitted!')
