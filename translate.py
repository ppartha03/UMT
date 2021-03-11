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

from tqdm.auto import tqdm
import time

# from comet_ml import OfflineExperiment

from perturbations import *

# commandline arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data > Model > Lang > Perturbation > source.perturb -
#                                    > target.perturb -
#                                    > source.gold -
#                                    > target.gold -
#                                    > source.gold.to.target
#                                    > source.perturb.to.target


def HyperEvaluate(config):
    ext_language = config["lang"]
    perturbation = config["perturb"]
    model_ = config["model"]
    batch_size = config["batch_size"]

    try:
        nlp_o = spacy.load(ext_language + "_core_news_sm")
    except:
        nlp_o = spacy.load(ext_language + "_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

    perturbations = [
        treeMirrorPre,
        treeMirrorPo,
        treeMirrorIn,
        verbAtBeginning,
        verbSwaps,
        adverbVerbSwap,
        nounVerbSwap,
        nounVerbMismatched,
        nounAdjSwap,
        shuffleHalvesFirst,
        shuffleHalvesLast,
        reversed,
        wordShuffle,
        rotateAroundRoot,
        functionalShuffle,
        nounSwaps,
        conjunctionShuffle,
    ]

    assert perturbation in perturbations

    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-" + ext_language).to(
        device
    )
    tokenizer = MarianTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-en-" + ext_language
    )

    # todo : Save samples in a csv : with metrics, perturbed example and beams
    # todo : ensure the beams have the same seed across runs and so do the perturbation functions. Use the index as seed value.

    target_dir = os.path.join("Data", model_, ext_language, perturbation.__name__)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    eng_gold_file = open(os.path.join(target_dir, "en.gold"), "r")
    eng_perturb_file = open(os.path.join(target_dir, "en.perturb"), "r")

    o_lang_gold_file = open(os.path.join(target_dir, "en.gold.translate"), "w")
    o_lang_perturb_file = open(os.path.join(target_dir, "en.perturb.translate"), "w")

    eng_gold_sents = eng_gold_file.readlines()
    eng_perturb_sents = eng_perturb_file.readlines()

    assert len(eng_gold_sents) == len(eng_perturb_sents)

    original_id = 0
    id_ = 0

    for i in tqdm(range(0, len(eng_gold_sents), batch_size)):
        e_gold_batch = [json.loads(_) for _ in eng_gold_sents[i : i + batch_size]]
        e_perturb_batch = [json.loads(_) for _ in eng_perturb_sents[i : i + batch_size]]

        eng_gold_batch = [_["text"] for _ in e_gold_batch]
        eng_perturb_batch = [_["text"] for _ in e_perturb_batch]

        batch_gold = tokenizer.prepare_seq2seq_batch(
            src_texts=eng_gold_batch, return_tensors="pt"
        ).to(device)
        batch_perturb = tokenizer.prepare_seq2seq_batch(
            src_texts=eng_perturb_batch, return_tensors="pt"
        ).to(device)

        gen = model.generate(**batch_gold)
        translate_gold = tokenizer.batch_decode(gen, skip_special_tokens=True)

        gen = model.generate(**batch_perturb)
        translate_perturb = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for k in range(len(e_gold_batch)):
            original_id = e_gold_batch[k]["original_id"]
            id_ = e_gold_batch[k]["id"]

            d = {
                "original_id": original_id,
                "id": id_,
                "text": translate_perturb[k],
            }
            d_f = json.dumps(d)
            o_lang_perturb_file.write(d_f + "\n")

            d = {"original_id": original_id, "id": id_, "text": translate_gold[k]}
            d_f = json.dumps(d)
            o_lang_gold_file.write(d_f + "\n")

    o_lang_gold_file.close()
    eng_gold_file.close()

    eng_perturb_file.close()
    o_lang_perturb_file.close()


if __name__ == "__main__":

    debug = False

    PARAM_GRID = list(
        product(
            ["wmt19"],  # model wmt18 ['de', 'ru', 'zh']
            ["de"],  # ['de','fr','ru','ja'], #languages
            [
                treeMirrorPre,
                treeMirrorPo,
                treeMirrorIn,
                verbSwaps,
                adverbVerbSwap,
                verbAtBeginning,
                nounVerbSwap,
                nounVerbMismatched,
                nounAdjSwap,
                shuffleHalvesFirst,
                shuffleHalvesLast,
                reversed,
                wordShuffle,
                rotateAroundRoot,
                functionalShuffle,
                nounSwaps,
                conjunctionShuffle,
            ],
        )
    )

    h_param_list = []

    for param_ix in range(len(PARAM_GRID)):

        params = PARAM_GRID[param_ix]

        model, lang, pert = params
        config = {}
        config["lang"] = lang
        config["perturb"] = pert
        config["model"] = model
        config["batch_size"] = 32

        h_param_list.append(config)

    if debug:
        for hp in h_param_list:
            print(hp)
            HyperEvaluate(hp)

    else:
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
            timeout_min=360,
            gpus_per_node=num_gpus,
            slurm_partition="dev",
            tasks_per_node=num_gpus,
            cpus_per_task=workers_per_gpu,
            slurm_mem="",  # 16G
            slurm_array_parallelism=50,
        )
        job = executor.map_array(HyperEvaluate, h_param_list)
        print("Jobs submitted!")
