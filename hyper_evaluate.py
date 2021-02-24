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

def HyperEvaluate(config):
    ext_language = config['lang']
    perturbation = config['perturb']
    n_beams = config['num_beams']

    nlp_o = spacy.load(ext_language+"_core_news_sm")
    nlp = spacy.load("en_core_web_sm")

    tokenizer = AutoTokenizer.from_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-en-" + ext_language)
    model = AutoModelForSeq2SeqLM.from_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-en-" + ext_language + "-model")

    tokenizer_inv = AutoTokenizer.from_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-" + ext_language + "-en")
    model_inv = AutoModelForSeq2SeqLM.from_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-" + ext_language + "-en-model")

    translation = pipeline("translation_en_to_"+config['lang'], model=model, tokenizer=tokenizer)
    translation_inv = pipeline("translation_"+config['lang']+"_to_back", model=model_inv, tokenizer=tokenizer_inv)

    perturbations = [treeMirrorPre, treeMirrorPo, treeMirrorIn, verbAtBeginning, verbSwaps, adverbVerbSwap,
      nounVerbSwap, nounVerbMismatched, nounAdjSwap, shuffleHalvesFirst, shuffleHalvesLast,
      reversed, wordShuffle, rotateAroundRoot,functionalShuffle, nounSwaps, conjunctionShuffle]

    assert perturbation in perturbations

    test_file = open('test_data/opus-'+ext_language+'-en.test.txt')

    lines = test_file.readlines()

    sys.argv = sys.argv[:1]

    bleurt_ops = score.create_bleurt_ops('/home/pparth2/scratch/UMT/UMT/Results/cached/bleurt-base-128')

    wandb.init(project="UMT-Analysis", reinit = True)
    wandb.run.name = config['lang'] + '-' + config['perturb'].__name__

    wandb.config.update(config)
    # todo : Save samples in a csv : with metrics, perturbed example and beams
    # todo : ensure the beams have the same seed across runs and so do the perturbation functions. Use the index as seed value.
    fieldnames = ['Gold English'] + ['Beam/'+str(i+1) for i in range(n_beams)] + ["index", "len",
    "M1_bleu2", "M2_bleu2_max", "M2_bleu2_rand", "M1_bleurt", "M2_bleurt_max", "M2_bleurt_rand", "M3_bleu2_max", "M3_bleu2_rand"]

    target_NL =  open(os.path.join('Results','Samples','UMT_'+ config['lang'] + '_' + str(config['num_beams']) + '_' + config['perturb'].__name__ +'.csv'), "w")
    writer_NL = csv.DictWriter(target_NL, fieldnames=fieldnames[:-1])
    lock = FileLock(os.path.join(open(os.path.join('Results','Samples','UMT_'+ config['lang'] + '_' + str(config['num_beams']) + '_' + config['perturb'].__name__ +'.csv.lock'))

    with lock:
        with open(target_NL,'w') as f:
            writer_NL.writeheader()
        target_NL.close()
    for i in range(0,len(lines),4):

        other_lang_gold = lines[i].strip()
        eng_gold = lines[i+1].strip()

        eng_perturbed = config['perturb'](eng_gold)

        other_lang_translated = translation(eng_gold, max_length=400)[0]['translation_text']

        input_ids = tokenizer.encode(eng_perturbed, return_tensors="pt")
        other_lang_translated_p_beams = model.generate(input_ids, max_length=100, num_beams = 50 ,num_return_sequences=n_beams, do_sample=True)

        #other_lang_translated_p = translation(eng_perturbed, max_length=400)[0]['translation_text']

        eng_back_translated = translation_inv(other_lang_translated, max_length=400)[0]['translation_text']

        #BLEU(e', e)
        bleu2_eng_p_eng_gold = bleu_score([[str(_) for _ in nlp(eng_gold)]], [str(_) for _ in nlp(eng_perturbed)], (0.5,0.5))
        #BLEU(e, e_1)

        bleu2_eng_back_translated_eng_gold = bleu_score([[str(_) for _ in nlp(eng_gold)]], [str(_) for _ in nlp(eng_back_translated)], (0.5,0.5))

        bleu2_other_translated_other_gold = bleu_score([[str(_) for _ in nlp_o(other_lang_gold)]], [str(_) for _ in nlp_o(other_lang_translated)])
        M1_bleu2 = bleu2_eng_p_eng_gold / bleu2_eng_back_translated_eng_gold

        bleu2_eng_back_p_eng_perturbed_beam = []

        M2_bleu2 = []
        M3_bleu2 = []

        ###
        # BLEURT

        bleurt_out = bleurt_ops([eng_gold], [eng_perturbed])
        assert bleurt_out["predictions"].shape == (1,)
        bleurt_eng_p_eng_gold = 3.5 + float(bleurt_out["predictions"])

        bleurt_out = bleurt_ops([eng_gold], [eng_back_translated])
        assert bleurt_out["predictions"].shape == (1,)
        bleurt_eng_back_translated_eng_gold = 3.5 + float(bleurt_out["predictions"])

        M1_bleurt = bleurt_eng_p_eng_gold / bleurt_eng_back_translated_eng_gold

        bleurt_eng_back_p_eng_perturbed_beam = []

        M2_bleurt = []
        ###

        ### add saving sample with index
        eng_back_translated_per = {}
        for k, other_lang_translated_p in enumerate(other_lang_translated_p_beams):

            eng_back_translated_p = translation_inv(tokenizer.decode(other_lang_translated_p), max_length=400)[0]['translation_text']
            eng_back_translated_per.update({k:eng_back_translated_p})
            # BLEU-2
            bleu2_eng_back_p_eng_perturbed_beam.append(bleu_score([[str(_) for _ in nlp(eng_perturbed)]], [str(_) for _ in nlp(eng_back_translated_p)], (0.5,0.5)))

            M2_bleu2.append(bleu2_eng_back_p_eng_perturbed_beam[-1]/bleu2_eng_back_translated_eng_gold)

            bleu2_other_translated_p_other_gold = bleu_score([[str(_) for _ in nlp_o(tokenizer.decode(other_lang_translated_p))]],[str(_) for _ in nlp_o(other_lang_gold)])

            M3_bleu2.append(bleu2_other_translated_p_other_gold/bleu2_other_translated_other_gold)

            bleurt_out = bleurt_ops([eng_perturbed], [eng_back_translated_p])
            assert bleurt_out["predictions"].shape == (1,)
            bleurt_eng_back_p_eng_perturbed_beam.append(3.5 + float(bleurt_out["predictions"]))

            M2_bleu2.append(bleurt_eng_back_p_eng_perturbed_beam[-1]/bleu2_eng_back_translated_eng_gold)

        M3_bleu2_max = max(M3_bleu2)
        M3_bleu2_rand = sum(M3_bleu2)/float(len(M3_bleu2))

        M2_bleu2_max = max(M2_bleu2)
        M2_bleu2_rand = sum(M2_bleu2)/float(len(M2_bleu2))

        M2_bleurt_max = max(M2_bleurt)
        M2_bleurt_rand = sum(M2_bleurt)/float(len(M2_bleurt))

        log_dict = {
        "index": i,
        "len": len(nlp(eng_gold)),
        "M1_bleu2": M1_bleu2,
        "M2_bleu2_max": M2_bleu2_max,
        "M2_bleu2_rand": M2_bleu2_rand,
        "M1_bleurt": M1_bleurt,
        "M2_bleurt_max": M2_bleurt_max,
        "M2_bleurt_rand": M2_bleurt_rand,
        "M3_bleu2_max": M3_bleu2_max,
        "M3_bleu2_rand": M3_bleu2_rand
        }

        wandb.log(log_dict)

        for k, v in eng_back_translated_per:
            log_dict.update({'Beam/'+str(k+1): v})
        lock = FileLock(os.path.join(open(os.path.join('Results','Samples','UMT_'+ config['lang'] + '_' + str(config['num_beams']) + '_' + config['perturb'].__name__ +'.csv.lock'))
        with lock:
            with open(target_NL,'a') as f:
                writer_NL.writerow(log_dict)



if __name__ == '__main__':

    PARAM_GRID = list(product(
    ['de'],#'fr','ru','ja'], #languages
    [treeMirrorPre, treeMirrorPo, treeMirrorIn, verbAtBeginning, verbSwaps, adverbVerbSwap,
      nounVerbSwap, nounVerbMismatched, nounAdjSwap, shuffleHalvesFirst, shuffleHalvesLast,
      reversed, wordShuffle, rotateAroundRoot,functionalShuffle, nounSwaps, conjunctionShuffle],
    [1,5,10] # beam-width
    )
    )

    h_param_list = []

    for param_ix in range(len(PARAM_GRID)):

        params = PARAM_GRID[param_ix]


        lang, pert, b = params
        config = {}
        config['lang'] = lang
        config['perturb'] = pert
        config['num_beams'] = b

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


# config = AutoConfig.from_pretrained("Helsinki-NLP/opus-mt-en-jap")
# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-jap")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-jap")
#
# tokenizer.save_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-en-ja")
# config.save_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-en-ja")
# model.save_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-en-ja-model")
#
# config = AutoConfig.from_pretrained("Helsinki-NLP/opus-mt-jap-en")
# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-jap-en")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-jap-en")
#
# tokenizer.save_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-ja-en")
# model.save_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-ja-en-model")
# config.save_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-ja-en")
