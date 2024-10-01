import os
import numpy as np

from argparse import ArgumentParser, Namespace
import json
import yaml
from pytorch_lightning import Trainer
#from misc.shared import DATA_DIR

def get_hparams():
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", default='./data/motions_processed')
    parser.add_argument("--hparams_file", default='./configs/train.yaml')
    parser = Trainer.add_argparse_args(parser)
    default_params = parser.parse_args()

    conf_name = os.path.basename(default_params.hparams_file)
    if default_params.hparams_file.endswith(".yaml"):
        hparams_json = yaml.full_load(open(default_params.hparams_file))

    params = vars(default_params)
    params.update(hparams_json)
    params.update(vars(default_params))

    hparams = Namespace(**params)

    return hparams, conf_name
