# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
CURRENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from argparse import Namespace
from datasets.utils.continual_dataset import ContinualDataset
from datasets.seq_tcga import Sequential_Generic_TCGA_Dataset


def get_all_models():
    return [model.split('.')[0] for model in os.listdir(os.path.join(CURRENT_PATH, 'datasets'))
            if not model.find('__') > -1 and 'py' in model]


NAMES = {
    'seq_tcga': Sequential_Generic_TCGA_Dataset
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES
    return NAMES[args.dataset](args)
