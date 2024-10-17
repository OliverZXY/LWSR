import random
import torch
import numpy as np

global DEVICE
global SAMPLING_NUM
global PAIR_LOSS_WEIGHT
global CE_LOSS_WEIGHT
global DC_LOSS_WEIGHT

global_config_data = {
    'device': None,
    'sampling_num': float('-inf'),
    'pair_loss_weight': float('-inf'),
    'ce_loss_weight': float('-inf'),
    'dc_loss_weight': float('-inf'),
}


def base_path() -> str:
    return './data/'


def base_path_dataset() -> str:
    return '/tmp/mammoth_datasets/'


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        print('Could not set cuda seed.')
        pass
