# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
CURRENT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from models.derpp import Derpp
from models.lwf import Lwf
from models.ewc_on import EwcOn
from models.agem import AGem
from models.er_ace import ErACE
from models.joint import Joint
from models.finetune import Finetune
from models.lwsr import LWSR
from models.lwsr_wo_dcr import LWSR_WO_DCR
from models.lwsr_buro import LWSR_BURO


def get_all_models():
    return [model.split('.')[0] for model in os.listdir(os.path.join(CURRENT_PATH, 'models'))
            if not model.find('__') > -1 and 'py' in model]


names = {
    'derpp': Derpp,
    'lwf': Lwf,
    'ewc_on': EwcOn,
    'agem': AGem,
    'er_ace': ErACE,
    'joint': Joint,
    'finetune': Finetune,
    'lwsr': LWSR,
    'lwsr_wo_dcr': LWSR_WO_DCR,
    'lwsr_buro': LWSR_BURO
}


def get_model(args, backbone, loss, transform):
    return names[args.model](backbone, loss, args, transform)
