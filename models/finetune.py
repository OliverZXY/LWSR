# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from utils.loss import RetrievalLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Finetune')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Finetune(ContinualModel):
    NAME = 'finetune'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Finetune, self).__init__(backbone, loss, args, transform)

    def observe(self, inputs, labels, task):
        criterion = RetrievalLoss()

        self.opt.zero_grad()

        results_dict = self.net(data=inputs)
        criterion.set(pred=results_dict['logits'], label=labels, features=results_dict['cls_token'], task=task, is_ours=False)
        
        loss = criterion.total_loss_func()
        loss.backward()
        self.opt.step()

        return loss.item()
