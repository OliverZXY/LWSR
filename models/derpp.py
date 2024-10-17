# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F
import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.loss import RetrievalLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    return parser


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, task):
        criterion = RetrievalLoss()

        self.opt.zero_grad()
        results_dict = self.net(data=inputs)
        criterion.set(pred=results_dict['logits'], label=labels, features=results_dict['cls_token'], task=task, is_ours=False)
        loss = criterion.total_loss_func()

        if not self.buffer.is_empty():
            if self.buffer.__len__() <= self.args.minibatch_size:
                get_size = self.buffer.__len__()
            else:
                get_size = self.args.minibatch_size
            
            buf_inputs, _, buf_logits = self.buffer.get_data(get_size)
            buf_results_dict_alpha = self.net(data=buf_inputs)
            loss_alpha = self.args.alpha * F.mse_loss(buf_results_dict_alpha['logits'], buf_logits)
            loss += torch.tensor(loss_alpha.item())

            buf_inputs, buf_labels, _ = self.buffer.get_data(get_size)
            buf_results_dict_beta = self.net(data=buf_inputs)
            criterion.set(pred=buf_results_dict_beta['logits'], label=buf_labels, features=buf_results_dict_beta['cls_token'], task=task, is_ours=False)
            loss += self.args.beta * criterion.total_loss_func()

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=inputs,
                             labels=labels,
                             logits=results_dict['logits'])

        return loss.item()
