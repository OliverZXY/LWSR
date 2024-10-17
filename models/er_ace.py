# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.loss import RetrievalLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErACE(ContinualModel):
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErACE, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

    def end_task(self, dataset):
        self.task += 1

    def observe(self, inputs, labels, task):
        criterion = RetrievalLoss()

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        results_dict = self.net(data=inputs)
        logits = results_dict['logits']
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
        

        criterion.set(pred=logits, label=labels, features=results_dict['cls_token'], task=task, is_ours=False)
        loss = criterion.total_loss_func()
        loss_re = torch.tensor(0.)

        if self.task > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_results_dict = self.net(data=buf_inputs)
            criterion.set(pred=buf_results_dict['logits'], label=buf_labels, features=buf_results_dict['cls_token'], task=task, is_ours=False)
            loss_re = criterion.total_loss_func()

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=inputs,
                             labels=labels)

        return loss.item()
