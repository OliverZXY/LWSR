from utils.buffer import Retrieval_Buffer as Buffer
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import math
import numpy as np
from utils.loss import RetrievalLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Lifelong Whole Slide Retrieval without Distance Consistency Rehearsal')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class LWSR_WO_DCR(ContinualModel):
    NAME = 'lwsr_wo_dcr'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(LWSR_WO_DCR, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device, n_tasks=4)

    def observe(self, inputs, labels, task):
        criterion = RetrievalLoss()
        if task == 0:
            self.opt.zero_grad()
            results_dict = self.net(data=inputs)
            criterion.set(pred=results_dict['logits'], label=labels, features=results_dict['cls_token'], task=task, is_ours=True)
            loss = criterion.normal_loss_func()
            loss.backward()
            self.opt.step()
        else:
            self.opt.zero_grad()
            results_dict = self.net(data=inputs)

            if not self.buffer.is_empty():
                size = self.args.minibatch_size

                buf_data = self.buffer.get_data(size=size)
                return_index_list = buf_data[0].tolist()

                buf_inputs = buf_data[1]
                buf_labels = buf_data[2]

                buf_outputs = self.net(data=buf_inputs)

                input_loss_features = torch.concat((results_dict['cls_token'], buf_outputs['cls_token']), dim=0)
                input_loss_labels = torch.cat((labels, buf_labels))
                input_loss_logits = torch.concat((results_dict['logits'], buf_outputs['logits']), dim=0)
                criterion.set(pred=input_loss_logits, label=input_loss_labels, features=input_loss_features, task=task, is_ours=True,
                              return_idx=return_index_list)
                loss = criterion.normal_loss_func()

            loss.backward()
            self.opt.step()

        return loss.item()

    def save_buffer(self, inputs, labels, task):
        self.opt.zero_grad()
        sample_idx = np.random.randint(inputs.shape[0], size=math.ceil(0.5 * inputs.shape[0]))
        sample_inputs = inputs[sample_idx]
        sample_labels = labels[sample_idx]

        self.buffer.add_data(examples=sample_inputs, labels=sample_labels)
