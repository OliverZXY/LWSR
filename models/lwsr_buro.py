from utils.buffer import BURO_Buffer as Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
from einops import rearrange
import math
import numpy as np
import bisect
from utils.loss import RetrievalLoss
import copy
import time


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Slide Retrieval with BuRo')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


def clear_gpu(device):
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class LWSR_BURO(ContinualModel):
    NAME = 'lwsr_buro'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(LWSR_BURO, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device, n_tasks=4)
        self.checkpoint = None

    def observe(self, inputs, labels, task):
        criterion = RetrievalLoss()
        if task == 0:
            self.opt.zero_grad()
            results_dict = self.net(data=inputs)
            criterion.set(pred=results_dict['logits'], label=labels, features=results_dict['cls_token'], task=task,
                          is_ours=True)
            loss = criterion.normal_loss_func()
            loss.backward()
            self.opt.step()
        else:
            self.opt.zero_grad()
            results_dict = self.net(data=inputs)

            model_proxy = type(self.net)()
            model_proxy.load_state_dict(self.checkpoint)
            model_proxy = model_proxy.to(self.device)

            if not self.buffer.is_empty():
                size = self.args.minibatch_size // 2 * 2048
                minibatch_size = self.args.minibatch_size // 2
                t = np.random.randint(task)

                buf_data = self.buffer.get_data(size=size, task=t)

                buf_inputs0 = buf_data[0]
                bf0 = buf_inputs0.shape[0] // 2048

                if bf0 != minibatch_size:
                    bf0_sample_num = bf0 * 2048
                    indices = torch.randperm(buf_inputs0.size(0))
                    selected_indices = indices[:bf0_sample_num]
                    buf_inputs0 = buf_inputs0[selected_indices]

                buf_inputs0 = buf_inputs0.view(bf0, 2048, 512)
                buf_labels0 = torch.full((bf0,), t).to(self.device)

                buf_inputs1 = buf_data[1]
                bf1 = buf_inputs1.shape[0] // 2048

                if bf1 != minibatch_size:
                    bf1_sample_num = bf1 * 2048
                    indices = torch.randperm(buf_inputs1.size(0))
                    selected_indices = indices[:bf1_sample_num]
                    buf_inputs1 = buf_inputs1[selected_indices]

                buf_inputs1 = buf_inputs1.view(bf1, 2048, 512)
                buf_labels1 = torch.full((bf1,), t + 1).to(self.device)

                buf_inputs = torch.cat((buf_inputs0, buf_inputs1), dim=0)
                buf_labels = torch.cat((buf_labels0, buf_labels1))
                buf_outputs = self.net(data=buf_inputs)

                with torch.no_grad():
                    model_proxy.eval()
                    buf_outputs_proxy = model_proxy(data=buf_inputs)
                    previous_dist_mat = self.calc_dist(buf_outputs_proxy['cls_token'])

                current_dist_mat = self.calc_dist(buf_outputs['cls_token'])

                input_loss_features = torch.concat((results_dict['cls_token'], buf_outputs['cls_token']), dim=0)
                input_loss_labels = torch.cat((labels, buf_labels))
                input_loss_logits = torch.concat((results_dict['logits'], buf_outputs['logits']), dim=0)
                criterion.set(pred=input_loss_logits, label=input_loss_labels, features=input_loss_features, task=task,
                              is_ours=True,
                              previous_dist_mat=previous_dist_mat, current_dist_mat=current_dist_mat)
                loss = criterion.all_loss_func()

                del model_proxy
                clear_gpu(self.device)

            loss.backward()
            self.opt.step()

        return loss.item()

    def save_buffer(self, inputs, labels, task):
        self.opt.zero_grad()

        reshaped_inputs = inputs.view(-1, 512)
        repeated_labels = labels.repeat_interleave(inputs.shape[1])
        sample_idx = np.random.randint(reshaped_inputs.shape[0], size=math.ceil(0.8 * reshaped_inputs.shape[0]))
        sample_inputs = reshaped_inputs[sample_idx]
        sample_labels = repeated_labels[sample_idx]

        self.buffer.add_data(examples=sample_inputs, labels=sample_labels)

    def calc_dist(self, buf_inputs):
        differences = buf_inputs.unsqueeze(1) - buf_inputs.unsqueeze(0)
        squared_diff = differences ** 2
        squared_distances = squared_diff.sum(dim=2)
        distances = torch.sqrt(squared_distances + 1e-8)

        return distances

    def end_task(self, dataset):
        self.checkpoint = self.net.state_dict()
