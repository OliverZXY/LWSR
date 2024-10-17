from __future__ import print_function, division
from asyncio import base_tasks
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
import collections
from itertools import islice
import bisect
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from backbone.vit import ViT
from argparse import ArgumentParser
from utils.conf import global_config_data


class Generic_TCGA_Dataset():
    def __init__(self,
                 label_path,
                 label_dict,
                 ):
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.label_path = label_path

    def generate_dataset(self):
        label_dict = {v: k for k, v in self.label_dict.items()}

        label_train_path = os.path.join(self.label_path, f'train.csv')
        label_val_path = os.path.join(self.label_path, f'val.csv')
        label_test_path = os.path.join(self.label_path, f'test.csv')

        train_csv = pd.read_csv(label_train_path)
        val_csv = pd.read_csv(label_val_path)
        test_csv = pd.read_csv(label_test_path)

        train_df = train_csv[train_csv['disease_id'].isin(label_dict.keys())]
        val_df = val_csv[val_csv['disease_id'].isin(label_dict.keys())]
        test_df = test_csv[test_csv['disease_id'].isin(label_dict.keys())]

        train_slide_ids = train_df['slide_id'].values
        train_labels = train_df['disease_id'].values
        train_uuids = train_df['uuid'].values
        val_slide_ids = val_df['slide_id'].values
        val_labels = val_df['disease_id'].values
        val_uuids = val_df['uuid'].values
        test_slide_ids = test_df['slide_id'].values
        test_labels = test_df['disease_id'].values
        test_uuids = test_df['uuid'].values

        train_ds = TCGA_Dataset(slide_ids=train_slide_ids, labels=train_labels, uuids=train_uuids)
        val_ds = TCGA_Dataset(slide_ids=val_slide_ids, labels=val_labels, uuids=val_uuids)
        test_ds = TCGA_Dataset(slide_ids=test_slide_ids, labels=test_labels, uuids=test_uuids)

        return train_ds, val_ds, test_ds


class TCGA_Dataset(Dataset):
    def __init__(self, slide_ids, labels, uuids):
        self.slide_ids = slide_ids
        self.labels = labels
        self.uuids = uuids
        self.feature_sample_num = global_config_data['sampling_num']

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]
        uuid = self.uuids[idx]
        feature = torch.load(slide_id)['feats']
        num_feat = min(feature.shape[0], self.feature_sample_num)
        feature = self.pack_data(feature, num_feat)

        if hasattr(self, 'logits'):
            return feature, label, self.logits[idx], uuid

        return feature, label, uuid

    def pack_data(self, feat, num_feat):
        wsi_feat = np.zeros((self.feature_sample_num, feat.shape[-1]))
        wsi_feat[:num_feat] = np.squeeze(feat)

        return wsi_feat

    def sample_feature(self, feature):
        indices = torch.randperm(feature.shape[0])[:self.feature_sample_num]
        sampled_tensor = feature[indices]

        if sampled_tensor.shape[0] < self.feature_sample_num:
            num_padding = self.feature_sample_num - sampled_tensor.shape[0]
            padding_tensor = np.zeros((num_padding, feature.shape[1]))
            sampled_tensor = np.concatenate((sampled_tensor, padding_tensor), axis=0)

        return sampled_tensor


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class Sequential_Generic_TCGA_Dataset(ContinualDataset):
    NAME = 'seq-tcga'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 4
    TRANSFORM = None

    datasets = [
        Generic_TCGA_Dataset(
            label_path='/media/disk1/zhuxinyu/labels/transmil-8s/',
            label_dict={'COAD': 6, 'STAD': 7},
        ),
        Generic_TCGA_Dataset(
            label_path='/media/disk1/zhuxinyu/labels/transmil-8s/',
            label_dict={'IDC': 4, 'ILC': 5},
        ),
        Generic_TCGA_Dataset(
            label_path='/media/disk1/zhuxinyu/labels/transmil-8s/',
            label_dict={'KIRC': 2, 'KIRP': 3},
        ),
        Generic_TCGA_Dataset(
            label_path='/media/disk1/zhuxinyu/labels/transmil-8s/',
            label_dict={'LUSC': 0, 'LUAD': 1},
        ),
    ]
    datasets.reverse()

    def get_data_loaders(self):
        dataset = self.datasets[self.i]
        train_dataset, val_dataset, test_dataset = dataset.generate_dataset()
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        retrieval_train_loader = DataLoader(train_dataset,
                                            batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        retrieval_val_loader = DataLoader(val_dataset,
                                          batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        retrieval_test_loader = DataLoader(test_dataset,
                                           batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        self.i += 1
        self.train_datasets.append(train_dataset)
        self.test_datasets.append(test_dataset)

        self.train_loaders.append(retrieval_train_loader)
        self.test_loaders.append(retrieval_test_loader)
        self.val_loaders.append(retrieval_val_loader)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        return train_loader, val_loader, test_loader

    def get_joint_data_loaders(self):
        train_datasets, val_datasets, test_datasets = [], [], []

        for n in range(self.N_TASKS):
            dataset = self.datasets[n]
            train_dataset, val_dataset, test_dataset = dataset.generate_dataset()

            single_test_loader = DataLoader(test_dataset,
                                            batch_size=self.args.batch_size, shuffle=False, num_workers=4)
            self.test_loaders.append(single_test_loader)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)

        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        test_dataset = ConcatDataset(test_datasets)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.joint_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.args.joint_batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.args.joint_batch_size, shuffle=False, num_workers=4)

        self.i = self.N_TASKS
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        return train_loader, val_loader, test_loader

    def get_train_data_loader(self):
        train_datasets = []

        for n in range(self.N_TASKS):
            dataset = self.datasets[n]
            train_dataset, val_dataset, test_dataset = dataset.generate_dataset()
            train_datasets.append(train_dataset)

        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=False, num_workers=4)

        return train_loader

    @staticmethod
    def get_backbone(args):
        return ViT(n_classes=args.n_classes)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return torch.optim.lr_scheduler.StepLR(model.opt, step_size=500, gamma=0.7)

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 6

    @staticmethod
    def get_minibatch_size():
        return 40
