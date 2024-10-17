# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import sys
from argparse import Namespace
from typing import Tuple
import time

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

import numpy as np
np.set_printoptions(threshold=np.inf)

from utils.loggers import *
from utils.status import ProgressBar
from utils.metrics import rank_retrievals, calc_all_rank_metrics

try:
    import wandb
except ImportError:
    wandb = None


def list2str(input_list):
    output_str = '+'.join(map(str, input_list))

    return output_str


def list_to_str_expression(lst):
    sum_expressions = ["+".join(str(x) for x in sublist) for sublist in lst]
    result_expression = "[" + "]+[".join(sum_expressions) + "]"
    return result_expression


def get_hash_code(data_loader, device, model):
    model.eval()

    feature_set = None
    uuids_set = []

    with torch.no_grad():
        for i, loader_data in enumerate(data_loader):
            if hasattr(data_loader.dataset, 'logits'):
                data, label, _, uuids = loader_data
            else:
                data, label, uuids = loader_data
            
            uuids_set += list(uuids)

            data = data.to(device)

            results_dict = model(data=data)
            hash_origin = results_dict['cls_token'].cpu().data
            hash_label = torch.t(torch.unsqueeze(label, 0))

            result = torch.cat((hash_origin, hash_label), dim=1)

            if feature_set is None:
                feature_set = result
            else:
                feature_set = torch.cat((feature_set, result), dim=0)
        
        retrieval_data = feature_set.numpy()
        feature = retrieval_data[:, 0:-1]
        label = retrieval_data[:, -1].astype(int)

        return feature, label, uuids_set


def calculate_distance_matrix(X, Y):
    dists = -2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
    return dists


def compute_AP(retrieved_labels, true_label):
    relevant = np.array(retrieved_labels) == true_label
    cumsum = np.cumsum(relevant)
    precision_at_k = cumsum / (np.arange(len(relevant)) + 1)
    AP = np.sum(precision_at_k * relevant) / np.sum(relevant)
    return AP


def mAP_R_at_k_and_P_at_t(database_labels, retrieved_indices, query_labels, k=3, t=5):
    num_queries = query_labels.shape[0]
    AP_scores = []
    correct_at_k = 0
    precision_at_t_sum = 0
    
    for i in range(num_queries):
        true_label = query_labels[i]        
        retrieved_k = [database_labels[idx] for idx in retrieved_indices[i, :k]]
        retrieved_t = [database_labels[idx] for idx in retrieved_indices[i, :t]]  # 取前5个检索结果
        
        if true_label in retrieved_k:
            correct_at_k += 1
        
        correct_in_top_t = sum([1 for label in retrieved_t if label == true_label])
        precision_at_t = correct_in_top_t / 5.0
        precision_at_t_sum += precision_at_t
            
        all_retrieved = [database_labels[idx] for idx in retrieved_indices[i]]
        AP = compute_AP(all_retrieved, true_label)
        AP_scores.append(AP)
    
    mAP = np.mean(AP_scores)
    R_at_k = correct_at_k / num_queries
    P_at_t = precision_at_t_sum / num_queries
    
    return mAP, R_at_k, P_at_t


def calc_retrieval_metrics(train_feature, train_label, test_feature, test_label):
    dists = calculate_distance_matrix(test_feature, train_feature)

    N = train_feature.shape[0]
    retrieved_indices = np.argsort(dists, axis=1)[:, :N]

    mAP, R_at_3, P_at_5 = mAP_R_at_k_and_P_at_t(train_label, retrieved_indices, test_label, k=3, t=5)

    return mAP, R_at_3, P_at_5


def calc_retrieval_metrics_cancer(n_classes, train_feature, train_label_origin, test_feature, test_label_origin):
    train_label = train_label_origin.copy()
    test_label = test_label_origin.copy()

    train_label[(train_label == 0) | (train_label == 1)] = 0
    train_label[(train_label == 2) | (train_label == 3)] = 1
    train_label[(train_label == 4) | (train_label == 5)] = 2
    train_label[(train_label == 6) | (train_label == 7)] = 3

    test_label[(test_label == 0) | (test_label == 1)] = 0
    test_label[(test_label == 2) | (test_label == 3)] = 1
    test_label[(test_label == 4) | (test_label == 5)] = 2
    test_label[(test_label == 6) | (test_label == 7)] = 3

    mAP, R_at_3, P_at_5 = calc_retrieval_metrics(train_feature, train_label, test_feature, test_label)

    return mAP, R_at_3, P_at_5


def select_uuids(uuids, ranks):
    uuid_array = np.array(uuids)
    selected_uuids = uuid_array[ranks]

    return selected_uuids


def evaluate_retrieval(args, model: ContinualModel, dataset: ContinualDataset, task: int):
    model.net.eval()
    mAPs, R3s, P5s, mAPs_cancer, R3s_cancer, P5s_cancer = [], [], [], [], [], []
    single_mAPs, single_R3s, single_P5s, single_mAPs_cancer, single_R3s_cancer, single_P5s_cancer = [], [], [], [], [], []
    marc_list, src_list, krc_list = [], [], []
    initial_uuids_list, updated_uuids_list = [], []
    with torch.no_grad():
        train_loader_list = dataset.train_loaders

        all_train_feature = None
        all_train_label = None
        all_train_uuid = []

        all_test_feature = None
        all_test_label = None
        all_test_uuid = []

        test_feature_list = []
        test_label_list = []
        test_uuid_list = []

        for k, test_loader in enumerate(dataset.test_loaders):
            curr_train_loader = train_loader_list[k]
            train_feature, train_label, train_uuid = get_hash_code(curr_train_loader, model.device, model.net)
            test_feature, test_label, test_uuid = get_hash_code(test_loader, model.device, model.net)

            test_feature_list.append(test_feature)
            test_label_list.append(test_label)
            test_uuid_list.append(test_uuid)

            if k==0:
                all_train_feature = train_feature
                all_train_label = train_label
                all_train_uuid = train_uuid
                all_test_feature = test_feature
                all_test_label = test_label
                all_test_uuid = test_uuid
            else:
                all_train_feature = np.concatenate((all_train_feature, train_feature), axis=0)
                all_train_label = np.concatenate((all_train_label, train_label), axis=0)
                all_train_uuid += train_uuid
                all_test_feature = np.concatenate((all_test_feature, test_feature), axis=0)
                all_test_label = np.concatenate((all_test_label, test_label), axis=0)
                all_test_uuid += test_uuid

            if k == task:
                initial_ranks = rank_retrievals(test_feature, all_train_feature)
                initial_uuids = select_uuids(all_train_uuid, initial_ranks)
                dataset.initial_ranks_list.append(initial_uuids)
            
            if task != 0 and k != task:
                marc, src, krc, updated_uuids = calc_all_rank_metrics(test_feature, all_train_feature, all_train_uuid, dataset.initial_ranks_list[k])
                initial_uuids_list.append(dataset.initial_ranks_list[k])
                updated_uuids_list.append(updated_uuids)
                marc_list.append(marc)
                src_list.append(src)
                krc_list.append(krc)

            test_mAP, test_R3, test_P5 = calc_retrieval_metrics(all_train_feature, all_train_label, test_feature, test_label)
            test_mAP_cancer, test_R3_cancer, test_P5_cancer = calc_retrieval_metrics_cancer(args.n_classes, all_train_feature, all_train_label, test_feature, test_label)

            mAPs.append(test_mAP)
            R3s.append(test_R3)
            P5s.append(test_P5)
            mAPs_cancer.append(test_mAP_cancer)
            R3s_cancer.append(test_R3_cancer)
            P5s_cancer.append(test_P5_cancer)

        all_test_mAP, all_test_R3, all_test_P5 = calc_retrieval_metrics(all_train_feature, all_train_label, all_test_feature, all_test_label)
        all_test_mAP_cancer, all_test_R3_cancer, all_test_P5_cancer = calc_retrieval_metrics_cancer(args.n_classes, all_train_feature, all_train_label, all_test_feature, all_test_label)

        for i, single_test_feature in enumerate(test_feature_list):
            single_test_mAP, single_test_R3, single_test_P5 = calc_retrieval_metrics(all_train_feature, all_train_label, single_test_feature, test_label_list[i])
            single_test_mAP_cancer, single_test_R3_cancer, single_test_P5_cancer = calc_retrieval_metrics_cancer(args.n_classes, all_train_feature, all_train_label, single_test_feature, test_label_list[i])
            single_mAPs.append(single_test_mAP)
            single_R3s.append(single_test_R3)
            single_P5s.append(single_test_P5)

            single_mAPs_cancer.append(single_test_mAP_cancer)
            single_R3s_cancer.append(single_test_R3_cancer)
            single_P5s_cancer.append(single_test_P5_cancer)

        if task != 0:
            dataset.all_marc_list.append(marc_list)
            dataset.all_src_list.append(src_list)
            dataset.all_krc_list.append(krc_list)

        marc_mean, src_mean, krc_mean = 0, 0, 0

        if task == dataset.N_TASKS-1:
            marc_mean = average_sum_of_lists(dataset.all_marc_list)
            src_mean = average_sum_of_lists(dataset.all_src_list)
            krc_mean = average_sum_of_lists(dataset.all_krc_list)

        print("single mAP list:", single_mAPs)
        print("single R@3 list:", single_R3s)
        print("single P@5 list:", single_P5s)
        print("single cancer mAP list:", single_mAPs_cancer)
        print("single cancer R@3 list:", single_R3s_cancer)
        print("single cancer P@5 list:", single_P5s_cancer)
        print("mAP list:", mAPs)
        print("R@3 list:", R3s)
        print("P@5 list:", P5s)
        print("cancer mAP list:", mAPs_cancer)
        print("cancer R@3 list:", R3s_cancer)
        print("cancer P@5 list:", P5s_cancer)
        print("all mAP:", all_test_mAP)
        print("all R@3:", all_test_R3)
        print("all P@5:", all_test_P5)
        print("all cancer mAP:", all_test_mAP_cancer)
        print("all cancer R@3:", all_test_R3_cancer)
        print("all cancer P@5:", all_test_P5_cancer)
        print("marc list:", dataset.all_marc_list)
        print("src list:", dataset.all_src_list)
        print("krc list:", dataset.all_krc_list)
        print("marc:", marc_mean)
        print("src:", src_mean)
        print("krc:", krc_mean)
    
    return single_mAPs, single_R3s, single_P5s, single_mAPs_cancer, single_R3s_cancer, single_P5s_cancer, mAPs, R3s, P5s, mAPs_cancer, R3s_cancer, P5s_cancer,\
    all_test_mAP, all_test_R3, all_test_P5, all_test_mAP_cancer, all_test_R3_cancer, all_test_P5_cancer,\
    dataset.all_marc_list, dataset.all_src_list, dataset.all_krc_list, marc_mean, src_mean, krc_mean, \
    train_feature, train_label, train_uuid, test_feature_list, test_label_list, test_uuid_list, \
    all_test_feature, all_test_label, all_test_uuid, initial_uuids_list, updated_uuids_list


def evaluate_retrieval_epoch(args, model: ContinualModel, dataset: ContinualDataset, task: int, epoch):
    model.net.eval()
    mAPs, R3s, P5s, mAPs_cancer, R3s_cancer, P5s_cancer = [], [], [], [], [], []
    single_mAPs, single_R3s, single_P5s, single_mAPs_cancer, single_R3s_cancer, single_P5s_cancer = [], [], [], [], [], []
    marc_list, src_list, krc_list = [], [], []

    with torch.no_grad():
        train_loader_list = dataset.train_loaders

        all_train_feature = None
        all_train_label = None
        all_train_uuid = []

        all_test_feature = None
        all_test_label = None
        all_test_uuid = []

        test_feature_list = []
        test_label_list = []
        test_uuid_list = []

        for k, test_loader in enumerate(dataset.test_loaders):
            curr_train_loader = train_loader_list[k]
            train_feature, train_label, train_uuid = get_hash_code(curr_train_loader, model.device, model.net)
            test_feature, test_label, test_uuid = get_hash_code(test_loader, model.device, model.net)

            test_feature_list.append(test_feature)
            test_label_list.append(test_label)
            test_uuid_list.append(test_uuid)

            if k==0:
                all_train_feature = train_feature
                all_train_label = train_label
                all_train_uuid = train_uuid
                all_test_feature = test_feature
                all_test_label = test_label
                all_test_uuid = test_uuid
            else:
                all_train_feature = np.concatenate((all_train_feature, train_feature), axis=0)
                all_train_label = np.concatenate((all_train_label, train_label), axis=0)
                all_train_uuid += train_uuid
                all_test_feature = np.concatenate((all_test_feature, test_feature), axis=0)
                all_test_label = np.concatenate((all_test_label, test_label), axis=0)
                all_test_uuid += test_uuid
            
            if task != 0 and k != task:
                marc, src, krc, updated_uuids = calc_all_rank_metrics(test_feature, all_train_feature, all_train_uuid, dataset.initial_ranks_list[k])
                marc_list.append(marc)
                src_list.append(src)
                krc_list.append(krc)

            test_mAP, test_R3, test_P5 = calc_retrieval_metrics(all_train_feature, all_train_label, test_feature, test_label)
            test_mAP_cancer, test_R3_cancer, test_P5_cancer = calc_retrieval_metrics_cancer(args.n_classes, all_train_feature, all_train_label, test_feature, test_label)

            mAPs.append(test_mAP)
            R3s.append(test_R3)
            P5s.append(test_P5)
            mAPs_cancer.append(test_mAP_cancer)
            R3s_cancer.append(test_R3_cancer)
            P5s_cancer.append(test_P5_cancer)

        all_test_mAP, all_test_R3, all_test_P5 = calc_retrieval_metrics(all_train_feature, all_train_label, all_test_feature, all_test_label)
        all_test_mAP_cancer, all_test_R3_cancer, all_test_P5_cancer = calc_retrieval_metrics_cancer(args.n_classes, all_train_feature, all_train_label, all_test_feature, all_test_label)

        for i, single_test_feature in enumerate(test_feature_list):
            single_test_mAP, single_test_R3, single_test_P5 = calc_retrieval_metrics(all_train_feature, all_train_label, single_test_feature, test_label_list[i])
            single_test_mAP_cancer, single_test_R3_cancer, single_test_P5_cancer = calc_retrieval_metrics_cancer(args.n_classes, all_train_feature, all_train_label, single_test_feature, test_label_list[i])
            single_mAPs.append(single_test_mAP)
            single_R3s.append(single_test_R3)
            single_P5s.append(single_test_P5)

            single_mAPs_cancer.append(single_test_mAP_cancer)
            single_R3s_cancer.append(single_test_R3_cancer)
            single_P5s_cancer.append(single_test_P5_cancer)

        marc_mean, src_mean, krc_mean = 0, 0, 0
        curr_marc_list, curr_src_list, curr_krc_list = [], [], []

        if task != 0:
            curr_marc_list = dataset.all_marc_list.copy()
            curr_src_list = dataset.all_src_list.copy()
            curr_krc_list = dataset.all_krc_list.copy()

            curr_marc_list.append(marc_list)
            curr_src_list.append(src_list)
            curr_krc_list.append(krc_list)

            marc_mean = average_sum_of_lists(curr_marc_list)
            src_mean = average_sum_of_lists(curr_src_list)
            krc_mean = average_sum_of_lists(curr_krc_list)            

        print("single mAP list:", single_mAPs)
        print("single R@3 list:", single_R3s)
        print("single P@5 list:", single_P5s)
        print("single cancer mAP list:", single_mAPs_cancer)
        print("single cancer R@3 list:", single_R3s_cancer)
        print("single cancer P@5 list:", single_P5s_cancer)
        print("mAP list:", mAPs)
        print("R@3 list:", R3s)
        print("P@5 list:", P5s)
        print("cancer mAP list:", mAPs_cancer)
        print("cancer R@3 list:", R3s_cancer)
        print("cancer P@5 list:", P5s_cancer)
        print("all mAP:", all_test_mAP)
        print("all R@3:", all_test_R3)
        print("all P@5:", all_test_P5)
        print("all cancer mAP:", all_test_mAP_cancer)
        print("all cancer R@3:", all_test_R3_cancer)
        print("all cancer P@5:", all_test_P5_cancer)
        print("curr marc list:", curr_marc_list)
        print("curr src list:", curr_src_list)
        print("curr krc list:", curr_krc_list)
        print("marc:", marc_mean)
        print("src:", src_mean)
        print("krc:", krc_mean)
    
    return single_mAPs, single_R3s, single_P5s, single_mAPs_cancer, single_R3s_cancer, single_P5s_cancer, mAPs, R3s, P5s, mAPs_cancer, R3s_cancer, P5s_cancer, all_test_mAP, all_test_R3, all_test_P5, all_test_mAP_cancer, all_test_R3_cancer, all_test_P5_cancer,\
    curr_marc_list, curr_src_list, curr_krc_list, marc_mean, src_mean, krc_mean


def average_sum_of_lists(list_of_lists):
    list_length = len(list_of_lists)
    sums = []
    counts = []
    for lst in list_of_lists:
        for index, value in enumerate(lst):
            if index >= len(sums):
                sums.append(value)
                counts.append(1)
            else:
                sums[index] += value
                counts[index] += 1
    
    total_average_sum = sum(s / c for s, c in zip(sums, counts)) / list_length
    
    return total_average_sum


def evaluate_retrieval_joint(args, model: ContinualModel, dataset: ContinualDataset):
    model.net.eval()
    with torch.no_grad():
        train_dl, val_dl, test_dl = dataset.get_joint_data_loaders()

        train_feature, train_label, train_uuid = get_hash_code(train_dl, model.device, model.net)
        test_feature, test_label, test_uuid = get_hash_code(test_dl, model.device, model.net)

        test_mAP, test_R3, test_P5 = calc_retrieval_metrics(train_feature, train_label, test_feature, test_label)
        test_mAP_cancer, test_R3_cancer,test_P5_cancer = calc_retrieval_metrics_cancer(args.n_classes, train_feature, train_label, test_feature, test_label)

        print("all mAP:", test_mAP)
        print("all R@3:", test_R3)
        print("all P@5:", test_P5)
        print("all cancer mAP:", test_mAP_cancer)
        print("all cancer R@3:", test_R3_cancer)
        print("all cancer P@5:", test_P5_cancer)
    
    return test_mAP, test_R3, test_P5, test_mAP_cancer, test_R3_cancer, test_P5_cancer, train_feature, train_label, train_uuid, test_feature, test_label, test_uuid


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)
    model.net.to(model.device)

    if model.NAME != 'joint':
        results_dir = os.path.join(args.checkpoints_save_path)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        results_dir = os.path.join(results_dir, args.exp_desc)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        results_dir = os.path.join(results_dir, 'fold_'+str(args.fold))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        results_dir = os.path.join(results_dir, args.model_desc)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        for task in range(dataset.N_TASKS):
            model.net.train()
            train_loader, val_loader, test_loader, bilevel_train_loader = dataset.get_data_loaders()
            
            if hasattr(model, 'begin_task'):
                model.begin_task(dataset)
            
            scheduler = dataset.get_scheduler(model, args)

            for epoch in range(model.args.n_epochs):
                total_loss = 0
                for i, data in enumerate(train_loader):
                    model.net.train()
                    if hasattr(dataset.train_loader.dataset, 'logits'):
                        input, target, logits, uuids = data
                        input, target, logits = input.to(model.device), target.to(model.device), logits.to(model.device)
                        loss = model.observe(input, target, task, logits)
                        print(' * task {task} epoch {epoch} idx {idx} Loss {Loss:.3f}'.format(task=task, epoch=epoch, idx=i, Loss=loss))

                    else:
                        input, target, uuids = data
                        input, target = input.to(model.device), target.to(model.device)
                        loss = model.observe(input, target, task)
                        print(' * task {task} epoch {epoch} idx {idx} Loss {Loss:.3f}'.format(task=task, epoch=epoch, idx=i, Loss=loss))
                    
                    total_loss += loss
                
                train_epoch_loss = total_loss / len(train_loader)
                print(' * task {task} epoch {epoch} average_loss {average_loss}'.format(task=task, epoch=epoch, average_loss=train_epoch_loss))
                with open(os.path.join(results_dir, 'record.csv'), 'a') as f:
                    f.write(
                        '{}, {}, {:.3f}'.format(
                            task, epoch, train_epoch_loss
                            )
                        )
                    f.write('\n')
                
                if (epoch+1) % 10 == 0:
                    single_mAPs, single_R3s, single_P5s, single_mAPs_cancer, single_R3s_cancer, single_P5s_cancer, mAPs, R3s, P5s, mAPs_cancer, R3s_cancer, P5s_cancer, all_test_mAP, all_test_R3, all_test_P5, all_test_mAP_cancer, all_test_R3_cancer, all_test_P5_cancer, marc_list, src_list, krc_list, marc_mean, src_mean, krc_mean = evaluate_retrieval_epoch(args, model, dataset, task, epoch)

                    single_mAPs = list2str(single_mAPs)
                    single_R3s = list2str(single_R3s)
                    single_P5s = list2str(single_P5s)
                    single_mAPs_cancer = list2str(single_mAPs_cancer)
                    single_R3s_cancer = list2str(single_R3s_cancer)
                    single_P5s_cancer = list2str(single_P5s_cancer)
                    mAPs = list2str(mAPs)
                    R3s = list2str(R3s)
                    P5s = list2str(P5s)
                    mAPs_cancer = list2str(mAPs_cancer)
                    R3s_cancer = list2str(R3s_cancer)
                    P5s_cancer = list2str(P5s_cancer)
                    marc_list = list_to_str_expression(marc_list)
                    src_list = list_to_str_expression(src_list)
                    krc_list = list_to_str_expression(krc_list)
                
                    with open(os.path.join(results_dir, 'evaluate_epoch_test.csv'), 'a') as f:
                        f.write(
                            '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {}, {}, {}, {:.6f}, {:.6f}, {:.6f}'.format(
                                task, epoch, single_mAPs, single_R3s, single_P5s, single_mAPs_cancer, single_R3s_cancer, single_P5s_cancer, mAPs, R3s, P5s, mAPs_cancer, R3s_cancer, P5s_cancer, all_test_mAP, all_test_R3, all_test_P5, all_test_mAP_cancer, all_test_R3_cancer, all_test_P5_cancer, marc_list, src_list, krc_list, marc_mean, src_mean, krc_mean
                            )
                        )
                        f.write('\n')

                    torch.save({
                        'epoch': epoch,
                        'task': task,
                        'state_dict': model.net.state_dict(),
                        'optimizer': model.opt.state_dict(),
                        'args': args
                        }, os.path.join(results_dir, 'task{task}_model_{epoch}.pth.tar'.format(task=task, epoch=epoch)))
                
                if scheduler is not None:
                    scheduler.step()

            if hasattr(model, 'save_buffer') and task != dataset.N_TASKS-1:
                for i, data in enumerate(train_loader):
                    input, target, uuids = data
                    input, target = input.to(model.device), target.to(model.device)
                    model.save_buffer(input, target, task)
            
            if hasattr(model, 'save_buffer_drs') and task != dataset.N_TASKS-1:
                model.save_buffer_drs(train_loader, task)

            if hasattr(model, 'end_task'):
                model.end_task(dataset)
            
            single_mAPs, single_R3s, single_P5s, single_mAPs_cancer, single_R3s_cancer, single_P5s_cancer, mAPs, R3s, P5s, mAPs_cancer, R3s_cancer, P5s_cancer, all_test_mAP, all_test_R3, all_test_P5, all_test_mAP_cancer, all_test_R3_cancer, all_test_P5_cancer, marc_list, src_list, krc_list, marc_mean, src_mean, krc_mean, train_feature, train_label, train_uuid, test_feature_list, test_label_list, test_uuid_list, all_test_feature, all_test_label, all_test_uuid, initial_uuids_list, updated_uuids_list = evaluate_retrieval(args, model, dataset, task)
            
            single_mAPs = list2str(single_mAPs)
            single_R3s = list2str(single_R3s)
            single_P5s = list2str(single_P5s)
            single_mAPs_cancer = list2str(single_mAPs_cancer)
            single_R3s_cancer = list2str(single_R3s_cancer)
            single_P5s_cancer = list2str(single_P5s_cancer)
            mAPs = list2str(mAPs)
            R3s = list2str(R3s)
            P5s = list2str(P5s)
            mAPs_cancer = list2str(mAPs_cancer)
            R3s_cancer = list2str(R3s_cancer)
            P5s_cancer = list2str(P5s_cancer)
            marc_list = list_to_str_expression(marc_list)
            src_list = list_to_str_expression(src_list)
            krc_list = list_to_str_expression(krc_list)

            with open(os.path.join(results_dir, 'evaluate.csv'), 'a') as f:
                f.write(
                    '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {}, {}, {}, {:.6f}, {:.6f}, {:.6f},'.format(
                        task, single_mAPs, single_R3s, single_P5s, single_mAPs_cancer, single_R3s_cancer, single_P5s_cancer, mAPs, R3s, P5s, mAPs_cancer, R3s_cancer, P5s_cancer, all_test_mAP, all_test_R3, all_test_P5, all_test_mAP_cancer, all_test_R3_cancer, all_test_P5_cancer, marc_list, src_list, krc_list, marc_mean, src_mean, krc_mean
                    )
                )
                f.write('\n')
            torch.save({
                    'train_feature': train_feature,
                    'train_label': train_label,
                    'train_uuid': train_uuid,
                    'test_feature_list': test_feature_list,
                    'test_label_list': test_label_list,
                    'test_uuid_list': test_uuid_list,
                    'all_test_feature': all_test_feature,
                    'all_test_label': all_test_label,
                    'all_test_uuid': all_test_uuid,
                    'initial_uuids_list': initial_uuids_list, 
                    'updated_uuids_list': updated_uuids_list
                    }, os.path.join(results_dir, 'task{task}_results.pth.tar'.format(task=task)))
    else:
        results_dir = os.path.join(args.checkpoints_save_path)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        results_dir = os.path.join(results_dir, args.exp_desc)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        results_dir = os.path.join(results_dir, 'fold_'+str(args.fold))
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        results_dir = os.path.join(results_dir, model.NAME)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        scheduler = dataset.get_scheduler(model, args)

        model.net.train()
        if hasattr(model, 'joint_task'):
            model.joint_task(dataset, results_dir, scheduler)
        
        test_mAP, test_R3, test_mAP_cancer, test_R3_cancer, train_feature, train_label, train_uuid, test_feature, test_label, test_uuid = evaluate_retrieval_joint(args, model, dataset)
        with open(os.path.join(results_dir, 'evaluate.csv'), 'a') as f:
            f.write(
                '{:.6f}, {:.6f}, {:.6f}, {:.6f},'.format(
                    test_mAP, test_R3, test_mAP_cancer, test_R3_cancer
                )
            )
            f.write('\n')
        
        torch.save({
                'train_feature': train_feature,
                'train_label': train_label,
                'train_uuid': train_uuid,
                'test_feature': test_feature,
                'test_label': test_label,
                'test_uuid': test_uuid,
            }, os.path.join(results_dir, 'joint_results.pth.tar'))
