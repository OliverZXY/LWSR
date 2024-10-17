# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math

import numpy as np
import torch
from datasets.utils.validation import ValidationDataset
from torch.optim import SGD
from torchvision import transforms

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from utils.loss import RetrievalLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


def get_hash_code(data_loader, device, model):
    # switch to test mode
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
            target = label.to(device, non_blocking=True)

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
    """
    计算两组向量之间的欧氏距离矩阵。
    X: 第一组向量，大小为 (N, D)
    Y: 第二组向量，大小为 (M, D)
    """
    dists = -2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
    return dists


def compute_AP(retrieved_labels, true_label):
    """
    计算单个查询的平均精度（AP）。
    retrieved_labels: 检索到的标签列表。
    true_label: 真实标签。
    """
    relevant = np.array(retrieved_labels) == true_label
    cumsum = np.cumsum(relevant)
    precision_at_k = cumsum / (np.arange(len(relevant)) + 1)
    AP = np.sum(precision_at_k * relevant) / np.sum(relevant)
    return AP


def mAP_R_at_k_and_P_at_t(database_labels, retrieved_indices, query_labels, k=3, t=5):
    """
    计算mAP和R@k。
    database_labels: 数据库中的标签。
    retrieved_indices: 检索到的索引，大小为 (num_queries, num_database)。
    query_labels: 查询的标签。
    k: 计算R@k时的k值。
    """
    num_queries = query_labels.shape[0]
    AP_scores = []
    correct_at_k = 0
    precision_at_t_sum = 0  # 用于计算P@5的累计值
    
    for i in range(num_queries):
        true_label = query_labels[i]        
        retrieved_k = [database_labels[idx] for idx in retrieved_indices[i, :k]]
        retrieved_t = [database_labels[idx] for idx in retrieved_indices[i, :t]]  # 取前5个检索结果
        
        if true_label in retrieved_k:
            correct_at_k += 1
        
        # 计算P@5
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

    if n_classes == 8:
        train_label = train_label // 2
        test_label = test_label // 2
    elif n_classes == 13:
        train_label[(train_label == 0) | (train_label == 1) | (train_label == 2) | (train_label == 3) | (train_label == 4) | (train_label == 5)] = 0
        train_label[(train_label == 6) | (train_label == 7)] = 1
        train_label[(train_label == 8) | (train_label == 9) | (train_label == 10)] = 2
        train_label[(train_label == 11) | (train_label == 12)] = 3

        test_label[(test_label == 0) | (test_label == 1) | (test_label == 2) | (test_label == 3) | (test_label == 4) | (test_label == 5)] = 0
        test_label[(test_label == 6) | (test_label == 7)] = 1
        test_label[(test_label == 8) | (test_label == 9) | (test_label == 10)] = 2
        test_label[(test_label == 11) | (test_label == 12)] = 3
    elif n_classes == 19:
        train_label[(train_label == 0) | (train_label == 1)] = 0
        train_label[(train_label == 2) | (train_label == 3) | (train_label == 4) | (train_label == 5)] = 1
        train_label[(train_label == 6) | (train_label == 7) | (train_label == 8) | (train_label == 9)] = 2
        train_label[(train_label == 10) | (train_label == 11) | (train_label == 12)] = 3
        train_label[(train_label == 13) | (train_label == 14) | (train_label == 15) | (train_label == 16)] = 4
        train_label[(train_label == 17) | (train_label == 18)] = 5

        test_label[(test_label == 0) | (test_label == 1)] = 0
        test_label[(test_label == 2) | (test_label == 3) | (test_label == 4) | (test_label == 5)] = 1
        test_label[(test_label == 6) | (test_label == 7) | (test_label == 8) | (test_label == 9)] = 2
        test_label[(test_label == 10) | (test_label == 11) | (test_label == 12)] = 3
        test_label[(test_label == 13) | (test_label == 14) | (test_label == 15) | (test_label == 16)] = 4
        test_label[(test_label == 17) | (test_label == 18)] = 5
    
    mAP, R_at_3, P_at_5 = calc_retrieval_metrics(train_feature, train_label, test_feature, test_label)

    return mAP, R_at_3, P_at_5


def evaluate_retrieval_joint(args, model, dataset, device):
    model.eval()
    with torch.no_grad():
        train_dl, val_dl, test_dl = dataset.get_joint_data_loaders()

        train_feature, train_label, train_uuid = get_hash_code(train_dl, device, model)
        test_feature, test_label, test_uuid = get_hash_code(test_dl, device, model)

        test_mAP, test_R3, test_P5 = calc_retrieval_metrics(train_feature, train_label, test_feature, test_label)
        test_mAP_cancer, test_R3_cancer, test_P5_cancer = calc_retrieval_metrics_cancer(args.n_classes, train_feature, train_label, test_feature, test_label)

        print("all mAP:", test_mAP)
        print("all R3:", test_R3)
        print("all P5:", test_P5)
        print("all cancer mAP:", test_mAP_cancer)
        print("all cancer R3:", test_R3_cancer)
        print("all cancer P5:", test_P5_cancer)
    
    return test_mAP, test_R3, test_P5, test_mAP_cancer, test_R3_cancer, test_P5_cancer, train_feature, train_label, train_uuid, test_feature, test_label, test_uuid


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)

    def joint_task(self, dataset, results_dir, scheduler):
        criterion = RetrievalLoss()
        train_loader, val_loader, test_loader = dataset.get_joint_data_loaders()

        best_test_mAP, best_test_R3, best_test_P5, best_test_mAP_cancer, best_test_R3_cancer, best_test_P5_cancer = 0, 0, 0, 0, 0, 0

        # train
        for epoch in range(self.args.n_epochs):
            total_loss = 0
            for i, data in enumerate(train_loader):
                self.net.train()
                inputs, labels, uuids = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.opt.zero_grad()
                results_dict = self.net(data=inputs)
                criterion.set(pred=results_dict['logits'], label=labels, features=results_dict['cls_token'], task=0, is_ours=False)
                loss = criterion.total_loss_func()
                total_loss += loss.item()
                loss.backward()
                self.opt.step()
                
                print(' * epoch {epoch} idx {idx} Loss {Loss:.3f}'.format(epoch=epoch, idx=i, Loss=loss.item()))

                with open(os.path.join(results_dir, 'record.csv'), 'a') as f:
                    f.write(
                        '{}, {}, {:.3f}'.format(
                            epoch, i, loss
                        )
                    )
                    f.write('\n')
            
            if scheduler is not None:
                scheduler.step()
            
            average_loss = total_loss / len(train_loader)
            print(' * epoch {epoch} average Loss {Loss:.3f}'.format(epoch=epoch, Loss=average_loss))
            
            if (epoch+1) % 10 == 0:
                test_mAP, test_R3, test_P5, test_mAP_cancer, test_R3_cancer, test_P5_cancer, train_feature, train_label, train_uuid, test_feature, test_label, test_uuid = evaluate_retrieval_joint(self.args, self.net, dataset, self.device)

                with open(os.path.join(results_dir, 'evaluate_per_10epoch.csv'), 'a') as f:
                    f.write(
                        '{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
                            test_mAP, test_R3, test_P5, test_mAP_cancer, test_R3_cancer, test_P5_cancer
                        )
                    )
                    f.write('\n')
                
                if test_mAP >= best_test_mAP:
                    best_test_mAP = test_mAP
                    best_test_R3 = test_R3
                    best_test_P5 = test_P5
                    best_test_mAP_cancer = test_mAP_cancer
                    best_test_R3_cancer = test_R3_cancer
                    best_test_P5_cancer = test_P5_cancer

                print("best mAP:", best_test_mAP)
                print("best R3:", best_test_R3)
                print("best P5:", best_test_P5)
                print("best cancer mAP:", best_test_mAP_cancer)
                print("best cancer R3:", best_test_R3_cancer)
                print("best cancer P5:", best_test_P5_cancer)

            torch.save({
                'epoch': epoch,
                'state_dict': self.net.state_dict(),
                'optimizer': self.opt.state_dict(),
                'args': self.args
            }, os.path.join(results_dir, 'model_{epoch}.pth.tar'.format(epoch=epoch)))

    def observe(self, inputs, labels, task):
        return 0
