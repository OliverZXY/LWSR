# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--exp_desc', type=str, required=True,
                        help='Experiment description.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.')
    parser.add_argument('--model_desc', type=str,
                        help='Model name.')

    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=1e-6,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.9,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int, default=400,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size.')
    parser.add_argument('--bilevel_batch_size', type=int, default=10,
                        help='Bilevel batch size.')
    
    parser.add_argument('--joint_batch_size', type=int, default=40,
                        help='Batch size.')
    
    parser.add_argument('--n_classes', type=int, required=True,
                        help='Number of classes.')
    parser.add_argument('--fold', type=int, default=0,
                        help='Number of fold.')
    parser.add_argument('--checkpoints_save_path', type=str, default='',
                        help='Path for saving checkpoints and records.')
    
    parser.add_argument('--load_checkpoint', type=int, default=0,
                        help='Weather to load checkpoint.')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--test_mode', type=int, default=0,
                        help='Weather to execute testing mode.')
    parser.add_argument('--test_weights_path', type=str, default='',
                        help='Path to testing checkpoints.')

    parser.add_argument('--n_tasks', type=int, default=4,
                        help='Number of tasks.')
    parser.add_argument('--n_classes_per_task', type=int, default=2,
                        help='Number of classes per task.')
    parser.add_argument('--n_clusters_train', type=int, default=5,
                        help='Number of training data clusters.')
    parser.add_argument('--n_clusters_buffer', type=int, default=3,
                        help='Number of buffer data clusters.')
    
    parser.add_argument('--pair_loss_weight', type=float, default=0.5,
                        help='Coefficient of pair loss weight.')
    parser.add_argument('--ce_loss_weight', type=float, default=0.5,
                        help='Coefficient of ce loss weight.')
    parser.add_argument('--dc_loss_weight', type=float, default=0.5,
                        help='Coefficient of dc loss weight.')
    
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for model.')
    parser.add_argument('--n_sample_num', type=int, default=2048,
                        help='Number of sampling instances.')
    parser.add_argument('--sampling_num', type=int, required=True)


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=0,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging', default=True)
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, default=100,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, default=20,
                        help='The batch size of the memory buffer.')
