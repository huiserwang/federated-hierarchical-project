#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
from torchvision import datasets, transforms
import math


def data_sample(args, dataset, d_matrix, D_fr, num_fv, num_fr):
    """
    Sample data for each FV an FR from the whole dataset
    """
    norm_factor = d_matrix.sum() + num_fr * D_fr  # \sum d_{i,j} + M*D
    d_ratio = d_matrix / norm_factor
    D_ratio = D_fr / norm_factor

    dataset_size = len(dataset)
    dict_fv_group, dict_fr_group, all_idxs = {}, {}, [i for i in range(dataset_size)]
    np.random.seed(args.seed)

    for i in range(num_fv):
        dict_fv_group[i] = dict()
        for j in range(num_fr):
            dict_fv_group[i][j] = set(np.random.choice(all_idxs,
                                                    max(math.floor(d_ratio[i][j] * dataset_size), 1),
                                                    replace=False))  # sample FV data according to d_{i, j}
            all_idxs = list(set(all_idxs) - dict_fv_group[i][j])

    for j in range(num_fr):
        dict_fr_group[j] = set(np.random.choice(all_idxs,
                                                max(math.floor(D_ratio * dataset_size), 1),
                                                replace=False))  # sample FR data according to D
        all_idxs = list(set(all_idxs) - dict_fr_group[j])
    return dict_fv_group, dict_fr_group


def get_dataset(args, d_matrix, D_fr):
    """ Returns train and test datasets and two user groups which are dict where
    the keys are the user index and the values are the corresponding data idxs for
    each of those users.
    """
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data for FVs and FRs
        fv_groups, fr_groups = data_sample(args, train_dataset, d_matrix, D_fr, args.num_fv, args.num_fr)

    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data for FVs and FRs
        fv_groups, fr_groups = data_sample(args, train_dataset, d_matrix, D_fr, args.num_fv, args.num_fr)

    return train_dataset, test_dataset, fv_groups, fr_groups


def aggregate_params(params, weights=None):
    """
    Returns the weighted average of the params.
    """
    w_avg = copy.deepcopy(params[0])
    device = list(params[0].items())[0][1].get_device()
    if weights is None:
        # average sum
        weights = torch.ones(len(params), device=device) / len(params)
    else:
        weights = weights / weights.sum()
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(params)):
            w_avg[key] += params[i][key] * weights[i]
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Number of FVs  : {args.num_fv}')
    print(f'    Number of FRs  : {args.num_fr}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
