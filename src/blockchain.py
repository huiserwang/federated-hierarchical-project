import torch


def get_weights_from_ground_chain(args, fr_idx):
    # TODO: implement searching in block chain
    weights = None
    return weights


def get_weights_from_top_chain(args, d_matrix, D_fr):
    # TODO: implement searching in block chain
    weights = d_matrix.sum(axis=0) + D_fr
    return weights