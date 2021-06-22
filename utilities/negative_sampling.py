from utilities import loss_functions
from utilities.data_utilities import KnowledgeGraph
import numpy as np
from time import time
from sklearn.utils import shuffle as skshuffle
import os
import random
np.random.seed(0)
random.seed(0)


def sample_negatives(X, C):
    """
    Perform negative sampling by corrupting head or tail of each triplets in
    dataset.

    Params:
    -------
    X: int matrix of M x 3, where M is the (mini)batch size
        First column contains index of head entities.
        Second column contains index of relationships.
        Third column contains index of tail entities.

    n_e: int
        Number of entities in dataset.

    Returns:
    --------
    X_corr: int matrix of M x 3, where M is the (mini)batch size
        Similar to input param X, but at each column, either first or third col
        is subtituted with random entity.
    """
    M = X.shape[0]
    # head, tail, relation
    X_corr = np.zeros([M * C, 3])

    for i in range(0, M):
        corrupt_head_prob = np.random.rand(1)

        e_idxs = np.random.choice(M, C)
        if corrupt_head_prob > 0.5:
            for j in range(0, C):
                if e_idxs[j] != i:
                    X_corr[i + j * M, 0] = X[e_idxs[j], 0]
                else:
                    X_corr[i + j * M, 0] = X[e_idxs[j] // 2, 0]
                X_corr[i + j * M, 1] = X[i, 1] #tail
                X_corr[i + j * M, 2] = X[i, 2] #relation
        else:
            for j in range(0, C):
                X_corr[i + j * M, 0] = X[i, 0]
                if e_idxs[j] != i:
                    X_corr[i + j * M, 1] = X[e_idxs[j], 1]
                else:
                    X_corr[i + j * M, 1] = X[e_idxs[j] // 2, 1]
                X_corr[i + j * M, 2] = X[i, 2]

    return X_corr


def sample_negatives_head_and_tail(X, C, entity_list):
    M = X.shape[0]  #batch size
    X_corr = np.zeros([M * C* 2, 3])  # x2 for head and tail corruption at the same time(UKGE)

    for i in range(0, M):
        true_head = X[i, 0]
        true_tail = X[i, 1]
        true_relation = X[i, 2]
        for j in range(0, C*2, 2):
            # Head Corruption
            corrupt_head = random.choice(entity_list)
            X_corr[j + (i * C * 2), 0] = corrupt_head  # head
            X_corr[j + (i * C * 2), 1] = true_tail   # tail
            X_corr[j + (i * C * 2 ), 2] = true_relation  # relation
            # Tail Corruption
            corrupt_tail = random.choice(entity_list)
            X_corr[j + 1 + (i * C * 2), 0] = true_head  # head
            X_corr[j + 1 + (i * C * 2), 1] = corrupt_tail  # tail
            X_corr[j + 1 + (i * C * 2), 2] = true_relation  # relation

    return X_corr


def sample_negatives_head_and_tail2(X, C, entity_list):
    M = X.shape[0]  #batch size
    X_corr = np.zeros([M * C*2, 3])  # x2 for head and tail corruption at the same time(UKGE)

    for i in range(0, M):
        true_head = X[i, 0]
        true_tail = X[i, 1]
        true_relation = X[i, 2]
        # Head Corruption
        for j in range(0, C):
            corrupt_head = random.choice(entity_list)
            X_corr[i + j * M, 0] = corrupt_head  # head
            X_corr[i + j * M, 1] = true_tail   # tail
            X_corr[i + j * M, 2] = true_relation  # relation
        # Tail Corruption
        for k in range(C, C*2):
            corrupt_tail = random.choice(entity_list)
            X_corr[i + k * M, 0] = true_head  # head
            X_corr[i + k * M, 1] = corrupt_tail  # tail
            X_corr[i + k * M, 2] = true_relation  # relation
    return X_corr


def sample_negatives_only(X, C, entity_list, pos_triple_filter):
    M = X.shape[0]  #batch size
    X_corr = np.zeros([M * C*2, 3])  # x2 for head and tail corruption at the same time(UKGE)

    for i in range(0, M):
        true_head = X[i, 0]
        true_tail = X[i, 1]
        true_relation = X[i, 2]

        for j in range(0, C * 2, 2):
            # Head Corruption
            corrupt_head = random.choice(entity_list)
            while not is_neg_sample(corrupt_head, true_tail, true_relation, pos_triple_filter):
                corrupt_head = random.choice(entity_list)
            X_corr[j + (i * C * 2), 0] = corrupt_head  # head
            X_corr[j + (i * C * 2), 1] = true_tail   # tail
            X_corr[j + (i * C * 2 ), 2] = true_relation  # relation
            # Tail Corruption
            corrupt_tail = random.choice(entity_list)
            while not is_neg_sample(true_head, corrupt_tail, true_relation, pos_triple_filter):
                corrupt_tail = random.choice(entity_list)
            X_corr[j + 1 + (i * C * 2), 0] = true_head  # head
            X_corr[j + 1 + (i * C * 2), 1] = corrupt_tail  # tail
            X_corr[j + 1 + (i * C * 2), 2] = true_relation  # relation

    return X_corr


def sample_negatives_only2(X, C, entity_list, all_triples):

    M = X.shape[0]  #batch size
    X_corr = np.zeros([M * C*2, 3])  # x2 for head and tail corruption at the same time(UKGE)

    for i in range(0, M):
        true_head = X[i, 0]
        true_tail = X[i, 1]
        true_relation = X[i, 2]
        # Head Corruption
        for j in range(0, C):
            corrupt_head = random.choice(entity_list)
            while not is_neg_sample(corrupt_head, true_tail, true_relation, all_triples):
                corrupt_head = random.choice(entity_list)
            X_corr[i + j * M, 0] = corrupt_head  # head
            X_corr[i + j * M, 1] = true_tail   # tail
            X_corr[i + j * M, 2] = true_relation  # relation
        # Tail Corruption
        for k in range(C, C*2):
            corrupt_tail = random.choice(entity_list)
            while not is_neg_sample(true_head, corrupt_tail, true_relation, all_triples):
                corrupt_tail = random.choice(entity_list)
            X_corr[i + k * M, 0] = true_head  # head
            X_corr[i + k * M, 1] = corrupt_tail  # tail
            X_corr[i + k * M, 2] = true_relation  # relation

    return X_corr


def is_neg_sample(head, tail, relation, positive_filter):
    if positive_filter.get((head, tail, relation)) == 1:
        return False
    else:
        return True

