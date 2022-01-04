# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from config.cfg import path, cfg
from pytorch_transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(path['roberta_path'])

def load_data(path):
    data_X_1, data_X_2, y = [], [], []

    with open(path, 'r', encoding='utf-8') as file:
        idx = 0
        for line in file:
            if idx > 0:
                line = line.strip()
                line = line.split('\t')
                data_X_1.append(line[1])
                data_X_2.append(line[2])

                if line[3] == 'not_entailment':
                    y.append(0)
                elif line[3] == 'entailment':
                    y.append(1)
                else:
                    print('error !')

            idx += 1
    return data_X_1, data_X_2, y

def generate_template(data_X_1, data_X_2):
    data_X = []
    CLS = '<s>'
    SEP = '</s>'
    MASK = '<mask>'

    for i in range(len(data_X_1)):
        template = cfg['template']

        template = template.replace('[X1]', CLS + ' ' + data_X_1[i])
        template = template.replace('[X2]', SEP + ' ' + data_X_2[i] + ' ' + SEP)
        template = template.replace('[MASK]', MASK)

        data_X.append(template)
    return data_X

def get_random_sample_ids(length, K):
    import random
    ids_list = []
    for i in range(length):
        ids_list.append(i)
    ids = random.sample(ids_list, K)
    return ids

def data_split_type(data_X, data_y, K=8, Kt=1000):
    '''
    对当前类划分
    :param data_X:
    :param data_y:
    :param K:
    :param Kt:
    :return:
    '''
    train_ids = get_random_sample_ids(len(data_X), K)
    train_X, train_y = [], []
    test_all_X, test_all_y = [], []

    for i in range(len(data_X)):
        if i in train_ids:
            train_X.append(data_X[i])
            train_y.append(data_y[i])
        else:
            test_all_X.append(data_X[i])
            test_all_y.append(data_y[i])

    test_ids = get_random_sample_ids(len(test_all_X), Kt)
    test_X, test_y = [], []
    for i in range(len(test_all_X)):
        if i in test_ids:
            test_X.append(test_all_X[i])
            test_y.append(test_all_y[i])

    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


def data_split(data_X, data_y, K=8, Kt=1000):
    '''
    对每一类划分
    :param data_X:
    :param data_y:
    :param K:
    :param Kt:
    :return:
    '''
    ss = set()
    for i in range(len(data_y)):
        ss.add(data_y[i])

    train_X, train_y, test_X, test_y = [], [], [], []
    for value in ss:
        X_t, y_t = [], []

        for i in range(len(data_y)):
            if data_y[i] == value:
                X_t.append(data_X[i])
                y_t.append(data_y[i])

        train_X_t, train_y_t, test_X_t, test_y_t = data_split_type(X_t, y_t, K, Kt)

        if len(train_X) == 0:
            train_X, train_y, test_X, test_y = train_X_t, train_y_t, test_X_t, test_y_t
        else:
            train_X = np.hstack([train_X, train_X_t])
            train_y = np.hstack([train_y, train_y_t])
            test_X = np.hstack([test_X, test_X_t])
            test_y = np.hstack([test_y, test_y_t])

    return train_X, train_y, test_X, test_y










