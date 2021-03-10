# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: BERT-score.py 
@time: 2020年12月16日10时18分 
"""
import numpy as np
import sys
import math
import time
import pickle as pk
from sklearn.metrics import accuracy_score


class LogUtil(object):
    """
    Tool of Log
    """
    def __init__(self):
        pass

    @staticmethod
    def log(typ, msg):
        print("[%s]\t[%s]\t%s" % (TimeUtil.t_now(), typ, str(msg)))
        sys.stdout.flush()
        return


class TimeUtil(object):
    """
    Tool of Time
    """
    def __init__(self):
        return

    @staticmethod
    def t_now():
        """
        Get the current time, e.g. `2016-12-27 17:14:01`
        :return: string represented current time
        """
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    @staticmethod
    def t_now_YmdH():
        return time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))


def load_file(token_file):
    data_file = open(token_file, 'r', encoding='UTF-8').readlines()
    data_list = []
    for line in data_file:
        ses = line.strip().split("\t")
        data_list.append(ses)
    return data_list


def generate_idf(data_list):
    idf = {}
    q_set = set()
    for data in data_list:
        if data[0] not in q_set:
            q_set.add(data[0])
            words = data[0].split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
        if data[1] not in q_set:
            q_set.add(data[1])
            words = data[1].split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
    num_docs = len(data_list)
    for word in idf:
        idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
    LogUtil.log("INFO", "idf calculation done, len(idf)=%d" % len(idf))
    return idf


def bert_score(data_list, matrix, idf_dict, corref):
    all_BERT_score = []
    true_label = []
    for index in range(len(matrix)):
        idf_list = []
        ses = data_list[index]
        sentence_1 = ses[0].split()
        sentence_2 = ses[1].split()
        true_label.append(int(ses[2]))
        for word in sentence_2:
            idf_list.append(idf_dict[word])
        score = np.max(matrix[index][0], axis=0)
        assert len(score) == len(idf_list), print(sentence_2, matrix[index][0].shape, len(score), len(idf_list))
        numerator = np.dot(idf_list, score)
        # axis=1 求得每一行最大值 axis=0 求得每一列最大值
        denominator = np.sum(idf_list)
        BERT_score = numerator/denominator
        if BERT_score > corref:
            all_BERT_score.append(1)
        else:
            all_BERT_score.append(0)
    assert len(all_BERT_score) == len(true_label), "数量异常"
    acc = accuracy_score(y_pred=all_BERT_score, y_true=true_label)
    print(acc)


train_path = "E:/2020-GD/Semantic Interoperation/test/lcqmc_test_Elmo_interaction.pk"
all_file = "E:/2020-GD/评价数据/lcqmc/all_token_simply_line.txt"
train_data = pk.load(open(train_path, mode='rb'))
all_data = load_file(all_file)[238766:]
print(len(all_data))
print(len(train_data))
data_idf = generate_idf(all_data)
bert_score(all_data, train_data, data_idf, 0.74)
bert_score(all_data, train_data, data_idf, 0.75)
bert_score(all_data, train_data, data_idf, 0.76)
bert_score(all_data, train_data, data_idf, 0.78)
bert_score(all_data, train_data, data_idf, 0.79)
bert_score(all_data, train_data, data_idf, 0.80)
bert_score(all_data, train_data, data_idf, 0.82)


