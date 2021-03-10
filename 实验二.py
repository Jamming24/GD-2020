# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: 实验二.py 
@time: 2021年02月23日18时23分 
"""

import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.svm import SVC
import pickle as pk
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def getTF_IDF_matrix(question_text):
    vectorizer = TfidfVectorizer(min_df=5, sublinear_tf=True)
    vectorizer.fit(question_text)
    # word_list = vectorizer.get_feature_names()
    # print(len(word_list))
    # print(tf_idf_vectorizer.toarray())
    return vectorizer


def load_BERT_embedding(BERT_json_file, BERT_embedding_file, save_flag=False):
    print("加载BERT json 文件")
    json_f = open(BERT_json_file)
    BERT_CLS_info_list = []
    # 将json格式的数据映射成list的形式
    for json_line in tqdm(json_f.readlines()):  # 按行读取json文件，每行为一个字符串
        data = json.loads(json_line)  # 将字符串转化为列表
        # index = data["linex_index"]
        # 返回CLS向量 768维度
        CLS_info = data["features"][0]["layers"][0]["values"]
        BERT_CLS_info_list.append(CLS_info)
    json_f.close()
    BERT_CLS_info = np.array(BERT_CLS_info_list)
    print(BERT_CLS_info.shape)
    print(BERT_CLS_info[:5])
    if save_flag:
        np.savetxt(BERT_embedding_file, BERT_CLS_info, delimiter="\t")
        print("BERT embedding 保存完成")
    return BERT_CLS_info


def load_tf_idf_features(train, dev):
    print("转换成TF-IDF特征")
    train_csv = pd.read_csv(open(train, encoding="UTF-8"), header=None, names=["text1", "text2", "label"], sep="\t")
    dev_csv = pd.read_csv(open(dev, encoding="UTF-8"), header=None, names=["text1", "text2", "label"], sep="\t")

    train_data_sentence_1 = train_csv["text1"].tolist()
    train_data_sentence_2 = train_csv["text2"].tolist()
    print("句子总量: ", len(train_data_sentence_1))
    train_data_sentence = []
    for index in range(len(train_data_sentence_1)):
        train_data_sentence.append(train_data_sentence_1[index] + "  " + train_data_sentence_2[index])
    # train_data = data_process.text_stemmer(train_csv["question_text"], flag=1).tolist()
    train_label = train_csv["label"]

    dev_data_sentence_1 = dev_csv["text1"].tolist()
    dev_data_sentence_2 = dev_csv["text2"].tolist()
    # dev_data = data_process.text_stemmer(dev_csv["question_text"], flag=1).tolist()
    dev_data_sentence = []
    for index in range(len(dev_data_sentence_1)):
        dev_data_sentence.append(dev_data_sentence_1[index] + "  " + dev_data_sentence_2[index])
    dev_label = dev_csv["label"]

    tf_idf_vertorized = getTF_IDF_matrix(train_data_sentence)
    train_tf_idf_vectorized = normalize(tf_idf_vertorized.transform(train_data_sentence), norm='l2')
    dev_tf_idf_vectorized = normalize(tf_idf_vertorized.transform(dev_data_sentence), norm='l2')
    return train_tf_idf_vectorized, train_label, dev_tf_idf_vectorized, dev_label


# 合并TF-IDF和CLS信息到一个文件中，使用crs稀疏矩阵读取 返回作为特征
def merge_tfidf_cls(data_tfidf, cls_out_file, tfidf_file, features_file):
    # 保存TF-IDF特征到文件
    np.savetxt(tfidf_file, data_tfidf.toarray(), delimiter='\t')
    tfidf_file_object = open(tfidf_file, mode='r')
    cls_out_file_object = open(cls_out_file, mode='r')
    features_file_object = open(features_file, mode='w')
    temp_cls_list = []
    print("开始合并两种特征........")
    for line in cls_out_file_object:
        temp_cls_list.append(line.strip("\n"))
    index = 0
    for line in tfidf_file_object:
        newline = line.strip("\n") + "\t" + temp_cls_list[index]
        index += 1
        features_file_object.write(newline + "\n")
    tfidf_file_object.close()
    cls_out_file_object.close()
    features_file_object.close()


def merge_BERT(features, train_cls_out_file, test_cls_out_file, features_file):
    features_data = pk.load(open(features, 'rb'))
    train_features_data = features_data[:34334]
    test_features_data = features_data[34334:]
    train_CLS_info = np.loadtxt(train_cls_out_file, delimiter="\t")
    test_CLS_info = np.loadtxt(test_cls_out_file, delimiter="\t")
    train_merge_features_list = []
    test_merge_features_list = []
    print("开始合并两种特征........")
    print(len(train_features_data))
    print(len(test_features_data))
    for index in range(len(train_features_data)):
        train_merge_features_list.append(train_CLS_info[index].tolist() + train_features_data[index])
    train_merge_features_matrix = np.array(train_merge_features_list)
    print("训练特征形状: ", train_merge_features_matrix.shape)
    for index in range(len(test_features_data)):
        test_merge_features_list.append(test_CLS_info[index].tolist() + test_features_data[index])
    test_merge_features_matrix = np.array(test_merge_features_list)
    print("测试特征形状: ", test_merge_features_matrix.shape)
    return train_merge_features_matrix, test_merge_features_matrix


train_json_file = "E:/afqmc_features/afqmc_train_embedding_-1_maxAuc.json"
test_json_file = "E:/afqmc_features/afqmc_test_embedding_-1_maxAuc.json"
train_BERT_embedding_file = "E:/afqmc_features/afqmc_train_embedding_maxAuc.txt"
test_BERT_embedding_file = "E:/afqmc_features/afqmc_test_embedding_maxAuc.txt"
train_file = "E:/afqmc_features/afqmc/afqmc_train_token.txt"
test_file = "E:/afqmc_features/afqmc/afqmc_dev_token.txt"
train_features_file = "E:/afqmc_features/train_BERT+TFIDF_features.tsv"
test_features_file = "E:/afqmc_features/test_BERT+TFIDF_features.tsv"
train_TFIDF_file = "E:/afqmc_features/train_TFIDF_features.txt"
test_TFIDF_file = "E:/afqmc_features/test_TFIDF_features.txt"
train_BERT_embedding = load_BERT_embedding(train_json_file, train_BERT_embedding_file, True)
test_BERT_embedding = load_BERT_embedding(test_json_file, test_BERT_embedding_file, True)
# train_tf_idf, train_label, test_tf_idf, test_label = load_tf_idf_features(train_file, test_file)
# merge_tfidf_cls(train_tf_idf, train_BERT_embedding_file, train_TFIDF_file, train_features_file)
# merge_tfidf_cls(test_tf_idf, test_BERT_embedding_file, test_TFIDF_file, test_features_file)
# train_BERT_embedding_file = "E:/afqmc_features/afqmc_train_embedding.txt"
# test_BERT_embedding_file = "E:/afqmc_features/afqmc_test_embedding.txt"
# train_CLS_info = np.loadtxt(train_BERT_embedding_file, delimiter="\t")
# test_CLS_info = np.loadtxt(test_BERT_embedding_file, delimiter="\t")
# print("train shape: ", train_CLS_info.shape)
# print("test shape: ", test_CLS_info.shape)
# static_features = "E:/afqmc_features/afqmc_static_features.pk"
# features_data = pk.load(open(static_features, 'rb'))
# train_features_data = features_data[:34334]
# test_features_data = features_data[34334:]
# print("train shape: ", np.array(train_features_data).shape)
# print("test shape: ", np.array(test_features_data).shape)

# word2vector_features = "E:/afqmc_features/afqmc_glove_AveVec.pk"
# word2vector_features_data = pk.load(open(word2vector_features, 'rb'))
# word2vector_train_features_data = word2vector_features_data[:34334]
# word2vector_test_features_data = word2vector_features_data[34334:]
# print("train shape: ", np.array(word2vector_train_features_data).shape)
# print("test shape: ", np.array(word2vector_test_features_data).shape)
#
# train_merge_features_list = []
# test_merge_features_list = []
# print("开始合并三种特征........")
# for index in range(len(train_features_data)):
#     train_merge_features_list.append(train_CLS_info[index].tolist() + train_features_data[index] + word2vector_train_features_data[index])
# train_merge_features_matrix = np.array(train_merge_features_list)
# print("训练特征形状: ", train_merge_features_matrix.shape)
# for index in range(len(test_features_data)):
#     test_merge_features_list.append(test_CLS_info[index].tolist() + test_features_data[index] + word2vector_test_features_data[index])
# test_merge_features_matrix = np.array(test_merge_features_list)
# print("测试特征形状: ", test_merge_features_matrix.shape)
#
# param_grid_SVC = [210, 215, 220, 230, 240, 250, 300, 350, 400, 450, 500]
# for c in param_grid_SVC:
#     svm_clf = AdaBoostClassifier(n_estimators=c, random_state=0)
#     svm_clf.fit(train_merge_features_matrix, train_label)
#     y_predict_train = svm_clf.predict(train_merge_features_matrix)
#     y_predict_dev = svm_clf.predict(test_merge_features_matrix)
#     train_SVC_F1 = f1_score(y_true=train_label, y_pred=y_predict_train)
#     dev_SVC_F1 = f1_score(y_true=test_label, y_pred=y_predict_dev,)
#     train_SVC_auc = accuracy_score(y_true=train_label, y_pred=y_predict_train)
#     dev_SVC_auc = accuracy_score(y_true=test_label, y_pred=y_predict_dev)
#     print("参数C为", c, "训练集合Adaboost_f1_score", train_SVC_F1, sep=":")
#     print("参数C为", c, "测试集合Adaboost_f1_score", dev_SVC_F1, sep=":")
#     print("参数C为", c, "训练集合Adaboost_AUC", train_SVC_auc, sep=":")
#     print("参数C为", c, "测试集合Adaboost_AUC", dev_SVC_auc, sep=":")

# 1.生成释义句 2.筛选释义句 3,由于生成的都是正例， 为了保持数据平衡 要替换掉原数据中 一些距离分类面较远的释义数据