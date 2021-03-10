# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: ensemble_afqmc.py 
@time: 2021年02月25日12时43分 
"""

import numpy as np
import pickle as pk
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import clone
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore")

from Fire2019_CIQ.brew import Ensemble, EnsembleClassifier
from Fire2019_CIQ.brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from Fire2019_CIQ.brew.combination.combiner import Combiner


def voting_learning():
    log_clf = LogisticRegression(max_iter=10)
    svm_clf = SVC(probability=True, decision_function_shape='ovo', kernel="linear", C=0.2)
    knnclf = KNeighborsClassifier(n_neighbors=10)
    NB_clf = MultinomialNB(alpha=0.01)
    tree_clf = DecisionTreeClassifier(max_depth=3)
    rnd_clf = RandomForestClassifier()
    voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('SVM', svm_clf), ("KNN_10", knnclf), ('NB', NB_clf), ('Tree', tree_clf), ('rf', rnd_clf)], voting='soft')
    print("集成学习")
    voting_clf.fit(train_tf_idf_vectorize, train_label)
    y_predict = voting_clf.predict(dev_tf_idf_vectorize)
    print(classification_report(dev_label, y_predict))
    print("各个分类器的预测结果：")
    for clf in (log_clf, svm_clf, knnclf, NB_clf, tree_clf, rnd_clf, voting_clf):
        clf.fit(train_tf_idf_vectorize, train_label)
        y_pred = clf.predict(dev_tf_idf_vectorize)
        print(clf.__class__.__name__, clf.score(train_tf_idf_vectorize, train_label))
        print(clf.__class__.__name__, accuracy_score(dev_label, y_pred), sep=":")


def stacking_learning(train_data, train_Y, test_data, test_Y):
    log_clf = LogisticRegression(max_iter=10)
    svm_clf = SVC(probability=True, decision_function_shape='ovo', kernel="linear", C=0.2)
    knnclf = KNeighborsClassifier(n_neighbors=10)
    NB_clf = MultinomialNB(alpha=0.01)
    tree_clf = DecisionTreeClassifier(max_depth=3)
    rnd_clf = RandomForestClassifier()
    ensemble = Ensemble([log_clf, knnclf, tree_clf, rnd_clf])
    # ('lr', log_clf), ('SVM', svm_clf), ("KNN_10", knnclf), ('NB', NB_clf), ('Tree', tree_clf), ('rf', rnd_clf)
    eclf = EnsembleClassifier(ensemble=ensemble, combiner=Combiner('mean'))
    # Creating Stacking
    layer_1 = Ensemble([log_clf, knnclf, tree_clf, rnd_clf])
    layer_2 = Ensemble([clone(log_clf)])
    stack = EnsembleStack(cv=5)
    stack.add_layer(layer_1)
    stack.add_layer(layer_2)
    sclf = EnsembleStackClassifier(stack)
    clf_list = [log_clf, knnclf, tree_clf, rnd_clf, eclf, sclf]
    lbl_list = ['Logistic Regression', 'KNN', 'tree_clf', 'rnd_clf', 'Ensemble', 'Stacking']

    itt = itertools.product([0, 1, 2, 3, 4, 5, 6, 7], repeat=10)
    print("brew----------------")
    for clf, lab, grd in zip(clf_list, lbl_list, itt):
        clf.fit(train_data, train_Y)
        y_pred = clf.predict(test_data)
        print(clf.__class__.__name__, accuracy_score(test_Y, y_pred), sep=":")
        # if clf.__class__.__name__ == "EnsembleStackClassifier":
        #     print("EnsembleStackClassifier输出最终结果")
        #     print(clf.stack)


root_path = "E:/afqmc_features/"
train_path = root_path + "afqmc/afqmc_train_token.txt"
test_path = root_path + "afqmc/afqmc_dev_token.txt"

train_csv = pd.read_csv(open(train_path, encoding="UTF-8"), header=None, names=["text1", "text2", "label"], sep="\t")
train_label = np.array(train_csv["label"].tolist())
test_csv = pd.read_csv(open(test_path, encoding="UTF-8"), header=None, names=["text1", "text2", "label"], sep="\t")
test_label = np.array(test_csv["label"].tolist())

train_BERT_embedding_file = "E:/afqmc_features/afqmc_train_embedding.txt"
test_BERT_embedding_file = "E:/afqmc_features/afqmc_test_embedding.txt"
train_CLS_info = np.loadtxt(train_BERT_embedding_file, delimiter="\t")
test_CLS_info = np.loadtxt(test_BERT_embedding_file, delimiter="\t")
print("train shape: ", train_CLS_info.shape)
print("test shape: ", test_CLS_info.shape)
static_features = "E:/afqmc_features/afqmc_static_features.pk"
features_data = pk.load(open(static_features, 'rb'))
train_features_data = features_data[:34334]
test_features_data = features_data[34334:]
print("train shape: ", np.array(train_features_data).shape)
print("test shape: ", np.array(test_features_data).shape)

word2vector_features = "E:/afqmc_features/afqmc_glove_AveVec.pk"
word2vector_features_data = pk.load(open(word2vector_features, 'rb'))
word2vector_train_features_data = word2vector_features_data[:34334]
word2vector_test_features_data = word2vector_features_data[34334:]
print("train shape: ", np.array(word2vector_train_features_data).shape)
print("test shape: ", np.array(word2vector_test_features_data).shape)

train_merge_features_list = []
test_merge_features_list = []
print("开始合并三种特征........")
for index in range(len(train_features_data)):
    train_merge_features_list.append(train_CLS_info[index].tolist() + train_features_data[index] + word2vector_train_features_data[index])
train_merge_features_matrix = np.array(train_merge_features_list)
print("训练特征形状: ", train_merge_features_matrix.shape)
for index in range(len(test_features_data)):
    test_merge_features_list.append(test_CLS_info[index].tolist() + test_features_data[index] + word2vector_test_features_data[index])
test_merge_features_matrix = np.array(test_merge_features_list)
print("测试特征形状: ", test_merge_features_matrix.shape)
stacking_learning(train_merge_features_matrix, train_label, test_merge_features_matrix, test_label)