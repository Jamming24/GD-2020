# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: afqmc_F1_score.py 
@time: 2021年01月15日11时27分 
"""
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def predict_file_check():
    GPU_file_1 = "C:/Users/gaojiaming/Desktop/test_results_GPU_1.tsv"
    GPU_file_2 = "C:/Users/gaojiaming/Desktop/test_results_GPU_2.tsv"
    CPU_file = "C:/Users/gaojiaming/Desktop/test_results_CPU.tsv"
    GPU_file_1_object = open(GPU_file_1, mode='r')
    GPU_file_2_object = open(GPU_file_2, mode='r')
    CPU_file_object = open(CPU_file, mode='r')
    GPU_1_list = []
    GPU_2_list = []
    CPU_list = []
    for line in GPU_file_1_object:
        values = line.strip().split('\t')
        GPU_1_list.append([float(values[0]), float(values[1])])
    for line in GPU_file_2_object:
        values = line.strip().split('\t')
        GPU_2_list.append([float(values[0]), float(values[1])])
    for line in CPU_file_object:
        values = line.strip().split('\t')
        CPU_list.append([float(values[0]), float(values[1])])
    GPU_1_list_argmax = np.argmax(GPU_1_list, axis=1)
    GPU_2_list_argmax = np.argmax(GPU_2_list, axis=1)
    CPU_list_argmax = np.argmax(CPU_list, axis=1)
    assert "预测数量校验通过", len(GPU_1_list) == len(GPU_2_list) == len(CPU_list)
    print(len(GPU_2_list))
    print(GPU_1_list_argmax)
    print(GPU_2_list_argmax)
    print("比较两GPU")
    for index in range(len(GPU_2_list_argmax)):
        if GPU_1_list_argmax[index] != GPU_2_list_argmax[index]:
            print(index, "\t", GPU_1_list[index])
    print("比较GPU2与CPU")
    for index in range(len(GPU_2_list_argmax)):
        if CPU_list_argmax[index] != GPU_2_list_argmax[index]:
            print(CPU_list[index])
    GPU_file_1_object.close()
    GPU_file_2_object.close()
    CPU_file_object.close()

    afqmc_true_label = "E:/2020-GD/评价数据/afqmc/afqmc_dev.txt"
    afqmc_label = []
    afqmc_object = open(afqmc_true_label, mode='r', encoding="UTF-8")
    for line in afqmc_object:
        afqmc_label.append(int(line.strip().split('\t')[2]))
    afqmc_object.close()
    print("GPU_1 accuracy: ", '\t', accuracy_score(y_true=afqmc_label, y_pred=GPU_1_list_argmax))
    print("GPU_2 accuracy: ", '\t', accuracy_score(y_true=afqmc_label, y_pred=GPU_2_list_argmax))
    print("CPU accuracy: ", '\t', accuracy_score(y_true=afqmc_label, y_pred=CPU_list_argmax))

    print("GPU_1 report: ")
    print(classification_report(y_true=afqmc_label, y_pred=GPU_1_list_argmax))
    print(f1_score(y_true=afqmc_label, y_pred=GPU_1_list_argmax))
    print(precision_score(y_true=afqmc_label, y_pred=GPU_1_list_argmax))
    print(recall_score(y_true=afqmc_label, y_pred=GPU_1_list_argmax))

    print("GPU_2 report: ")
    print(classification_report(y_true=afqmc_label, y_pred=GPU_2_list_argmax))
    print(f1_score(y_true=afqmc_label, y_pred=GPU_2_list_argmax))
    print(precision_score(y_true=afqmc_label, y_pred=GPU_2_list_argmax))
    print(recall_score(y_true=afqmc_label, y_pred=GPU_2_list_argmax))

    print("CPU report: ")
    print(classification_report(y_true=afqmc_label, y_pred=CPU_list_argmax))
    print(f1_score(y_true=afqmc_label, y_pred=CPU_list_argmax))
    print(precision_score(y_true=afqmc_label, y_pred=CPU_list_argmax))
    print(recall_score(y_true=afqmc_label, y_pred=CPU_list_argmax))


def Print_Metrics(predict_file, test_file):
    predict_object = open(predict_file, mode='r')
    predict_list = []
    for line in predict_object:
        values = line.strip().split('\t')
        predict_list.append([float(values[0]), float(values[1])])
    predict_list = np.argmax(predict_list, axis=1)
    predict_object.close()
    afqmc_label = []
    afqmc_object = open(test_file, mode='r', encoding="UTF-8")
    for line in afqmc_object:
        afqmc_label.append(int(line.strip().split('\t')[2]))
    afqmc_object.close()
    print("classification_report: ")
    print(classification_report(y_true=afqmc_label, y_pred=predict_list, digits=5))
    print("precision_score: ", '\t', precision_score(y_true=afqmc_label, y_pred=predict_list))
    print("recall_score: ", '\t', recall_score(y_true=afqmc_label, y_pred=predict_list))
    print("f1_score: ", '\t', f1_score(y_true=afqmc_label, y_pred=predict_list))
    print("accuracy: ", '\t', accuracy_score(y_true=afqmc_label, y_pred=predict_list))


# 0:2978 1:1338
# baseline_predict = "C:/Users/gaojiaming/Desktop/afqmc_0.7344_baseline_output_test_results.tsv"
static_predict = "C:/Users/gaojiaming/Desktop/test_results.tsv"
afqmc_true_label = "E:/afqmc_features/afqmc/afqmc_dev.txt"
Print_Metrics(static_predict, afqmc_true_label)
# predict_file_check()
