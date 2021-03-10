# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: lcqmc_data_process.py
@time: 2020年11月22日13时48分 
"""
# 对数据进行分词处理
# 应该可以使用与斯坦福不同的分析工具 反正最后都是映射成向量

import os
import jieba
import pickle as pk
import numpy as np
import pandas as pd


def jieba_token_file(file, token_file):
    file_data = open(file, 'r', encoding="UTF-8")
    data_lines = file_data.readlines()
    file_data.close()
    token_file_object = open(token_file, 'w', encoding='UTF-8')
    for line in data_lines:
        ses = line.strip().split("\t")
        sentence_1 = ' '.join(jieba.cut(ses[0]))
        sentence_2 = ' '.join(jieba.cut(ses[1]))
        token_file_object.write(sentence_1+"\t"+sentence_2+"\n")
    token_file_object.close()
    print("分词完成")


def split_data_features(features_file, output_folder, features_name):
    offset = [238766, 8802, 12500]
    features = pk.load(open(features_file, 'rb'))

    train_data = features[0: offset[0]]
    assert len(train_data) == offset[0], "train_data 数量未对齐"
    dev_data = features[offset[0]: offset[0] + offset[1]]
    assert len(dev_data) == offset[1], "dev_data 数量未对齐"
    test_data = features[offset[0] + offset[1]:]
    assert len(test_data) == offset[2], "test_data 数量未对齐"
    train_out = open(os.path.join(output_folder, features_name + '_train.pk'), 'wb')
    pk.dump(train_data, train_out)
    dev_out = open(os.path.join(output_folder, features_name + '_dev.pk'), 'wb')
    pk.dump(dev_data, dev_out)
    test_out = open(os.path.join(output_folder, features_name + '_test.pk'), 'wb')
    pk.dump(test_data, test_out)
    print(features_name, "特征文件数据切分完成")


def features_NAN_check(feature_file):
    feature = pk.load(open(feature_file, 'rb'))
    feature = pd.DataFrame(feature)
    if not np.any(feature.isnull()):
        print(feature_file, " 文件通过校验 ")
    else:
        print(feature_file, " 数据中含有空值, 无法通过校验 ")


def fileToPK(in_file, out_file):
    feature_value = []
    in_file_object = open(in_file, encoding="UTF-8", mode='r')
    out_file_object = open(out_file, mode='wb')
    for line in in_file_object:
        feature_value.append(float(line.strip()))
    pk.dump(feature_value, out_file_object)
    print("总行数: ", len(feature_value))
    in_file_object.close()
    out_file_object.close()


def elmoPKtoText():
    chinese_elmo_Value = pk.load(open("./lcqmc_feature_forBERT/lcqmc_chinese_elmo_embedding_features_train.pk", 'rb'))
    print(chinese_elmo_Value[:2])
    save_object = open("./lcqmc_chinese_elmo_embedding_features_train.txt", encoding="UTF-8", mode='w')
    for line in tqdm(chinese_elmo_Value):
        new_list = [str(x) for x in line]
        save_object.write(" ".join(new_list) + "\n")
    save_object.close()
    print(len(chinese_elmo_Value))
    print("保存成功")


if __name__ == '__main__':
    # 把特征文件处理成train， dev test三种文件
    all_file = "C:/Users/gaojiaming/Desktop/2020-GD/评价数据/lcqmc/all.txt"
    all_token_file = "C:/Users/gaojiaming/Desktop/2020-GD/评价数据/lcqmc/all_token_simply_line.txt"
    # jieba_token_file(all_file, all_token_file)
    out_floder = "/data3/gaojiaming/2020-GD/Identification/BERT_withFeaturesEngineering/lcqmc_feature_forBERT"
    features_file = "/data3/gaojiaming/2020-GD/Identification/feature_engineering/lcqmc_static_features.pk"
    # split_data_features(features_file, out_floder, "lcqmc_static_features")
    # features_file = "/data3/gaojiaming/2020-GD/Identification/feature_engineering/lcqmc_glove_AveVec.pk"
    # split_data_features(features_file, out_floder, "lcqmc_glove_AveVec_features")
    features_NAN_check("C:/Users/gaojiaming/Desktop/2020-GD/判别/feature_engineering/lcqmc_tfidf_glove_cos_value.pk")

    # in_feature_file = "C:/Users/gaojiaming/Desktop/2020-GD/判别/feature_engineering/lcqmc_glove_cos_value.txt"
    # out_feature_file = "C:/Users/gaojiaming/Desktop/2020-GD/判别/feature_engineering/lcqmc_glove_cos_value.pk"
    # fileToPK(in_feature_file, out_feature_file)

    # in_feature_file = "C:/Users/gaojiaming/Desktop/2020-GD/判别/feature_engineering/lcqmc_tfidf_glove_cos_value.txt"
    # out_feature_file = "C:/Users/gaojiaming/Desktop/2020-GD/判别/feature_engineering/lcqmc_tfidf_glove_cos_value.pk"
    # fileToPK(in_feature_file, out_feature_file)


