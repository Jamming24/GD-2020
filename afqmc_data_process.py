# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: afqmc_data_process.py 
@time: 2020年12月07日17时29分 
"""

import os
import json
import jieba


def json_convert_text():
    afqmc_object = open("E:/2020-GD/蚂蚁金融NLP竞赛_afqmc_public/afqmc_all.json", encoding="UTF-8", mode='r')
    afqmc_text = open("E:/2020-GD/评价数据/afqmc_all.txt", encoding="UTF-8", mode='w')
    for line in afqmc_object:
        data = json.loads(line.strip())
        # print(data["sentence1"], data["sentence2"], data["label"], sep="\t")
        afqmc_text.write(data["sentence1"]+"\t"+data["sentence2"]+"\t"+data["label"]+"\n")
    afqmc_text.close()
    afqmc_object.close()


jieba.load_userdict("E:/2020-GD/评价数据/afqmc_dict.txt")


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


def stopwords(stop_word_file):
    stops = []
    if not os.path.exists(stop_word_file):
        print('未发现停用词表！')
    else:
        stops = [line.strip() for line in open(stop_word_file, encoding='UTF-8').readlines()]
    return stops


def jieba_token_file_forPower(file, token_file, stop_list):
    # 要去除停用词
    file_data = open(file, 'r', encoding="UTF-8")
    data_lines = file_data.readlines()
    file_data.close()
    token_file_object = open(token_file, 'w', encoding='UTF-8')
    for line in data_lines:
        ses = line.strip().split("\t")
        sentence_1_list = []
        sentence_2_list = []
        for word in jieba.cut(ses[0]):
            if word not in stop_list:
                sentence_1_list.append(word)
        for word in jieba.cut(ses[1]):
            if word not in stop_list:
                sentence_2_list.append(word)
        sentence_1 = ' '.join(sentence_1_list)
        sentence_2 = ' '.join(sentence_2_list)
        token_file_object.write(sentence_1+"\t"+sentence_2+"\t"+ses[2]+"\n")
    token_file_object.close()
    print("分词完成")


afqmc_floder = "E:/2020-GD/afqmc_features_engineering"
stop_file = afqmc_floder + "/cn_stopwords.txt"
jieba_token_file("E:/2020-GD/评价数据/afqmc_all.txt", "E:/2020-GD/评价数据/afqmc_all_token.txt")
stop_list = stopwords(stop_file)
jieba_token_file_forPower("E:/2020-GD/评价数据/afqmc_all.txt", "E:/2020-GD/afqmc_features_engineering/afqmc_train_forPower.txt", stop_list)
