# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: data_shuffle.py 
@time: 2021年03月02日09时50分 
"""
import random

file = "C:/Users/gaojiaming/Desktop/lcqmc_chinese_reap.order"
out_file = "C:/Users/gaojiaming/Desktop/lcqmc_chinese_reap_positive_2.order"
out_object = open(out_file, encoding="UTF-8", mode='w')
file_object = open(file, encoding="UTF-8", mode='r')
data_list = []
while True:
    single_sentence = []
    input_sentence = file_object.readline()
    true_sentence = file_object.readline()
    order_1 = file_object.readline()
    order_2 = file_object.readline()
    single_sentence.append(input_sentence)
    single_sentence.append(true_sentence)
    single_sentence.append(order_1)
    single_sentence.append(order_2)
    file_object.readline()
    if input_sentence == '':
        break
    data_list.append(single_sentence)

# 首先要把 0标签数据获取得到 原数据仅仅对train文件进行了 处理 只需要加载train的数据就行
negative_sample = []
afqmc_train_file = "E:/afqmc_features/afqmc/afqmc_train.txt"
lcqmc_train_file = "E:/lcqmc_features/lcqmc/train.txt"
afqmc_object = open(afqmc_train_file, encoding="UTF-8", mode='r')
lcqmc_object = open(lcqmc_train_file, encoding="UTF-8", mode='r')
# for line in afqmc_object:
#     lines = line.strip("\n").split("\t")
#     if lines[2] == '1':
#         negative_sample.append([lines[0], lines[1]])
# print(len(negative_sample)) # 23761
for line in lcqmc_object:
    lines = line.strip("\n").split("\t")
    if lines[2] == '1':
        negative_sample.append([lines[0], lines[1]]) # 100192
print(len(negative_sample)) # 123953
afqmc_object.close()
lcqmc_object.close()
# 找到数据中 句子对 标记为零的样例 去除掉
# random.shuffle(data_list)
# random.shuffle(data_list)
print(len(data_list)) # 247600 # 去除负样本数量:  108051
file_object.close()
negative_index = 0
from tqdm import tqdm
for line in tqdm(data_list):
    soure_sentence_1 = line[0].strip("\n").replace(" ", "")
    soure_sentence_2 = line[1].strip("\n").replace(" ", "")
    for negative in negative_sample:
        if soure_sentence_1 == negative[0] and soure_sentence_2 == negative[1]:
            # print("###", negative[0])
            negative_index += 1
            for sentence in line:
                out_object.write(sentence)
            out_object.write("\n")
            break
out_object.close()
print("正样本数量: ", negative_index)
print("在 哪里 有 卖 欢天喜地 七 仙女 灵石 ？".replace(" ", ""))