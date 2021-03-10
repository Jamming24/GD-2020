# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: select_1_label.py 
@time: 2021年01月22日10时03分 
"""


def load_parse_tree(order_file):
    parse_order_file = open(order_file, encoding='UTF-8', mode='r')
    sentences = []
    while True:
        sentence_1 = parse_order_file.readline().strip()  # 第一句文本
        sentence_2 = parse_order_file.readline().strip()  # 第二句文本
        reorder1 = parse_order_file.readline().strip()  # 第一个句子序列
        reorder2 = parse_order_file.readline().strip()  # 第二个句子序列
        parse_order_file.readline()
        if sentence_1 == '':
            break
        sentence_1 = sentence_1.split(' ')
        sentence_2 = sentence_2.split(' ')
        sentences.append((sentence_1, sentence_2, reorder1, reorder2))
    print("加载句对数量: ", len(sentences))
    parse_order_file.close()
    return sentences


afqmc_order_file = "E:/实验三/afqmc_chinese_reap.order"
afqmc_train = "E:/2020-GD/评价数据/afqmc/afqmc_train.txt"
afqmc_order_sentence = load_parse_tree(afqmc_order_file)
train_object = open(afqmc_train, encoding='UTF-8', mode='r')
positive_list = [] # 10573 个正例
negtive_list = []
for line in train_object:
    datas = line.strip("\n").split("\t")
    if datas[2] == '1':
        positive_list.append(datas[0]+datas[1])
    elif datas[2] == '0':
        negtive_list.append(datas[0]+datas[1])
train_object.close()
print("得到正例: ", len(positive_list))
print("得到负例: ", len(negtive_list))
afqmc_positive_object = open("E:/实验三/afqmc_positive_reap.order", encoding="UTF-8", mode='w')
afqmc_negtive_object = open("E:/实验三/afqmc_negtive_reap.order", encoding="UTF-8", mode='w')
positive_count = 0
negtive_count = 0
for line in afqmc_order_sentence:
    sentence_1 = "".join(line[0])
    sentence_2 = "".join(line[1])
    sentence = sentence_1 + sentence_2
    if sentence in positive_list:
        print(sentence)
        positive_count += 1
        afqmc_positive_object.write(" ".join(line[0])+"\n")
        afqmc_positive_object.write(" ".join(line[1])+"\n")
        afqmc_positive_object.write(line[2] + "\n")
        afqmc_positive_object.write(line[3] + "\n")
        afqmc_positive_object.write("\n")
    elif sentence in negtive_list:
        print(sentence)
        negtive_count += 1
        afqmc_negtive_object.write(" ".join(line[0])+"\n")
        afqmc_negtive_object.write(" ".join(line[1])+"\n")
        afqmc_negtive_object.write(line[2] + "\n")
        afqmc_negtive_object.write(line[3] + "\n")
        afqmc_negtive_object.write("\n")
afqmc_positive_object.close()
afqmc_negtive_object.close()
print("正列 order: ", positive_count)
print("负列 order: ", negtive_count)
# 正列 order:  8578 / 10573
# 负列 order:  19337 / 23761
