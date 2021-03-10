# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: Uniml.py 
@time: 2021年03月07日21时06分 
"""
import random


def load_train_file(train_file, type="1"):
    train_object = open(train_file, encoding='UTF-8', mode='r')
    train_list = []
    for line in train_object:
        datas = line.strip("\n").split("\t")
        if datas[2] == type:
            train_list.append([datas[0], datas[1]])
    train_object.close()
    return train_list


def load_Uniml_generate_data(data_file):
    generate_object = open(data_file, encoding='UTF-8', mode='r')
    generate_list = []
    for line in generate_object:
        generate_list.append(line.strip("\n"))
    generate_object.close()
    return generate_list


afqmc_train = "E:/afqmc_features/afqmc/afqmc_train.txt"
afqmc_train_Unilm = "E:/afqmc_features/afqmc/afqmc_train_Unilm.txt"
afqmc_generate_file = "E:/实验三-新/predict.json"
train_data = load_train_file(afqmc_train, '1')
negative_data = load_train_file(afqmc_train, '0')
afqmc_generate = load_Uniml_generate_data(afqmc_generate_file)
print(len(train_data))
print("negative: ", len(negative_data))
afqmc_augment = []
augment_data = "E:/实验三-新/afqmc_uniml_augmentation_all_data_shuffle.json"
augment_data_object = open(augment_data, mode='w', encoding="UTF-8")
for index in range(len(train_data)):
    # augment_data_object.write(train_data[index][0] + "\t" + afqmc_generate[index] + "\t1\n")
    # augment_data_object.write(train_data[index][0] + "\t" + train_data[index][1]+"\t1\n")
    afqmc_augment.append(train_data[index][0] + "\t" + afqmc_generate[index].replace(" ", "") + "\t1\n")
    afqmc_augment.append(train_data[index][0] + "\t" + train_data[index][1]+"\t1\n")
for line in negative_data:
    afqmc_augment.append(line[0] + '\t' + line[1] + '\t0\n')
random.shuffle(afqmc_augment)
for line in afqmc_augment:
    augment_data_object.write(line)

augment_data_object.close()
# out_object = open(afqmc_train_Unilm, mode='w', encoding="UTF-8")
# for line in train_data:
#     out_object.write("{'src_text':\""+line[0]+"\",'tgt_text':\""+line[1]+"\"}\n")
# out_object.close()