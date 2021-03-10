# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: Data_Augmentation.py 
@time: 2021年01月14日09时25分 
"""
import random


def load_generation_data(data_file):
    file_data = open(data_file, encoding="UTF-8", mode='r')
    result_sentence = []
    blank_line = 1
    while True:
        single_sentence = []
        input_sentence = file_data.readline().strip()[15:].lstrip(" ").rstrip(" ")
        true_sentence = file_data.readline().strip()[22:].lstrip(" ").rstrip(" ")
        single_sentence.append(input_sentence)
        single_sentence.append(true_sentence)
        for i in range(blank_line):
            file_data.readline()
        while True:
            generated_sentence = file_data.readline().strip()
            if "Generated Sentence" in generated_sentence:
                single_sentence.append(generated_sentence[19:].lstrip(" ").rstrip(" "))
                file_data.readline()
            else:
                if "Ground Truth nll" in generated_sentence:
                    single_sentence.append(generated_sentence[17:].lstrip(" ").rstrip(" "))
                    result_sentence.append(single_sentence)
                    file_data.readline()
                break
        if input_sentence == '':
            break
    file_data.close()
    print("senetence nums: ", len(result_sentence))
    return result_sentence


def load_train_file(train_file):
    train_object = open(train_file, encoding='UTF-8', mode='r')
    train_dict = {}
    train_list = []
    for line in train_object:
        datas = line.strip("\n").split("\t")
        train_dict[datas[0]+datas[1]] = datas
        train_list.append(line.strip("\n"))
    train_object.close()
    return train_dict, train_list


def load_train_negative_file(train_file):
    train_object = open(train_file, encoding='UTF-8', mode='r')
    train_negative_list = []
    for line in train_object:
        datas = line.strip("\n").split("\t")
        if datas[2] == "0":
            train_negative_list.append(line)
    train_object.close()
    return train_negative_list


# 21738句
# 提取出 负例样本
# lcqmc_data = load_train_negative_file("E:/lcqmc_features/lcqmc/train.txt")
# lcqmc_negative = "E:/lcqmc_features/lcqmc/train_negative_sample.txt"
# file_object = open(lcqmc_negative, encoding="UTF-8", mode='w')
# for line in lcqmc_data:
#     file_object.write(line)
# file_object.close()

# afqmc_data = load_train_negative_file("E:/afqmc_features/afqmc/afqmc_train.txt")
# afqmc_negative = "E:/afqmc_features/afqmc/afqmc_train_negative_sample.txt"
# file_object = open(afqmc_negative, encoding="UTF-8", mode='w')
# for line in afqmc_data:
#     file_object.write(line)
# file_object.close()


afqmc_generation_positive_file = "E:/实验三-新/afqmc_reap_transformer_train_shuffle_positive_2.pt-26.out"
afqmc_generation_positive_sentence = load_generation_data(afqmc_generation_positive_file)
print(len(afqmc_generation_positive_sentence))

lcqmc_generation_positive_file = "E:/实验三-新/lcqmc_reap_transformer_train_shuffle_positive_2.pt-26.out"
lcqmc_generation_positive_sentence = load_generation_data(lcqmc_generation_positive_file)
print(len(lcqmc_generation_positive_sentence))

# afqmc_data_augmentation_1 = []
# afqmc_data_augmentation_2 = []
# for line in afqmc_generation_positive_sentence:
#     generation_sentence_1 = line[0].replace(" ", "") + "\t" + line[2].replace(" ", "") + "\t" + "1\n"
#     generation_sentence_2 = line[1].replace(" ", "") + "\t" + line[3].replace(" ", "") + "\t" + "1\n"
#     afqmc_data_augmentation_1.append(generation_sentence_1)
#     afqmc_data_augmentation_2.append(generation_sentence_2)
# afqmc_data_augmentation_writer_1 = open("E:/实验三-新/afqmc_reap_train_sentence_1_generate.out", encoding="UTF-8", mode='w')
# afqmc_data_augmentation_writer_2 = open("E:/实验三-新/afqmc_reap_train_sentence_2_generate.out", encoding="UTF-8", mode='w')
# for line in afqmc_data_augmentation_1:
#     afqmc_data_augmentation_writer_1.write(line)
# afqmc_data_augmentation_writer_1.close()
#
# for line in afqmc_data_augmentation_2:
#     afqmc_data_augmentation_writer_2.write(line)
# afqmc_data_augmentation_writer_2.close()


# lcqmc_data_augmentation_1 = []
# lcqmc_data_augmentation_2 = []
# for line in lcqmc_generation_positive_sentence:
#     generation_sentence_1 = line[0].replace(" ", "") + "\t" + line[2].replace(" ", "") + "\t" + "1\n"
#     generation_sentence_2 = line[1].replace(" ", "") + "\t" + line[3].replace(" ", "") + "\t" + "1\n"
#     lcqmc_data_augmentation_1.append(generation_sentence_1)
#     lcqmc_data_augmentation_2.append(generation_sentence_2)
# lcqmc_data_augmentation_writer_1 = open("E:/实验三-新/lcqmc_reap_train_sentence_1_generate.out", encoding="UTF-8", mode='w')
# lcqmc_data_augmentation_writer_2 = open("E:/实验三-新/lcqmc_reap_train_sentence_2_generate.out", encoding="UTF-8", mode='w')
# for line in lcqmc_data_augmentation_1:
#     lcqmc_data_augmentation_writer_1.write(line)
# lcqmc_data_augmentation_writer_1.close()
# for line in lcqmc_data_augmentation_2:
#     lcqmc_data_augmentation_writer_2.write(line)
# lcqmc_data_augmentation_writer_2.close()

# 保持1:1 shuffle
# name = "afqmc"
# negative_file = "E:/afqmc_features/afqmc/train_negative_sample.txt"
# generate_file = "E:/实验三-新/afqmc_reap_train_sentence_1_generate.out"
# banlance_file = "E:/实验三-新/afqmc_reap_train_sentence_1_generate_add_negative_banlance.out"
# negative_object = open(negative_file, encoding="UTF-8", mode='r')
# generate_object = open(generate_file, encoding="UTF-8", mode='r')
# banlance_augment = []
# for line in generate_object:
#     banlance_augment.append(line)
# negative_list = []
# for line in negative_object:
#     negative_list.append(line)
# if name == "lcqmc":
#     newlist = random.sample(negative_list, 83241)
#     banlance_augment = banlance_augment + newlist
#
# if name == "afqmc":
#     newlist = random.sample(negative_list, 6239)
#     banlance_augment = banlance_augment + newlist
#
# print(len(banlance_augment))
# random.shuffle(banlance_augment)
# balance_out = open(banlance_file, encoding="UTF-8", mode="w")
# for line in banlance_augment:
#     balance_out.write(line)
# balance_out.close()

# random.shuffle(afqmc_data_augmentation)
data_list = []
data_object = open("E:/实验三-新/afqmc_sentence_1_add_all_negaivte.txt", mode='r', encoding="UTF-8")
for line in data_object:
    data_list.append(line)
data_object.close()
random.shuffle(data_list)
writer_data = open("E:/实验三-新/afqmc_sentence_1_add_all_negaivte_shuffle.txt", mode='w', encoding='UTF-8')
for line in data_list:
    writer_data.write(line)
writer_data.close()

