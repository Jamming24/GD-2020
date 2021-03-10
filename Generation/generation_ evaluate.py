# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: generation_ evaluate.py 
@time: 2020年12月05日12时08分 
"""

# from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.corpus import wordnet
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu


def convert_result_tolist(result_file, model_type="sow"):
    result_file_object = open(result_file, encoding="UTF-8", mode='r')
    result_sentence = []
    blank_line = 0
    if model_type == "sow":
        blank_line = 6
    elif model_type == "reap":
        blank_line = 1
    elif model_type == "baseline":
        blank_line = 0
    while True:
        single_sentence = []
        input_sentence = result_file_object.readline().strip()[15:].lstrip(" ").rstrip(" ")
        true_sentence = result_file_object.readline().strip()[22:].lstrip(" ").rstrip(" ")
        single_sentence.append(input_sentence)
        single_sentence.append(true_sentence)
        for i in range(blank_line):
            result_file_object.readline()
        while True:
            generated_sentence = result_file_object.readline().strip()
            if "Generated Sentence" in generated_sentence:
                single_sentence.append(generated_sentence[19:].lstrip(" ").rstrip(" "))
                result_file_object.readline()
            else:
                if "Ground Truth nll" in generated_sentence:
                    single_sentence.append(generated_sentence[17:].lstrip(" ").rstrip(" "))
                    result_sentence.append(single_sentence)
                    result_file_object.readline()
                break
        if input_sentence == '':
            break
    result_file_object.close()
    print(model_type, " senetence nums: ", len(result_sentence))
    return result_sentence


def convert_sow_reap_tolist(sow_out_file):
    # 将结果输出文件转换为tsv格式 读取sow_reap输出文件 已废除函数
    file_object = open(sow_out_file, encoding="UTF-8", mode='r')
    sentences = []
    while True:
        single_sentence = []
        input_sentence = file_object.readline().strip()[15:].lstrip(" ").rstrip(" ")
        true_sentence = file_object.readline().strip()[22:].lstrip(" ").rstrip(" ")
        single_sentence.append(input_sentence)
        single_sentence.append(true_sentence)
        file_object.readline()
        file_object.readline()
        file_object.readline()
        file_object.readline()
        file_object.readline()
        file_object.readline()
        while True:
            generated_sentence = file_object.readline().strip()
            if "Generated Sentence" in generated_sentence:
                single_sentence.append(generated_sentence[19:].lstrip(" ").rstrip(" "))
                file_object.readline()
            else:
                if "Ground Truth nll" in generated_sentence:
                    single_sentence.append(generated_sentence[17:].lstrip(" ").rstrip(" "))
                    sentences.append(single_sentence)
                    file_object.readline()
                break
        if input_sentence == '':
            break
    file_object.close()
    print("senetence nums: ", len(sentences))
    return sentences


def data_process():
    reap_list = []
    sow_list = []
    test_set = set()
    for reap in reap_list:
        for sow in sow_list:
            if sow[0] == reap[0]:
                test_set.add(reap[0])
    print(len(test_set))
    # print(test_set)
    test_list = []
    for line in reap_list:
        if line[0] in test_set:
            test_list.append(line[0])
    test_line = '基度山 问道 。 是 ， 伯爵 阁下 。'
    print(test_list.index(test_line))  # 4234

    # reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
    # reference = ["是的 ， 我 回来 了 ， 卡德鲁斯 邻居 ， 我 正想 让 你 高兴 一下 呢 。".split()]
    # candidate = [['this', 'is', 'a', 'test'], ['弯下', '也', '去', '休息,', '伯爵', '先生', '还', '有', '多么', '厉害', '。']]
    # # candidate = " 弯下 也 去 休息, 伯爵 先生 还 有 多么 厉害 。".split()
    # print(candidate)
    # score = sentence_bleu(reference, candidate)
    # print(score)
    # reap_file: 28914 条数据
    # print(int(28914*0.1))
    # s = "你 连 一 滴 水 ， 一 粒 沙 明天 正午 会 在 哪里 也 说 不 出 ； 然而 ， 你 却 以 你 的 无能 来 侮辱 太阳 ！" 130120


def get_blue_score(result_list):
    # 0:输入句子，1 表示输出句子 2-11表示生成句子 12 表示 truth nll
    # a = ['this', 'is', 'a', 'test']
    # b = ['this', 'is' 'test']
    # references = [[a, b]]
    # candidates = [['this', 'is', 'a', 'test', "bbbbb", "hhhhh"]]
    references = []
    candidates = []
    for sentence in result_list:
        single_references = [sentence[0].split(), sentence[1].split()]
        single_candidates = [line.split() for line in sentence[2:-1]]
        references.append(single_references)
        if len(single_candidates) > 2:
            candidates.append(single_candidates[0])
        else:
            candidates.append(single_candidates[0])
        truth_nll = sentence[-1]
    # corpus_bleu 预测句子 一个样本只能有一个预测句子 多了就会报错 参考句子可以有多个
    blue_score = corpus_bleu(references, candidates)
    print("blue_score: ", blue_score)
    print('Cumulative 1-gram: %f' % corpus_bleu(references, candidates, weights=(1, 0, 0, 0)))
    print('Cumulative 2-gram: %f' % corpus_bleu(references, candidates, weights=(0, 1, 0, 0)))
    print('Cumulative 3-gram: %f' % corpus_bleu(references, candidates, weights=(0, 0, 1, 0)))
    print('Cumulative 4-gram: %f' % corpus_bleu(references, candidates, weights=(0, 0, 0, 1)))


def get_rouge_score(result_rouge_list):
    # 0:输入句子，1 表示输出句子 2-11表示生成句子 12 表示 truth nll  一般都取f
    rouge_1_f = 0.0
    rouge_2_f = 0.0
    rouge_l_f = 0.0
    rouge = Rouge()
    for line in result_rouge_list:
        rouge_score = rouge.get_scores(hyps=line[2], refs=line[1])
        rouge_1_f += rouge_score[0]["rouge-1"]['f']
        rouge_2_f += rouge_score[0]["rouge-2"]['f']
        rouge_l_f += rouge_score[0]["rouge-l"]['f']
    print("rouge_1_f : ", rouge_1_f / len(result_rouge_list))
    print("rouge_2_f : ", rouge_2_f / len(result_rouge_list))
    print("rouge_l_f : ", rouge_l_f / len(result_rouge_list))
    # print(rouge_score[0]["rouge-2"])
    # print(rouge_score[0]["rouge-l"])


def get_meteor_score(result_list):

    total_meteor = 0
    for line in result_list:
        single_reference = [line[0], line[1]]
        # reference_list.append(single_reference)
        # candidate_list.append(line[2])
        score = meteor_score(single_reference, line[2], wordnet=wordnet)
        total_meteor += score
    print("meteor_score: ", total_meteor/len(result_list))


def get_Truth_nll(test_result):
    total_nll = 0
    for line in test_result:
        total_nll += float(line[-1])
    print(total_nll/len(test_result))


def get_test_data(reap_result_list, test_out_file):
    # 选择那些 索引在4234以后的,包括4234 并且没有在train_set中出现的句子
    train_set = set()
    test_set = set()
    test2 = set()
    for line in reap_result_list[0:4234]:
        train_set.add(line[0])
    for line in reap_result_list[4234:]:
        if line[0] not in train_set:
            test_set.add(line[0]+"\t"+line[1])
    for line in reap_result_list[4234:]:
        if line[0] not in train_set:
            test2.add(line[0])
    print(len(test_set))
    print(len(test2))
    for line in test_set:
        s = line.split("\t")[0]
        if s in test2:
            test2.remove(s)
        else:
            print(s)
    # test_out_file_object = open(test_out_file, encoding="UTF-8", mode='w')
    # for line in test_set:
    #     test_out_file_object.write(line+"\n")
    # test_out_file_object.close()
    # print("train set 集合: ", len(train_set))
    # print("符合标准的test set集合: ", len(test_set))


def load_test_file(file):
    data_list = []
    file_object = open(file, encoding="UTF-8", mode='r')
    for line in file_object:
        data_list.append(line.strip())
    file_object.close()
    print("test 数量: ", len(data_list))
    return data_list


# 主要评价指标 blue1 blue2 OUGE-1，ROUGE-2，ROUGE-L, meteor
root_file = "E:/2020-GD/生成/result"
test_data_file = root_file + "/test_file_all.txt"
reap_file = root_file + "/chinese_all_reap_transformer.out"
baseline_file = root_file + "/chinese_transformer_baseline.out"
reap_list = convert_result_tolist(reap_file, model_type="reap") # 4691条句子
baseline_list = convert_result_tolist(baseline_file, model_type="baseline")  # 28914
# get_test_data(reap_list, test_data_file)
reap = []
baseline = []
for line in reap_list:
    reap.append(line[0])
for line in baseline_list:
    if line[0] in reap:
        baseline.append(line)
print(len(baseline))
# sow_reap_out_file = root_file + "/chinese_all_sow_reap_transfomer.out"

# sow_list = convert_result_tolist(sow_reap_out_file, model_type="sow") # sow_reap 5818对句子
# reap_list = convert_result_tolist(reap_file, model_type="reap") # 4691条句子
# print("baseline:", len(baseline_list))
# print("reap list:", len(reap_list))

# test_data_list = load_test_file(test_data_file)
# test_baseline_list = []
# test_reap_list = []
# test_sow_list = []
# count_dict = {}
# print(reap_list)
# for line in baseline_list:
#     if line[0] in test_data_list and line[0] not in count_dict.keys():
#         test_baseline_list.append(line)
#         count_dict[line[0]] = 1
#     elif line[0] in test_data_list and line[0] in count_dict.keys():
#         continue
# count_dict.clear()

# for line in reap_list:
#     if line[0] in test_data_list and line[0] not in count_dict.keys():
#         test_reap_list.append(line)
#         count_dict[line[0]] = 1
#     elif line[0] in test_data_list and line[0] in count_dict.keys():
#         continue
# count_dict.clear()

# for line in sow_list:
#     if line[0] in test_data_list and line[0] not in count_dict.keys():
#         test_sow_list.append(line)
#     elif line[0] in test_data_list and line[0] in count_dict.keys():
#         continue
# print("test_baseline_list: ", len(test_baseline_list))
# print("test_reap_list: ", len(test_reap_list))
# print("test_sow_list: ", len(test_sow_list))
print("baseline : ")
get_blue_score(baseline)
get_meteor_score(baseline)
get_Truth_nll(baseline)
get_rouge_score(baseline)

print("reap : ")
get_blue_score(reap_list)
get_meteor_score(reap_list)
get_Truth_nll(reap_list)
get_rouge_score(reap_list)
#
# print("sow : ")
# get_blue_score(test_sow_list)
# get_meteor_score(test_sow_list)
# get_Truth_nll(test_sow_list)
# get_rouge_score(test_sow_list)


# sow
# {'f': 0.6999999952000001, 'p': 0.5833333333333334, 'r': 0.875}
# {'f': 0.2222222174691359, 'p': 0.18181818181818182, 'r': 0.2857142857142857}
# {'f': 0.39999999520000007, 'p': 0.3333333333333333, 'r': 0.5}
# 0.4388365002848248
# nll: 7.112504072687221
# reap
# candidate_list:  454
# reference_list:  454
# {'f': 0.999999995, 'p': 1.0, 'r': 1.0}
# {'f': 0.999999995, 'p': 1.0, 'r': 1.0}
# {'f': 0.999999995, 'p': 1.0, 'r': 1.0}
# 0.42693109642901744
# nll:6.809682603524236


