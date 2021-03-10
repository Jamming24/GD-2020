# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: Semantic_Interoperation.py 
@time: 2020年12月12日18时04分 
"""

import json
from tqdm import tqdm
import sys
import time
import numpy as np
import pickle as pk
from sklearn.metrics.pairwise import cosine_similarity

""" 
    1.跑通基础的设计的卷积模型
    2.计算交互矩阵，并生成中文文件保存起来，作为上一步的输入
    3.根据2产生的交互矩阵 做BERT-score类似计算，这个只需要计算test数据集合的就可以 调节阈值 观察性能 并做记录
    4.将1和2 合并实验，放到服务器上跑
"""


class LogUtil(object):
    def __init__(self):
        pass

    @staticmethod
    def log(typ, msg):
        print("[%s]\t[%s]\t%s" % (TimeUtil.t_now(), typ, str(msg)))
        sys.stdout.flush()
        return


class TimeUtil(object):
    def __init__(self):
        return

    @staticmethod
    def t_now():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    @staticmethod
    def t_now_YmdH():
        return time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))


def get_interaction_matrix_forBERT(bert_feature_json, data_file, out_interaction_matrix_file):
    # Computing_interaction_matrix_forBERT
    data_file_object = open(data_file, encoding="UTF-8", mode='r')
    data_file_list = []
    for line in data_file_object:
        data_file_list.append(line.strip().split("\t"))
    data_file_object.close()
    LogUtil.log("INFO", " 从文件 {0}   加载数据 {1} 条 ".format(data_file, len(data_file_list)))
    LogUtil.log("INFO", " 加载 BERT json 文件: {0}  ".format(bert_feature_json))

    index_layer = 0
    json_file_object = open(bert_feature_json, encoding="UTF-8", mode='r')
    bert_interaction_features = []
    for single_data in tqdm(json_file_object):
        data = json.loads(single_data)
        # 数据行号
        line_index = data["linex_index"]
        single_sentence_list = []
        single_sentence_tensor = []
        for single_token in data["features"]:
            token_name = single_token["token"]
            single_sentence_list.append(token_name)
            layers = single_token["layers"]
            index_layer = layers[0]["index"]
            tensor_values = layers[0]["values"]
            single_sentence_tensor.append(tensor_values)
        assert len(single_sentence_list) == len(single_sentence_tensor), f"第 {line_index} 张量长度出现问题"
        first_SEP_index = single_sentence_list.index("[SEP]")
        sentence_1 = "".join(single_sentence_list[1:first_SEP_index])
        sentence_2 = "".join(single_sentence_list[first_SEP_index + 1:-1])
        text_a = single_sentence_tensor[1:first_SEP_index]
        text_b = single_sentence_tensor[first_SEP_index + 1:-1]
        semantic_similarity = cosine_similarity(text_a, text_b)
        bert_interaction_features.append((semantic_similarity, data_file_list[line_index][2], [sentence_1, sentence_2]))
    assert len(bert_interaction_features) == len(data_file_list), "数据数量与生成交互特征数量不匹配"
    LogUtil.log("INFO", " 处理 BERT json 特征层数: {0} , 开始保存到pickle文件中 ".format(index_layer))
    json_file_object.close()
    out_file_object = open(out_interaction_matrix_file, mode='wb')
    pk.dump(bert_interaction_features, out_file_object)
    out_file_object.close()
    LogUtil.log("INFO", " {0} BERT 交互特征文件保存完成 ".format(data_file))


def get_interaction_matrix_forGlove(vector_file, data_file_token, data_type):
    # 在本地计算
    we_dic = {}
    f = open(vector_file, 'r', encoding="UTF-8")
    for line in f:
        subs = line.strip().split(None, 1)  # 这句写的妙啊 牛皮牛皮 1表示只对第一个进行分割
        if 2 > len(subs):
            continue
        else:
            word = subs[0]
            vec = subs[1]
        we_dic[word] = np.array([float(s) for s in vec.split()])
    f.close()
    LogUtil.log("INFO", " 加载到词向量总数: {0} ".format(len(we_dic)))
    # 这里传入的是全部token的文件 所以要在这里进行train, test文件
    Glove_interaction_features = []
    data_file_token = open(data_file_token, encoding="UTF-8", mode='r').readlines()
    for line in tqdm(data_file_token):
        text_a = []
        text_b = []
        ses = line.strip().split("\t")
        sentence_1 = ses[0]
        sentence_2 = ses[1]
        label = ses[2]
        for word_1 in sentence_1.split():
            if word_1 in we_dic.keys():
                text_a.append(we_dic[word_1])
            else:
                text_a_vec = np.array(300 * [0.])
                for character in word_1:
                    if character in we_dic.keys():
                        text_a_vec + we_dic[character]
                text_a.append(text_a_vec)
        for word_2 in sentence_2.split():
            if word_2 in we_dic.keys():
                text_b.append(we_dic[word_2])
            else:
                text_b_vec = np.array(300 * [0.])
                for character in word_2:
                    if character in we_dic.keys():
                        text_b_vec + we_dic[character]
                text_b.append(text_b_vec)
        if len(text_a) == 0:
            print(sentence_1)
        if len(text_b) == 0:
            print(sentence_2)
        semantic_similarity = cosine_similarity(text_a, text_b)
        Glove_interaction_features.append((semantic_similarity, label, [sentence_1, sentence_2]))
        text_a.clear()
        text_b.clear()
    assert len(Glove_interaction_features) == len(data_file_token), "数据与交互特征数量不符合"
    LogUtil.log("INFO", " 基于Glove的静态词表示交互特征处理完成  开始保存到pickle文件中")
    train_num = 0
    test_num = 0
    if data_type == "lcqmc":
        train_num = 238766
        test_num = 21302
    elif data_type == "afqmc":
        train_num = 34334
        test_num = 4316
    elif data_type == "SRPI":
        train_num = 8146
        test_num = 3490
    train_interaction_file = "E:/2020-GD/Semantic Interoperation/" + data_type + "_train_Glove_interaction.pk"
    test_interaction_file = "E:/2020-GD/Semantic Interoperation/" + data_type + "_test_Glove_interaction.pk"
    train_interaction = Glove_interaction_features[:train_num]
    test_interaction = Glove_interaction_features[train_num:]
    assert len(train_interaction) == train_num, "{0} 训练数据数量出现问题 ".format(data_type)
    assert len(test_interaction) == test_num, "{0} 测试数据数量出现问题 ".format(data_type)
    train_out_file_object = open(train_interaction_file, mode='wb')
    pk.dump(train_interaction, train_out_file_object)
    train_out_file_object.close()

    test_out_file_object = open(test_interaction_file, mode='wb')
    pk.dump(test_interaction, test_out_file_object)
    test_out_file_object.close()
    LogUtil.log("INFO", " 基于Glove的静态词表示交互特征保存完成")


def get_interaction_matrix_forElmo(elmo_file, all_token_file, task_type, out_interaction_floder):
    # 在服务器计算 只能按照索引计算 为了加载标签所以还是要加载原数据
    data_file_object = open(all_token_file, encoding="UTF-8", mode='r')
    data_file_list = []
    for line in data_file_object:
        data_file_list.append(line.strip().split("\t"))
    data_file_object.close()
    LogUtil.log("INFO", " 从文件 {0}   加载数据 {1} 条 ".format(all_token_file, len(data_file_list)))
    train_num = 0
    test_num = 0
    if task_type == "lcqmc":
        train_num = 238766
        test_num = 21302
    elif task_type == "afqmc":
        train_num = 34334
        test_num = 4316
    elif task_type == "SRPI":
        train_num = 8146
        test_num = 3490
    elmo_data = pk.load(open(elmo_file, mode='rb'))
    assert len(elmo_data) == (train_num + test_num) * 2, " {0} elmo 特征数量出错, elmo数量 {1}".format(task_type, len(elmo_data))
    chinese_elmo_interaction_features = []
    for index in range(int(len(elmo_data) / 2)):
        sentence_1 = data_file_list[index][0]
        sentence_2 = data_file_list[index][1]
        semantic_similarity = cosine_similarity(elmo_data[index * 2], elmo_data[(index * 2) + 1])
        chinese_elmo_interaction_features.append((semantic_similarity, data_file_list[index][2], [sentence_1, sentence_2]))

    train_interaction_file = out_interaction_floder + task_type + "_train_Elmo_interaction.pk"
    test_interaction_file = out_interaction_floder + task_type + "_test_Elmo_interaction.pk"
    train_interaction = chinese_elmo_interaction_features[:train_num]
    test_interaction = chinese_elmo_interaction_features[train_num:]
    assert len(train_interaction) == train_num, "{0} 训练数据数量出现问题 ".format(task_type)
    assert len(test_interaction) == test_num, "{0} 测试数据数量出现问题 ".format(task_type)
    LogUtil.log("INFO", " 基于Elmo的静态词表示开始写入文件")
    train_out_file_object = open(train_interaction_file, mode='wb')
    pk.dump(train_interaction, train_out_file_object)
    train_out_file_object.close()

    test_out_file_object = open(test_interaction_file, mode='wb')
    pk.dump(test_interaction, test_out_file_object)
    test_out_file_object.close()
    LogUtil.log("INFO", " 基于Elmo的静态词表示交互特征保存完成")


chinese_elmo = "E:/2020-GD/Semantic Interoperation/SRPI_chinese_elmo_embedding.pk"
out_floder = "E:/2020-GD/Semantic Interoperation/"
SRPI_all_token = "E:/2020-GD/评价数据/GD-paraphrase_identification/sow_reap_paraphrase_identification_token.txt"
# get_interaction_matrix_forElmo(chinese_elmo, SRPI_all_token, "SRPI", out_floder)
data = pk.load(open("E:/2020-GD/Semantic Interoperation/SRPI_train_Glove_interaction.pk", mode='rb'))
print(data)

# BERT 特征
# BERT_json_file = "E:/2020-GD/Semantic Interoperation/afqmc_test_embedding_-1.json"
# afqmc_test = "E:/2020-GD/评价数据/afqmc/afqmc_dev.txt"
# afqmc_BERT_interaction_file = "E:/2020-GD/Semantic Interoperation/afqmc_test_BERT_interaction.pk"
# get_interaction_matrix_forBERT(BERT_json_file, afqmc_test, afqmc_BERT_interaction_file)


# glove 特征
# glove_file = "E:/2020-GD/Indentification_Vector.txt"
# lcqmc_all_token = "E:/2020-GD/评价数据/lcqmc/all_token_simply_line.txt"
# afqmc_all_token = "E:/2020-GD/评价数据/afqmc/afqmc_all_token.txt"
# SRPI_all_token = "E:/2020-GD/评价数据/GD-paraphrase_identification/sow_reap_paraphrase_identification_token.txt"
# get_interaction_matrix_forGlove(glove_file, afqmc_all_token, "afqmc")
# get_interaction_matrix_forGlove(glove_file, SRPI_all_token, "SRPI")
# get_interaction_matrix_forGlove(glove_file, lcqmc_all_token, "lcqmc")
