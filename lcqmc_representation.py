# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: lcqmc_representation.py 
@time: 2020年11月21日15时47分 
"""

import math

import numpy as np
from numpy import linalg
import pickle as pk
from tqdm import tqdm
import sys
import time

# 使用glove 静态词向量 使用elmo动态句子向量
from elmoformanylangs import Embedder


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


def get_elmo_embedding(elmo_floder, token_file, sentence_vector_file):
    file_data = open(token_file, 'r', encoding="UTF-8")
    data_lines = file_data.readlines()
    batch_size = 32
    file_data.close()
    # 按照一句一个向量的标准保存 将一个单句存到一行 原代码返回的特征值是要相加的 return q1_features + q2_features
    sentences = []
    for line in data_lines:
        sentences.append(line.strip().split(" "))
    print("句子数量: ", len(sentences))
    elmo = Embedder(elmo_floder)
    elmo_vectors = []
    for i in tqdm(range(int(len(sentences) / batch_size) + 1)):
        sentences_curr = sentences[i * batch_size: i * batch_size + batch_size]
        embedding = elmo.sents2elmo(sentences_curr, output_layer=-1)  # 1024维度
        elmo_vectors += embedding

    assert len(sentences) == len(elmo_vectors), "len(data_lines) != len(elmo_vectors)"
    print("len(elmo_vectors): ", len(elmo_vectors))
    output_file = open(sentence_vector_file, 'wb', encoding='UTF-8')
    pk.dump(elmo_vectors, output_file)
    output_file.close()


def ElmoEmbeddingTFIDFVec(data_list, token_idf, elmo_file, tf_idf_elmo_file, dimension=1024):
    # 使用tf-idf作为每个词向量前面的权重 得到加权的相似度计算
    elmo_features = pk.load(open(elmo_file, 'rb'))
    TFIDF_elmo_embedding_file = open(tf_idf_elmo_file, 'wb')
    tf_idf_elmo_features = []
    index = 0
    for data in data_list:
        q1_words = data[0].split()
        q2_words = data[1].split()

        q1_vec = np.array(dimension * [0.])
        q2_vec = np.array(dimension * [0.])
        q1_words_cnt = {}
        q2_words_cnt = {}
        for word in q1_words:
            q1_words_cnt[word] = q1_words_cnt.get(word, 0.) + 1.
        for word in q2_words:
            q2_words_cnt[word] = q2_words_cnt.get(word, 0.) + 1.

        assert len(q1_words) == len(elmo_features[index]), f"第 {index} 行数据出现问题"
        for s_1_index in range(len(q1_words)):
            # 句子1向量
            q1_vec += token_idf.get(q1_words[s_1_index], 0.) * q1_words_cnt[q1_words[s_1_index]] * elmo_features[index][s_1_index]
        index += 1
        assert len(q2_words) == len(elmo_features[index]), f"第 {index} 行数据出现问题"
        for s_2_index in range(len(q2_words)):
            # 句子2向量
            q2_vec += token_idf.get(q2_words[s_2_index], 0.) * q2_words_cnt[q2_words[s_2_index]] * elmo_features[index][s_2_index]
        index += 1
        tf_idf_elmo_features.append(list(q1_vec) + list(q2_vec))
    LogUtil.log("INFO", "TF_IDF elmo vectors nums: len(tf_idf_elmo_features)=%d" % len(tf_idf_elmo_features))
    LogUtil.log("INFO", " 开始保存 TF_IDF elmo vectors 到文件 %s" % TFIDF_elmo_embedding_file)
    pk.dump(tf_idf_elmo_features, TFIDF_elmo_embedding_file)
    LogUtil.log("INFO", " 文件 %s 保存 完成 " % TFIDF_elmo_embedding_file)


# GloveWordEmbedding():
def generate_idf(token_file):
    file_data = open(token_file, 'r', encoding="UTF-8").readlines()
    idf = {}
    for line in file_data:
        ses = line.strip().split("\t")
        for sentence in ses:
            words = sentence.strip().split(" ")
            for word in words:
                idf[word] = idf.get(word, 0) + 1
    num_docs = len(file_data) * 2
    for word in idf:
        idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
    LogUtil.log("INFO", "IDF calculation done, len(idf)=%d" % len(idf))
    return idf


def load_word_embedding(vector_file):
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
    print("加载到词向量总数: ", len(we_dic))
    return we_dic


def load_token_file(token_file):
    data_file = open(token_file, 'r', encoding='UTF-8').readlines()
    data_list = []
    for line in data_file:
        ses = line.strip().split("\t")
        data_list.append(ses)
    return data_list


def WordEmbeddingCosValue(data_list, dimension, word_embedding_dic, cos_value_file):
    # 输出相似度特征
    cos_value_file_obj = open(cos_value_file, 'wb')
    cos_sim_list = []
    for ses in data_list:
        q1_words = ses[0].strip().split()
        q2_words = ses[1].strip().split()
        q1_vec = np.array(dimension * [0.])
        q2_vec = np.array(dimension * [0.])
        # 求得句子中的每个词向量并将其加和到一起 算是一个整句的向量了
        for word in q1_words:
            if word in word_embedding_dic:
                q1_vec = q1_vec + word_embedding_dic[word]
        for word in q2_words:
            if word in word_embedding_dic:
                q2_vec = q2_vec + word_embedding_dic[word]

        cos_sim = 0.
        q1_vec = np.mat(q1_vec)
        q2_vec = np.mat(q2_vec)
        factor = linalg.norm(q1_vec) * linalg.norm(q2_vec)
        if 1e-6 < factor:
            cos_sim = float(q1_vec * q2_vec.T) / factor
        # 返回两个句子向量的cos值
        cos_sim_list.append([cos_sim])
    print("得到相似度个数: ", len(cos_sim_list))
    pk.dump(cos_sim_list, cos_value_file_obj)
    # for cos in cos_sim_list:
    #     cos_value_file_obj.write(str(cos) + "\n")
    cos_value_file_obj.close()


def WordEmbeddingTFIDFCosValue(token_idf, data_list, dimension, word_embedding_dic, idf_cos_value_file):
    # 使用tf-idf作为每个词向量前面的权重 得到加权的相似度计算
    cos_value_file_obj = open(idf_cos_value_file, 'wb')
    cos_sim_list = []
    for data in data_list:
        q1_words = data[0].split()
        q2_words = data[1].split()

        q1_vec = np.array(dimension * [0.])
        q2_vec = np.array(dimension * [0.])
        q1_words_cnt = {}
        q2_words_cnt = {}
        for word in q1_words:
            q1_words_cnt[word] = q1_words_cnt.get(word, 0.) + 1.
        for word in q2_words:
            q2_words_cnt[word] = q2_words_cnt.get(word, 0.) + 1.

        for word in q1_words_cnt:
            # 句子1向量
            if word in word_embedding_dic:
                q1_vec += token_idf.get(word, 0.) * q1_words_cnt[word] * word_embedding_dic[word]
        for word in q2_words_cnt:
            # 句子2向量
            if word in word_embedding_dic:
                q2_vec += token_idf.get(word, 0.) * q2_words_cnt[word] * word_embedding_dic[word]

        cos_sim = 0.
        q1_vec = np.mat(q1_vec)
        q2_vec = np.mat(q2_vec)
        factor = linalg.norm(q1_vec) * linalg.norm(q2_vec)
        if 1e-6 < factor:
            cos_sim = float(q1_vec * q2_vec.T) / factor
        # 计算得到了cos矩阵
        cos_sim_list.append([cos_sim])
    print("得到基于tf-idf距离的相似度个数: ", len(cos_sim_list))
    pk.dump(cos_sim_list, cos_value_file_obj)
    # for cos in cos_sim_list:
    #     cos_value_file_obj.write(str(cos) + "\n")
    cos_value_file_obj.close()


def WordEmbeddingAveVec(data_list, dimension, word_embedding_dic, WordEmbeddingAveVec_file):
    # 直接将两个句子的所有词向量加到一起 作为句子向量 拼接作为特征
    glove_vectors = []
    for data in data_list:
        q1_words = data[0].split()
        q2_words = data[1].split()

        q1_vec = np.array(dimension * [0.])
        q2_vec = np.array(dimension * [0.])

        for word in q1_words:
            if word in word_embedding_dic:
                q1_vec += word_embedding_dic[word]
        for word in q2_words:
            if word in word_embedding_dic:
                q2_vec += word_embedding_dic[word]
        # 将两个句子向量进行拼接
        # return list(q1_vec) + list(q2_vec)
        glove_vectors.append(list(q1_vec) + list(q2_vec))
    LogUtil.log("INFO", "glove vectors nums: len(glove_vectors)=%d" % len(glove_vectors))
    WordEmbeddingAveVec_file_obj = open(WordEmbeddingAveVec_file, mode='wb')
    pk.dump(glove_vectors, WordEmbeddingAveVec_file_obj)
    print("基于相加求和的glove词向量保存完成")
    WordEmbeddingAveVec_file_obj.close()


def WordEmbeddingTFIDFAveVec(data_list, token_idf, dimension, word_embedding_dic, WordEmbeddingTFIDFAveVec_file):
    # 结合tf-idf的glove词向量的句子拼接
    glove_tfidf_vectors = []
    for data in data_list:
        q1_words = data[0].split()
        q2_words = data[1].split()

        q1_vec = np.array(dimension * [0.])
        q2_vec = np.array(dimension * [0.])

        q1_words_cnt = {}
        q2_words_cnt = {}
        for word in q1_words:
            q1_words_cnt[word] = q1_words_cnt.get(word, 0.) + 1.
        for word in q2_words:
            q2_words_cnt[word] = q2_words_cnt.get(word, 0.) + 1.

        for word in q1_words_cnt:
            if word in word_embedding_dic:
                q1_vec += token_idf.get(word, 0.) * q1_words_cnt[word] * word_embedding_dic[word]
        for word in q2_words_cnt:
            if word in word_embedding_dic:
                q2_vec += token_idf.get(word, 0.) * q2_words_cnt[word] * word_embedding_dic[word]
        # return list(q1_vec) + list(q2_vec)
        glove_tfidf_vectors.append(list(q1_vec) + list(q2_vec))
    LogUtil.log("INFO", "base TF-IDF glove vectors nums: len(glove_vectors)=%d" % len(glove_tfidf_vectors))
    WordEmbeddingAveVec_file_obj = open(WordEmbeddingTFIDFAveVec_file, mode='wb')
    pk.dump(glove_tfidf_vectors, WordEmbeddingAveVec_file_obj)
    print("基于tf-idf的glove词向量保存完成")
    WordEmbeddingAveVec_file_obj.close()


def get_lcqmc_representation_features():
    root_path = "E:/2020-GD/lcqmc_features_engineering"
    # all_token_file = "C:/Users/gaojiaming/Desktop/2020-GD/评价数据/lcqmc/all_token.txt"
    # chinese_elmo_floder = root_path + '/chinese_elmo_zhs.model'
    # elmo_vector_file = root_path + '/lcqmc_chinese_elmo_embedding.txt'
    # get_elmo_embedding(chinese_elmo_floder, all_token_file, elmo_vector_file)
    # glove_file = root_path + "/Glove_Jieba_Segment_中文维基百科训练向量_300d_vectors.txt"
    glove_file = "E:/2020-GD/Indentification_Vector.txt"
    all_token_file_simply = "E:/2020-GD/评价数据/lcqmc/all_token_simply_line.txt"
    vector_dimension = 300
    glove_vector_cos_value = root_path + "/lcqmc_glove_cos_value.pk"
    idf_glove_vector_cos_value = root_path + "/lcqmc_tfidf_glove_cos_value.pk"
    glove_vector_AveVec = root_path + "/lcqmc_glove_AveVec.pk"
    tfidf_glove_vector_AveVec = root_path + "/lcqmc_tfidf_glove_AveVec.pk"
    # tfidf_elmo_file = root_path + "/lcqmc_tfidf_elmoVector.pk"
    idf = generate_idf(all_token_file_simply)
    all_data = load_token_file(all_token_file_simply)
    glove_word_embedding = load_word_embedding(vector_file=glove_file)
    WordEmbeddingCosValue(all_data, vector_dimension, glove_word_embedding, glove_vector_cos_value)
    WordEmbeddingTFIDFCosValue(idf, all_data, vector_dimension, glove_word_embedding, idf_glove_vector_cos_value)
    WordEmbeddingAveVec(all_data, vector_dimension, glove_word_embedding, glove_vector_AveVec)
    WordEmbeddingTFIDFAveVec(all_data, idf, vector_dimension, glove_word_embedding, tfidf_glove_vector_AveVec)
    # ElmoEmbeddingTFIDFVec(all_data, idf, elmo_vector_file, tfidf_elmo_file, 1024)
    print("文本表示特征处理完成")


def get_afqmc_representation_features():
    afqmc_floder = "E:/2020-GD/afqmc_features_engineering"
    afqmc_file = "E:/2020-GD/评价数据/afqmc/afqmc_all_token.txt"
    elmo_vector_file = afqmc_floder + '/afqmc_chinese_elmo_embedding.txt'

    # all_token_file = "C:/Users/gaojiaming/Desktop/2020-GD/评价数据/lcqmc/all_token.txt"
    # chinese_elmo_floder = root_path + '/chinese_elmo_zhs.model'
    # get_elmo_embedding(chinese_elmo_floder, all_token_file, elmo_vector_file)
    glove_file = "E:/2020-GD/Indentification_Vector.txt"
    vector_dimension = 300
    glove_vector_cos_value = afqmc_floder + "/afqmc_glove_cos_value.pk"
    idf_glove_vector_cos_value = afqmc_floder + "/afqmc_tfidf_glove_cos_value.pk"
    glove_vector_AveVec = afqmc_floder + "/afqmc_glove_AveVec.pk"
    tfidf_glove_vector_AveVec = afqmc_floder + "/afqmc_tfidf_glove_AveVec.pk"
    tfidf_elmo_file = afqmc_floder + "/afqmc_tfidf_elmoVector.pk"
    idf = generate_idf(afqmc_file)
    all_data = load_token_file(afqmc_file)
    glove_word_embedding = load_word_embedding(vector_file=glove_file)
    WordEmbeddingCosValue(all_data, vector_dimension, glove_word_embedding, glove_vector_cos_value)
    WordEmbeddingTFIDFCosValue(idf, all_data, vector_dimension, glove_word_embedding, idf_glove_vector_cos_value)
    WordEmbeddingAveVec(all_data, vector_dimension, glove_word_embedding, glove_vector_AveVec)
    WordEmbeddingTFIDFAveVec(all_data, idf, vector_dimension, glove_word_embedding, tfidf_glove_vector_AveVec)
    # ElmoEmbeddingTFIDFVec(all_data, idf, elmo_vector_file, tfidf_elmo_file, 1024)
    print("文本表示特征处理完成")


def get_SPRI_representation_features():
    SRPI_floder = "E:/2020-GD/SRPI_features_engineering"
    SRPI_file = "E:/2020-GD/评价数据/GD-paraphrase_identification/sow_reap_paraphrase_identification_token.txt"
    # elmo_vector_file = SRPI_floder + '/afqmc_chinese_elmo_embedding.txt'
    # all_token_file = "C:/Users/gaojiaming/Desktop/2020-GD/评价数据/lcqmc/all_token.txt"
    # chinese_elmo_floder = root_path + '/chinese_elmo_zhs.model'
    # get_elmo_embedding(chinese_elmo_floder, all_token_file, elmo_vector_file)
    glove_file = "E:/2020-GD/Indentification_Vector.txt"
    vector_dimension = 300
    glove_vector_cos_value = SRPI_floder + "/SRPI_glove_cos_value.pk"
    idf_glove_vector_cos_value = SRPI_floder + "/SRPI_tfidf_glove_cos_value.pk"
    glove_vector_AveVec = SRPI_floder + "/SRPI_glove_AveVec.pk"
    tfidf_glove_vector_AveVec = SRPI_floder + "/SRPI_tfidf_glove_AveVec.pk"
    tfidf_elmo_file = SRPI_floder + "/SRPI_tfidf_elmoVector.pk"
    idf = generate_idf(SRPI_file)
    all_data = load_token_file(SRPI_file)
    glove_word_embedding = load_word_embedding(vector_file=glove_file)
    WordEmbeddingCosValue(all_data, vector_dimension, glove_word_embedding, glove_vector_cos_value)
    WordEmbeddingTFIDFCosValue(idf, all_data, vector_dimension, glove_word_embedding, idf_glove_vector_cos_value)
    WordEmbeddingAveVec(all_data, vector_dimension, glove_word_embedding, glove_vector_AveVec)
    WordEmbeddingTFIDFAveVec(all_data, idf, vector_dimension, glove_word_embedding, tfidf_glove_vector_AveVec)
    # ElmoEmbeddingTFIDFVec(all_data, idf, elmo_vector_file, tfidf_elmo_file, 1024)
    print("文本表示特征处理完成")


if __name__ == '__main__':
    get_lcqmc_representation_features()


