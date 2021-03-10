# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: lcqmc_baseline.py 
@time: 2020年11月04日11时03分 
"""
# 毕设基本思路:
# 第一章做可控释义文本生成
# 第二章做多特征释义文本判别
# 第三章可能联合以上两章同时训练 (也可能换成其他思路)
import pandas as pd
import os
import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
""" 
释义文本生成baseline实验 
1.实验数据分析，统计 包括标签分布，句子长度（各个句子长度区间的数量）
2.实验数据预处理，Text Cleaning(去掉标点符号), 去除停用词
3.特征选择 cos相似度既能做特征 又能做baseline
4.基础模型选择（LR，SVM，Xgbooost） 余弦相似度 (这步未完成)
5.评价指标 P，R F1-score
"""

data_floder = "C:/Users/gaojiaming/Desktop/2020-GD_2020/评价数据"
train_file = data_floder + "/lcqmc/train.txt"
dev_file = data_floder + "/lcqmc/dev.txt"
test_file = data_floder + "/lcqmc/test.txt"


def load_stopwords():  # 设置停用词
    stop_words = []
    if not os.path.exists(data_floder + '/stopwords/hit_stopwords.txt'):
        print('未发现停用词表！')
    else:
        stop_words = [line.strip() for line in
                      open(data_floder + '/stopwords/hit_stopwords.txt', encoding='UTF-8').readlines()]
    return stop_words


def load_lcqmc_data(lcqmc_file):
    # 2.实验数据预处理
    print("数据处理.......")
    f = open(lcqmc_file, 'r', encoding='UTF-8')
    all_lines = f.readlines()
    stopwords = load_stopwords()
    sentences_1 = []
    sentences_2 = []
    for line in tqdm(all_lines):
        lines = line.strip().split('\t')
        s_1 = lines[0]
        s_2 = lines[1]
        # 先分词 后去掉停用词，会使得标点符号前后的词自然分开
        sts_1 = list(jieba.cut(s_1.strip(), cut_all=False))
        sts_2 = list(jieba.cut(s_2.strip(), cut_all=False))
        s_1_no_stop_list = []  # 去停用词后
        s_2_no_stop_list = []
        for w in sts_1:
            if w not in stopwords:
                s_1_no_stop_list.append(w)
        sentences_1.append(" ".join(s_1_no_stop_list))
        for w in sts_2:
            if w not in stopwords:
                s_2_no_stop_list.append(w)
        sentences_2.append(" ".join(s_2_no_stop_list))
    f.close()
    return sentences_1, sentences_2


def Cos_Similarity(data_file, threshold):
    # 使用余弦相似度方法 仅仅需要测试数据进行计算即可，不需要用训练数据
    data_csv = pd.read_csv(data_file, sep='\t', header=None)
    data_csv.columns = ['Sentence_1', 'Sentence_2', 'label']
    true_label = data_csv['label'].values.tolist()
    sentence_1, sentence_2 = load_lcqmc_data(data_file)
    all_sentence = sentence_1 + sentence_2
    vectorized = TfidfVectorizer(min_df=0, sublinear_tf=True)
    tf_idf_vectorized = vectorized.fit(all_sentence)
    sentence_1_tf_idf_vectorized = normalize(tf_idf_vectorized.transform(sentence_1))
    sentence_2_tf_idf_vectorized = normalize(tf_idf_vectorized.transform(sentence_2))
    cos_matrix = cosine_similarity(X=sentence_1_tf_idf_vectorized, Y=sentence_2_tf_idf_vectorized)
    predict_label = []
    for index in range(len(true_label)):
        if cos_matrix[index][index] >= threshold:
            predict_label.append(1)
        else:
            predict_label.append(0)
    accuracy = accuracy_score(y_true=true_label, y_pred=predict_label)
    f1_s = f1_score(y_true=true_label, y_pred=predict_label)
    precision_s = precision_score(y_true=true_label, y_pred=predict_label)
    recall_s = recall_score(y_true=true_label, y_pred=predict_label)
    return accuracy, f1_s, precision_s, recall_s


# auc, f1, precision, recall = Cos_Similarity(dev_file, 0.2)
# print("accuracy_score", auc, sep='\t')
# print("f1_score", f1, sep='\t')
# print("precision_score", precision, sep='\t')
# print("recall_score", recall, sep='\t')

file_object = open("C:/Users/gaojiaming/Desktop/text.txt", 'w', encoding='UTF-8')
file_object.write("sssss" + "\n")


def statical_data(train_file):
    # 1.统计数据规模
    train_data = pd.read_csv(train_file, sep='\t', header=None)
    train_data.columns = ['Sentence_1', 'Sentence_2', 'label']  # [238766 rows x 3 columns] (1    138574) (0    100192)
    print("train_data:", train_data.shape, sep="\t")
    # print(train_data['label'].value_counts())
    # dev_data = pd.read_csv(dev_file, sep='\t', header=None)
    # dev_data.columns = ['Sentence_1', 'Sentence_2', 'label']  # (8802, 3)  (1    4402)  (0    4400)
    # print("dev_data:", dev_data.shape, sep='\t')
    # print(dev_data['label'].value_counts())
    # test_data = pd.read_csv(test_file, sep='\t', header=None)
    # test_data.columns = ['Sentence_1', 'Sentence_2', 'label']  # (12500, 3)  (1    6250)  (0    6250)
    # print("test_data", test_data.shape, sep='\t')
    # print(test_data['label'].value_counts())
    # 统计句子长度
    # 如果将三个数据合并进行统计 启动此行
    # all_data = pd.concat([train_data, dev_data, test_data], axis=0, ignore_index=True)
    # print(all_data["label"].value_counts())  # (1    149226)   (0    110842)
    # print(all_data.shape)  # [260068 rows x 3 columns]
    # 最大句子长度 131 所以可划分为几个区间 (0-32], (32-64], (64-128]; (128-max_len]
    seq_length_24 = 0
    seq_length_54 = 0
    seq_length_64 = 0
    seq_length_max_len = 0
    max_len = 0
    min_len = 100
    for row in train_data.iterrows():
        sentence_len = len(row[1]["Sentence_1"]+row[1]["Sentence_2"])
        max_len = max(sentence_len, max_len)
        min_len = min(sentence_len, min_len)
        if 0 < sentence_len <= 64:
            seq_length_64 += 1
        # if 0 < sentence_len <= 24:
        #     seq_length_24 += 1
        # elif 24 < sentence_len <= 54:
        #     seq_length_54 += 1
        # elif sentence_len > 54:
        #     seq_length_max_len += 1
    print("句子长度(0-64]:", seq_length_64, seq_length_64 / train_data.shape[0], sep="\t")
    # print("句子长度(0-24]:", seq_length_24, seq_length_24/train_data.shape[0], sep="\t")  # 7079
    # print("句子长度(24-54]:", seq_length_54, seq_length_54/train_data.shape[0], sep="\t")  # 3756
    # print("句子长度(54-max]:", seq_length_max_len, seq_length_max_len/train_data.shape[0], sep="\t")  # 0
    # print("最大句子长度:", max_len, sep="\t")  # 160
    # print("最短句子长度:", min_len, sep="\t")
    # for row in train_data.iterrows():
    #     sentence_len = len(row[1]["Sentence_1"]+row[1]["Sentence_2"])
    #     max_len = max(sentence_len, max_len)
    #     min_len = min(sentence_len, min_len)
    #     if 0 < sentence_len <= 8:
    #         seq_length_8 += 1
    #     elif 8 <= sentence_len <= 16:
    #         seq_length_16 += 1
    #     elif 16 < sentence_len <= 24:
    #         seq_length_24 += 1
    #     elif 24 < sentence_len <= 32:
    #         seq_length_32 += 1
    #     elif 32 < sentence_len <= 64:
    #         seq_length_64 += 1
    #     elif 64 < sentence_len <= 128:
    #         seq_length_128 += 1
    #     elif sentence_len > 128:
    #         seq_length_max_len += 1
    # print("句子长度(0-8]:", seq_length_8, sep="\t")  # 147582
    # print("句子长度(8-16]:", seq_length_16, sep="\t")  # 332637
    # print("句子长度(16-24]:", seq_length_24, sep="\t")  # 29079
    # print("句子长度(24-32]:", seq_length_32, sep="\t")  # 7079
    # print("句子长度(32-64]:", seq_length_64, sep="\t")  # 3756
    # print("句子长度(64-128]:", seq_length_128, sep="\t")  # 0
    # print("句子长度(128-max_len]:", seq_length_max_len, sep="\t")  # 3
    # print("最大句子长度:", max_len, sep="\t")  # 160
    # print("最短句子长度:", min_len, sep="\t")


statical_data("E:/2020-GD/评价数据/afqmc/afqmc_all.txt")