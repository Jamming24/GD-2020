# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: lcqmc_NLP.py 
@time: 2020年11月22日19时42分 
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from sklearn.metrics import classification_report
from GD_2020.Identification.utils import LogUtil
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
import pickle as pk


def stopwords(stop_word_file):
    stops = []
    if not os.path.exists(stop_word_file):
        print('未发现停用词表！')
    else:
        stops = [line.strip() for line in open(stop_word_file, encoding='UTF-8').readlines()]
    return stops


def load_file(all_data_file):
    data_file = open(all_data_file, 'r', encoding='UTF-8').readlines()
    data_list = []
    for line in data_file:
        ses = line.strip().split("\t")[0:2]
        data_list.append(ses)
    return data_list


def ProcessPOSTag(stanford_core_nlp, data_list_no_token, stanford_token_file, stanford_postag_file):
    # 这里改成你stanford-corenlp所在的目录
    nlp = StanfordCoreNLP(stanford_core_nlp, lang='zh')
    stanford_token = []
    stanford_postag = []
    for data in tqdm(data_list_no_token):
        sentence_1_token = nlp.word_tokenize(data[0])
        sentence_1_POS = nlp.pos_tag(data[0])
        simply_POS_1 = []
        for POS in sentence_1_POS:
            simply_POS_1.append(POS[1])
        sentence_2_token = nlp.word_tokenize(data[1])
        sentence_2_POS = nlp.pos_tag(data[1])
        simply_POS_2 = []
        for POS in sentence_2_POS:
            simply_POS_2.append(POS[1])
        stanford_token.append([" ".join(sentence_1_token), " ".join(sentence_2_token)])
        stanford_postag.append([" ".join(simply_POS_1), " ".join(simply_POS_2)])
    nlp.close()
    assert len(data_list_no_token) == len(stanford_token), "token数量出现问题"
    assert len(data_list_no_token) == len(stanford_postag), "POStag数量出现问题"
    stanford_token_file_object = open(stanford_token_file, mode='w', encoding="UTF-8")
    for line in stanford_token:
        stanford_token_file_object.write(line[0] + "\t" + line[1] + "\n")
    stanford_token_file_object.close()
    LogUtil.log("INFO",
                "Stanford Token Writer to stanford_token_file done, len(stanford_token)=%d" % len(stanford_token))
    stanford_postag_file_object = open(stanford_postag_file, mode='w', encoding="UTF-8")
    for line in stanford_postag:
        stanford_postag_file_object.write(line[0] + "\t" + line[1] + "\n")
    stanford_postag_file_object.close()
    LogUtil.log("INFO",
                "Stanford POStag Writer to stanford_postag_file done, len(stanford_postag)=%d" % len(stanford_postag))


class POSTagCount(object):
    # postag标签特征 这个应该是NLP中的句法分析书的特征值吧 该特征可以映射成低维度稠密向量，原特征要处理成类似于one-hot的格式
    @staticmethod
    def load_postag(all_POStag_file):
        # load data set from disk
        all_POStag = open(all_POStag_file, mode='r', encoding="UTF-8").readlines()
        postag = {}

        # 应该是将所有的pos tag映射到字典里
        for line in all_POStag:
            POSTag_Sentence = line.strip().split("\t")
            q1_postag = POSTag_Sentence[0].split(" ")
            for pos in q1_postag:
                postag.setdefault(pos, len(postag))

            q2_postag = POSTag_Sentence[1].split(" ")
            for pos in q2_postag:
                postag.setdefault(pos, len(postag))
        print(postag)
        return postag, all_POStag

    def __init__(self, all_POStag_file):
        self.postag, self.all_POStag = POSTagCount.load_postag(all_POStag_file)


    def extract_features(self, POStag_features_file):
        all_POStag_features = []
        for pos_data in self.all_POStag:
            POSTag_Sentence = pos_data.strip().split("\t")
            q1_vec = len(self.postag) * [0]
            q1_postag = POSTag_Sentence[0].split(" ")
            for s in q1_postag:
                postag_id = self.postag[s]
                q1_vec[postag_id] += 1
            q2_vec = len(self.postag) * [0]
            q2_postag = POSTag_Sentence[1].split(" ")
            for s in q2_postag:
                postag_id = self.postag[s]
                q2_vec[postag_id] += 1

            q1_vec = np.array(q1_vec)
            q2_vec = np.array(q2_vec)
            sum_vec = q1_vec + q2_vec
            sub_vec = abs(q1_vec - q2_vec)
            dot_vec = q1_vec.dot(q2_vec)
            q1_len = np.sqrt(q1_vec.dot(q1_vec))
            q2_len = np.sqrt(q2_vec.dot(q2_vec))
            cos_sim = 0.
            if q1_len * q2_len > 1e-6:
                cos_sim = dot_vec / q1_len / q2_len
            all_POStag_features.append(
                list(q1_vec) + list(q2_vec) + list(sum_vec) + list(sub_vec) + [np.sqrt(dot_vec), q1_len, q2_len,
                                                                               cos_sim])
        LogUtil.log("INFO",
                    "all_data_file POStag features Writer to stanford_postag_features_file done, len(all_POStag_features)=%d" % len(
                        all_POStag_features))
        assert len(all_POStag_features) == len(self.all_POStag), "POStag 特征数量与原数据不一致"
        # for line in all_POStag_features[300:405]:
        #     print(line)
        POStag_features_file_object = open(POStag_features_file, mode='wb')
        pk.dump(all_POStag_features, POStag_features_file_object)
        POStag_features_file_object.close()
        LogUtil.log("INFO",
                    "POStag features num=%d, POStag feature Writer file done " % (len(self.postag) * 4 + 4))  # 140个特征


def getLDA_features(stop_word, all_file_with_label, features_num, features_file):
    # 获取LAD特征 LDA 受到分词效果的影响巨大
    all_data_frame = pd.read_csv(open(all_file_with_label, mode='r', encoding="UTF-8"), sep="\t", header=None)
    label = all_data_frame[2]
    data_list = []
    for index, line in tqdm(all_data_frame.iterrows()):
        data_list.append(line[0])
        data_list.append(line[1])
    LogUtil.log("INFO", "加载数据总量 = %d, file done " % (len(data_list) / 2))

    cntVector = CountVectorizer(stop_words=stop_word)
    cntTf = cntVector.fit_transform(data_list)
    # print(cntVector.vocabulary_)
    LogUtil.log("INFO", "开始训练LDA模型 ")
    lda = LatentDirichletAllocation(n_components=features_num, learning_method='online', batch_size=128, learning_offset=20., random_state=0, max_iter=10)
    docres = lda.fit_transform(cntTf).tolist()
    data_features_list = []
    for index in tqdm(range(int(len(docres)/2))):
        line_features = docres[index*2] + docres[index*2+1]
        data_features_list.append(line_features)
    LogUtil.log("INFO", "特征长度=%d, file done " % (len(data_features_list[0])))
    LogUtil.log("INFO", "数据总理=%d, file done " % (len(data_features_list)))

    log_reg = LogisticRegression(class_weight='balanced')
    log_reg.fit(data_features_list[0:238766], label[0:238766])
    predict_y = log_reg.predict(data_features_list[238766: 238766 + 8802])
    print(classification_report(label[238766: 238766 + 8802], predict_y))
    LDA_features_file_object = open(features_file, mode='wb')
    pk.dump(data_features_list, LDA_features_file_object)
    LDA_features_file_object.close()
    LogUtil.log("INFO", "Writer LDA features into file done ")


def get_lcqmc_NLP_features():
    root_path = "C:/Users/gaojiaming/Desktop/2020-GD/判别"
    all_file = "C:/Users/gaojiaming/Desktop/2020-GD/评价数据/lcqmc/all_token_simply_line.txt"  # all_token_simply_line
    StanfordCoreNLP_floder = "C:/Users/gaojiaming/Desktop/2020-GD/生成/sow-reap-paraphrasing-master/stanford-corenlp-full-2018-02-27"
    Stanford_token_file = "C:/Users/gaojiaming/Desktop/2020-GD/评价数据/lcqmc/all_token_Stanford.txt"
    Stanford_POS_file = root_path + "/all_POSTag_Stanford.txt"
    all_POStag_features_file = root_path + "/feature_engineering/lcqmc_POStag_features.pk"
    all_LDA_features_file = root_path + "/feature_engineering/lcqmc_LDA_features.pk"
    stop_file = root_path + "/cn_stopwords.txt"
    all_data = load_file(all_file)
    # ProcessPOSTag(StanfordCoreNLP_floder, all_data, Stanford_token_file, Stanford_POS_file)
    # POSTagCount(Stanford_POS_file).extract_features(all_POStag_features_file)
    stop_list = stopwords(stop_file)
    getLDA_features(stop_list, all_file, 10, all_LDA_features_file)


def get_afqmc_NLP_features():
    StanfordCoreNLP_floder = "E:/2020-GD/生成/sow-reap-paraphrasing-master/stanford-corenlp-full-2018-02-27"
    afqmc_floder = "E:/2020-GD/afqmc_features_engineering"
    afqmc_file = "E:/2020-GD/评价数据/afqmc_all_token.txt"
    stop_file = afqmc_floder + "/cn_stopwords.txt"

    Stanford_POS_file = afqmc_floder + "/afqmc_all_POSTag_Stanford.txt"
    Stanford_token_file = afqmc_floder + "/afqmc_all_token_Stanford.txt"
    all_POStag_features_file = afqmc_floder + "/afqmc_POStag_features.pk"
    all_LDA_features_file = afqmc_floder + "/afqmc_LDA_features.pk"
    all_data = load_file(afqmc_file)
    ProcessPOSTag(StanfordCoreNLP_floder, all_data, Stanford_token_file, Stanford_POS_file)
    POSTagCount(Stanford_POS_file).extract_features(all_POStag_features_file)
    # 132维度的特征
    # stop_list = stopwords(stop_file)
    # getLDA_features(stop_list, all_file, 10, all_LDA_features_file)


if __name__ == '__main__':
    StanfordCoreNLP_floder = "E:/2020-GD/生成/sow-reap-paraphrasing-master/stanford-corenlp-full-2018-02-27"
    SRPI_floder = "E:/2020-GD/SRPI_features_engineering"
    SRPI_file = "E:/2020-GD/评价数据/GD-paraphrase_identification/sow_reap_paraphrase_identification_token.txt"
    stop_file = SRPI_floder + "/cn_stopwords.txt"

    Stanford_POS_file = SRPI_floder + "/SRPI_all_POSTag_Stanford.txt"
    Stanford_token_file = SRPI_floder + "/SRPI_all_token_Stanford.txt"
    all_POStag_features_file = SRPI_floder + "/SRPI_POStag_features.pk"
    # all_LDA_features_file = SRPI_floder + "/SRPI_LDA_features.pk"
    all_data = load_file(SRPI_file)
    ProcessPOSTag(StanfordCoreNLP_floder, all_data, Stanford_token_file, Stanford_POS_file)
    POSTagCount(Stanford_POS_file).extract_features(all_POStag_features_file)
    # 132维度的特征
    # stop_list = stopwords(stop_file)
    # getLDA_features(stop_list, all_file, 10, all_LDA_features_file)

