# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: lcqmc_statistics.py 
@time: 2020年11月23日14时04分 
"""

import math

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
from sklearn import preprocessing
from tqdm import tqdm
from sklearn.metrics.pairwise import linear_kernel
import pickle as pk

from GD_2020.Identification.utils import MISSING_VALUE_NUMERIC
from GD_2020.Identification.utils import NgramUtil, DistanceUtil, LogUtil, MathUtil


def stopwords(stop_word_file):
    stops = []
    if not os.path.exists(stop_word_file):
        print('未发现停用词表！')
    else:
        stops = [line.strip() for line in open(stop_word_file, encoding='UTF-8').readlines()]
    return stops


def load_token_file(token_file):
    data_file = open(token_file, 'r', encoding='UTF-8').readlines()
    data_list = []
    for line in data_file:
        ses = line.strip().split("\t")
        data_list.append(ses)
    return data_list


def WordMatchShare(data_list, stop_word_list):
    # 统计句子1和句子2的词相同词的数量 作为特征 特征数量1
    static_WordMatchShare = []
    for line in data_list:
        q1words = {}
        q2words = {}
        for word in line[0].split():
            if word not in stop_word_list:
                q1words[word] = q1words.get(word, 0) + 1
        for word in line[1].split():
            if word not in stop_word_list:
                q2words[word] = q2words.get(word, 0) + 1
        n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
        n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
        n_tol = sum(q1words.values()) + sum(q2words.values())
        if 1e-6 > n_tol:
            static_WordMatchShare.append([0.])
        else:
            static_WordMatchShare.append([1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol])
    LogUtil.log("INFO", "句子1和句子2的词相同词的数量 特征数量1 WordMatchShare features, len(static_WordMatchShare)=%d" % len(static_WordMatchShare))
    return static_WordMatchShare


def generate_idf(data_list):
    idf = {}
    q_set = set()
    for data in data_list:
        if data[0] not in q_set:
            q_set.add(data[0])
            words = data[0].split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
        if data[1] not in q_set:
            q_set.add(data[1])
            words = data[1].split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
    num_docs = len(data_list)
    for word in idf:
        idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
    LogUtil.log("INFO", "idf calculation done, len(idf)=%d" % len(idf))
    return idf


def TFIDFWordMatchShare(data_list, tf_idf):
    # 词共现的基础上乘以tf-idf 特征数量1
    tfidf_WordMatch = []
    for data in data_list:
        q1words = {}
        q2words = {}
        for word in data[0].split():
            q1words[word] = q1words.get(word, 0) + 1
        for word in data[1].split():
            q2words[word] = q2words.get(word, 0) + 1
        sum_shared_word_in_q1 = sum([q1words[w] * tf_idf.get(w, 0) for w in q1words if w in q2words])
        sum_shared_word_in_q2 = sum([q2words[w] * tf_idf.get(w, 0) for w in q2words if w in q1words])
        sum_tol = sum(q1words[w] * tf_idf.get(w, 0) for w in q1words) + sum(
            q2words[w] * tf_idf.get(w, 0) for w in q2words)
        if 1e-6 > sum_tol:
            tfidf_WordMatch.append([0.])
        else:
            tfidf_WordMatch.append([1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol])
    LogUtil.log("INFO", "词共现的基础上乘以tf-idf 特征数量1 TFIDFWordMatchShare features, len(tfidf_WordMatch)=%d" % len(tfidf_WordMatch))
    return tfidf_WordMatch


def Length(data_list):
    # 句子长度 句子字符长度和句子分词长度 get_feature_num = 4
    sentence_length_list = []
    for data in data_list:
        fs = list()
        fs.append(len(data[0]))
        fs.append(len(data[1]))
        fs.append(len(data[0].split()))
        fs.append(len(data[1].split()))
        sentence_length_list.append(fs)
    LogUtil.log("INFO", "句子长度 句子字符长度和句子分词长度 get_feature_num = 4, Length features, len(sentence_length_list)=%d" % len(sentence_length_list))
    min_max_scaler = preprocessing.MinMaxScaler((0, 1))
    sentence_length_list = min_max_scaler.fit_transform(sentence_length_list)
    return sentence_length_list.tolist()


def LengthDiff(data_list):
    # 句子长度差 get_feature_num = 1
    length_diff = []
    for data in data_list:
        length_diff.append([abs(len(data[0]) - len(data[1]))])
    LogUtil.log("INFO", "句子长度差 get_feature_num = 1, LengthDiff features, len(length_diff)=%d" % len(length_diff))
    min_max_scaler = preprocessing.MinMaxScaler((0, 1))
    length_diff = min_max_scaler.fit_transform(length_diff)
    return length_diff.tolist()


def LengthDiffRate(data_list):
    # 两个句子长度比 get_feature_num = 1
    length_diff_rate_list = []
    for data in data_list:
        len_q1 = len(data[0])
        len_q2 = len(data[1])
        if max(len_q1, len_q2) < 1e-6:
            length_diff_rate_list.append([0.0])
        else:
            length_diff_rate_list.append([1.0 * min(len_q1, len_q2) / max(len_q1, len_q2)])
    LogUtil.log("INFO", "两个句子长度比 get_feature_num = 1, LengthDiffRate features, len(length_diff_rate_list)=%d" % len(length_diff_rate_list))
    return length_diff_rate_list


class PowerfulWord(object):
    # 没弄懂
    @staticmethod
    def load_powerful_word(fp):
        powful_word = []
        f = open(fp, 'r', encoding="UTF-8")
        for line in f:
            subs = line.split('\t')
            word = subs[0]
            stats = [float(num) for num in subs[1:]]
            powful_word.append((word, stats))
        f.close()
        return powful_word
    @staticmethod
    def save_powerful_word(words_power, fp):
        f = open(fp, 'w', encoding="UTF-8")
        for ele in words_power:
            f.write("%s" % ele[0])
            for num in ele[1]:
                f.write("\t%.5f" % num)
            f.write("\n")
        f.close()
        print("powerful_word 已经保存到文件:", fp)

    @staticmethod
    def generate_powerful_word(train_data_file):
        """
        计算数据中词语的影响力，格式如下：
            词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
        """
        train_data = open(train_data_file, encoding="UTF-8", mode='r').readlines()
        # 使用分词过的带标签数据
        words_power = {}
        for data in train_data:
            # print(data)
            sens = data.strip().split("\t")
            label = int(sens[2])
            q1_words = sens[0].split()
            q2_words = sens[1].split()
            all_words = set(q1_words + q2_words)
            q1_words = set(q1_words)
            q2_words = set(q2_words)
            for word in all_words:
                if word not in words_power:
                    words_power[word] = [0. for i in range(7)]
                # 计算出现语句对数量
                words_power[word][0] += 1.
                words_power[word][1] += 1.

                if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                    # 计算单侧语句数量
                    words_power[word][3] += 1.
                    if 0 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算单侧语句正确比例
                        words_power[word][4] += 1.
                if (word in q1_words) and (word in q2_words):
                    # 计算双侧语句数量
                    words_power[word][5] += 1.
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算双侧语句正确比例
                        words_power[word][6] += 1.
        for word in words_power:
            # 计算出现语句对比例
            words_power[word][1] /= len(train_data)
            # 计算正确语句对比例
            words_power[word][2] /= words_power[word][0]
            # 计算单侧语句对正确比例
            if words_power[word][3] > 1e-6:
                words_power[word][4] /= words_power[word][3]
            # 计算单侧语句对比例
            words_power[word][3] /= words_power[word][0]
            # 计算双侧语句对正确比例
            if words_power[word][5] > 1e-6:
                words_power[word][6] /= words_power[word][5]
            # 计算双侧语句对比例
            words_power[word][5] /= words_power[word][0]
        sorted_words_power = sorted(words_power.items(), key=lambda d: d[1][0], reverse=True)
        LogUtil.log("INFO", "power words calculation done, len(words_power)=%d" % len(sorted_words_power))
        return sorted_words_power


class PowerfulWordDoubleSide():
    # powerful_word_fp 是上一个类的输出文件

    def __init__(self, powerful_word_fp, thresh_num=500, thresh_rate=0.9):
        self.pword = PowerfulWord.load_powerful_word(powerful_word_fp)
        self.pword_dside = PowerfulWordDoubleSide.init_powerful_word_dside(self.pword, thresh_num, thresh_rate)

    @staticmethod
    def init_powerful_word_dside(pword, thresh_num, thresh_rate):
        pword_dside = []
        pword = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)
        pword_sort = sorted(pword, key=lambda d: d[1][6], reverse=True)
        pword_dside.extend(map(lambda x: x[0], filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))
        LogUtil.log('INFO', 'Double side power words(%d): %s' % (len(pword_dside), str(pword_dside)))
        return pword_dside

    def extract_all_features(self, data_list):
        all_data_tags = []
        for data in data_list:
            tags = []
            q1_words = data[0].split()
            q2_words = data[1].split()
            for word in self.pword_dside:
                if (word in q1_words) and (word in q2_words):
                    tags.append(1.0)
                else:
                    tags.append(0.0)
            all_data_tags.append(tags)
        LogUtil.log("INFO", "PowerfulWordDoubleSide get_feature_num = %d, len(all_data_tags)=%d" % (len(self.pword_dside), len(all_data_tags)))
        return all_data_tags


class PowerfulWordOneSide():

    def __init__(self, powerful_word_fp, thresh_num=500, thresh_rate=0.9):
        self.pword = PowerfulWord.load_powerful_word(powerful_word_fp)
        self.pword_oside = PowerfulWordOneSide.init_powerful_word_oside(self.pword, thresh_num, thresh_rate)

    @staticmethod
    def init_powerful_word_oside(pword, thresh_num, thresh_rate):
        pword_oside = []
        pword = filter(lambda x: x[1][0] * x[1][3] >= thresh_num, pword)
        pword_oside.extend(
            map(lambda x: x[0], filter(lambda x: x[1][4] >= thresh_rate, pword)))
        # LogUtil.log('INFO', 'One side power words(%d): %s' % (len(pword_oside), str(pword_oside)))
        return pword_oside

    def extract_all_features(self, data_list):
        all_data_tags = []
        for data in data_list:
            tags = []
            q1_words = data[0].split()
            q2_words = data[1].split()
            for word in self.pword_oside:
                if (word in q1_words) and (word not in q2_words):
                    tags.append(1.0)
                elif (word not in q1_words) and (word in q2_words):
                    tags.append(1.0)
                else:
                    tags.append(0.0)
            all_data_tags.append(tags)
        LogUtil.log("INFO", "PowerfulWordOneSide get_feature_num = %d, len(all_data_tags)=%d" % (len(self.pword_oside), len(all_data_tags)))
        return all_data_tags


class PowerfulWordDoubleSideRate():
    # get_feature_num = 1
    def __init__(self, powerful_word_fp):
        self.pword_dict = dict(PowerfulWord.load_powerful_word(powerful_word_fp))

    def extract_all_features(self, data_list):
        all_data_rate = []
        num_least = 300
        rate = [1.0]
        for data in data_list:
            q1_words = set(data[0].split())
            q2_words = set(data[1].split())
            share_words = list(q1_words.intersection(q2_words))
            for word in share_words:
                if word not in self.pword_dict:
                    continue
                if self.pword_dict[word][0] * self.pword_dict[word][5] < num_least:
                    continue
                rate[0] *= (1.0 - self.pword_dict[word][6])
            rate = [1 - num for num in rate]
            all_data_rate.append(rate)
        LogUtil.log("INFO", "PowerfulWordDoubleSideRate get_feature_num = 1, len(all_data_rate)=%d" % len(all_data_rate))
        return all_data_rate


class PowerfulWordOneSideRate(object):
    #  get_feature_num = 1
    def __init__(self, powerful_word_fp):
        self.pword_dict = dict(PowerfulWord.load_powerful_word(powerful_word_fp))

    def extract_all_features(self, data_list):
        all_data_rate = []
        num_least = 300
        rate = [1.0]
        for data in data_list:
            q1_words = data[0].split()
            q2_words = data[0].split()
            q1_diff = list(set(q1_words).difference(set(q2_words)))
            q2_diff = list(set(q2_words).difference(set(q1_words)))
            all_diff = set(q1_diff + q2_diff)
            for word in all_diff:
                if word not in self.pword_dict:
                    continue
                if self.pword_dict[word][0] * self.pword_dict[word][3] < num_least:
                    continue
                rate[0] *= (1.0 - self.pword_dict[word][4])
            rate = [1 - num for num in rate]
            all_data_rate.append(rate)
        LogUtil.log("INFO", "PowerfulWordOneSideRate get_feature_num = 1, len(all_data_rate)=%d" % len(all_data_rate))
        return all_data_rate


class TFIDF(object):
    # tf-idf值的求和特征，并不是纯tf-idf特征 6
    def __init__(self, token_data_file, stops_list):
        self.tfidf = self.init_tfidf(token_data_file, stops_list)

    @staticmethod
    def init_tfidf(token_data_file, stops_list):
        all_data_frame = pd.read_csv(open(token_data_file, encoding="UTF-8"), error_bad_lines=False, sep="\t", header=None)
        tfidf = TfidfVectorizer(stop_words=stops_list, ngram_range=(1, 1))
        tfidf_txt = pd.Series(all_data_frame[0].tolist() + all_data_frame[1].tolist()).astype(str)
        tfidf.fit_transform(tfidf_txt)
        LogUtil.log("INFO", "init tfidf done ")
        return tfidf

    def extract_features(self, data_list):
        tf_idf_statics_features = []
        for data in tqdm(data_list):
            q1_tf_idf = self.tfidf.transform([data[0]])
            q2_tf_idf = self.tfidf.transform([data[1]])
            fs = list()
            fs.append(np.sum(q1_tf_idf.data))
            fs.append(np.sum(q2_tf_idf.data))
            fs.append(len(q1_tf_idf.data))
            fs.append(len(q2_tf_idf.data))
            cosine_similarities = linear_kernel(q1_tf_idf, q2_tf_idf).flatten()
            fs.append(cosine_similarities[0])
            tf_idf_statics_features.append(fs)
        LogUtil.log("INFO", "tf-idf值的求和特征，并不是纯tf-idf特征 TFIDF get_feature_num = 6, len(tf_idf_statics_features)=%d" % len(tf_idf_statics_features))
        return tf_idf_statics_features


def NgramJaccardCoef(data_list):
    # n-gram jaccard系数特征 get_feature_num = 4
    all_jaccard = []
    for data in data_list:
        q1_words = data[0].split()
        q2_words = data[1].split()
        fs = list()
        for n in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n)
            q2_ngrams = NgramUtil.ngrams(q2_words, n)
            fs.append(DistanceUtil.jaccard_coef(q1_ngrams, q2_ngrams))
        all_jaccard.append(fs)
    LogUtil.log("INFO", "n-gram jaccard系数特征 NgramJaccardCoef get_feature_num = 4, len(all_jaccard)=%d" % len(all_jaccard))
    return all_jaccard


def NgramDiceDistance(data_list):
    # DiceDistance距离 get_feature_num = 4
    all_DiceDistance = []
    for data in data_list:
        q1_words = data[0].split()
        q2_words = data[1].split()
        fs = list()
        for n in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n)
            q2_ngrams = NgramUtil.ngrams(q2_words, n)
            fs.append(DistanceUtil.dice_dist(q1_ngrams, q2_ngrams))
        all_DiceDistance.append(fs)
    LogUtil.log("INFO", "DiceDistance距离 NgramDiceDistance get_feature_num = 4, len(all_DiceDistance)=%d" % len(all_DiceDistance))
    return all_DiceDistance


def NgramDistance(data_list):
    # get_feature_num = 4*5 = 20
    distance_func = getattr(DistanceUtil, 'edit_dist')
    all_NgramDistance = []
    for data in tqdm(data_list):
        q1_words = data[0].split()
        q2_words = data[1].split()
        fs = list()
        aggregation_modes_outer = ["mean", "max", "min", "median"]
        aggregation_modes_inner = ["mean", "std", "max", "min", "median"]
        for n_ngram in range(1, 4):
            q1_ngrams = NgramUtil.ngrams(q1_words, n_ngram)
            q2_ngrams = NgramUtil.ngrams(q2_words, n_ngram)
            val_list = list()
            for w1 in q1_ngrams:
                _val_list = list()
                for w2 in q2_ngrams:
                    s = distance_func(w1, w2)
                    _val_list.append(s)
                if len(_val_list) == 0:
                    _val_list = [MISSING_VALUE_NUMERIC]
                val_list.append(_val_list)
            if len(val_list) == 0:
                val_list = [[MISSING_VALUE_NUMERIC]]

            for mode_inner in aggregation_modes_inner:
                tmp = list()
                for l in val_list:
                    tmp.append(MathUtil.aggregate(l, mode_inner))
                fs.extend(MathUtil.aggregate(tmp, aggregation_modes_outer))
        all_NgramDistance.append(fs)
    LogUtil.log("INFO", "NgramDistance距离 NgramDistance get_feature_num = 4*5, len(all_NgramDistance)=%d" % len(all_NgramDistance))
    return all_NgramDistance


def LevenshteinDistance(data_list):
    # 编辑距离 4个特征
    all_levenshtein_feature = []
    for data in tqdm(data_list):
        every_levenshtein = []
        levenshtein_ratio = fuzz.ratio(data[0], data[1]) / 100
        levenshtein_partial_ratio = fuzz.partial_ratio(data[0], data[1]) / 100
        levenshtein_token_sort_ratio = fuzz.token_sort_ratio(data[0], data[1]) / 100
        levenshtein_set_ratio = fuzz.token_set_ratio(data[0], data[1]) / 100
        every_levenshtein.append(levenshtein_ratio)
        every_levenshtein.append(levenshtein_partial_ratio)
        every_levenshtein.append(levenshtein_token_sort_ratio)
        every_levenshtein.append(levenshtein_set_ratio)
        all_levenshtein_feature.append(every_levenshtein)
    LogUtil.log("INFO", "LevenshteinDistance距离 get_feature_num = 4, len(all_levenshtein_feature)=%d" % len(all_levenshtein_feature))
    return all_levenshtein_feature


def get_lcqmc_statistics_features():
    root_path = "E:/2020-GD/lcqmc_features_engineering"
    all_token_file_simply = "E:/2020-GD/评价数据/lcqmc/all_token_simply_line.txt"  # all_token_simply_line
    all_static_features_file = root_path + "/lcqmc_static_features.pk"
    stop_file = root_path + "/cn_stopwords.txt"
    train_file_for_power = "E:/2020-GD/评价数据/lcqmc/train_forPower.txt"
    words_power_file = root_path + "/train_power_file.txt"
    all_data = load_token_file(all_token_file_simply)
    stop_list = stopwords(stop_file)
    tf_idf = generate_idf(all_data)
    print("开始生成基于统计的文本特征.......")
    word_match_share_features = WordMatchShare(all_data, stop_list)
    tf_idf_wordMatch_share_features = TFIDFWordMatchShare(all_data, tf_idf)
    length_features = Length(all_data)
    length_diff_features = LengthDiff(all_data)
    length_diff_rate_features = LengthDiffRate(all_data)
    print("PowerfulWord features........")
    words_power = PowerfulWord().generate_powerful_word(train_file_for_power)
    PowerfulWord().save_powerful_word(words_power, words_power_file)
    powerful_word_double_side_features = PowerfulWordDoubleSide(words_power_file, 500, 0.75).extract_all_features(
        all_data)
    powerful_word_one_side_features = PowerfulWordOneSide(words_power_file, 500, 0.75).extract_all_features(all_data)
    powerful_word_double_rate_features = PowerfulWordDoubleSideRate(words_power_file).extract_all_features(all_data)
    powerful_Word_One_rate_features = PowerfulWordOneSideRate(words_power_file).extract_all_features(all_data)
    print("mean TF-IDF......")
    mean_tf_idf_features = TFIDF(all_token_file_simply, stop_list).extract_features(all_data)
    print("Distance features.........")
    Jaccard_features = NgramJaccardCoef(all_data)
    dice_features = NgramDiceDistance(all_data)
    distance_features = NgramDistance(all_data)
    leve_distance_features = LevenshteinDistance(all_data)
    assert len(word_match_share_features) == len(tf_idf_wordMatch_share_features) == len(length_features) == len(
        length_diff_features) == len(length_diff_rate_features), "基于统计的文本特征数量没有对齐"
    assert len(word_match_share_features) == len(powerful_word_double_side_features) == len(
        powerful_word_one_side_features) == len(powerful_word_double_rate_features) == len(
        powerful_Word_One_rate_features), "基于PowerfulWord features文本特征数量没有对齐"
    assert len(mean_tf_idf_features) == len(powerful_word_double_side_features), "基于mean TF-IDF文本特征数量没有对齐"
    assert len(powerful_word_double_side_features) == len(Jaccard_features) == len(dice_features) == len(
        distance_features) == len(leve_distance_features), "基于Distance features文本特征数量没有对齐"
    print("开始保存到文件......")
    # features_file_object = open(all_static_features_file, mode='w', encoding="UTF-8")
    all_statical_features = [word_match_share_features, tf_idf_wordMatch_share_features, length_features,
                             length_diff_features, length_diff_rate_features,
                             powerful_word_double_side_features, powerful_word_one_side_features,
                             powerful_word_double_rate_features, powerful_Word_One_rate_features,
                             mean_tf_idf_features, Jaccard_features, dice_features, distance_features,
                             leve_distance_features]
    print("总计数据量: ", len(leve_distance_features))
    static_features_length = 225

    lcqmc_statical_features = []
    for index in tqdm(range(len(leve_distance_features))):
        single_line_features = []
        for features in all_statical_features:
            single_line_features += features[index]
        # print(len(single_line_features))
        assert static_features_length == len(single_line_features), "统计特征数量出错"
        lcqmc_statical_features.append(single_line_features)
        # features_file_object.write(" ".join([str(num) for num in single_line_features]) + "\n")
    features_file_object_wb = open(all_static_features_file, mode='wb')
    pk.dump(lcqmc_statical_features, features_file_object_wb)
    print("基于统计的文本特征长度: ", static_features_length)
    print("统计特征写入完成")
    # features_file_object.close()


def get_afqmc_statistics_features():
    afqmc_floder = "E:/2020-GD/afqmc_features_engineering"
    afqmc_file = "E:/2020-GD/评价数据/afqmc/afqmc_all_token.txt"
    stop_file = afqmc_floder + "/cn_stopwords.txt"
    all_static_features_file = afqmc_floder + "/afqmc_static_features.pk"
    train_file_for_power = afqmc_floder + "/afqmc_train_forPower.txt"
    words_power_file = afqmc_floder + "/afqmc_train_power_file.txt"
    all_data = load_token_file(afqmc_file)
    stop_list = stopwords(stop_file)
    tf_idf = generate_idf(all_data)

    print("开始生成基于统计的文本特征.......")
    word_match_share_features = WordMatchShare(all_data, stop_list)
    tf_idf_wordMatch_share_features = TFIDFWordMatchShare(all_data, tf_idf)
    length_features = Length(all_data)
    length_diff_features = LengthDiff(all_data)
    length_diff_rate_features = LengthDiffRate(all_data)
    print("PowerfulWord features........")
    words_power = PowerfulWord().generate_powerful_word(train_file_for_power)
    PowerfulWord().save_powerful_word(words_power, words_power_file)
    powerful_word_double_side_features = PowerfulWordDoubleSide(words_power_file, 500, 0.2).extract_all_features(
        all_data)
    powerful_word_one_side_features = PowerfulWordOneSide(words_power_file, 800, 0.3).extract_all_features(all_data)
    powerful_word_double_rate_features = PowerfulWordDoubleSideRate(words_power_file).extract_all_features(all_data)
    powerful_Word_One_rate_features = PowerfulWordOneSideRate(words_power_file).extract_all_features(all_data)
    print("mean TF-IDF......")
    mean_tf_idf_features = TFIDF(afqmc_file, stop_list).extract_features(all_data)
    print("Distance features.........")
    Jaccard_features = NgramJaccardCoef(all_data)
    dice_features = NgramDiceDistance(all_data)
    distance_features = NgramDistance(all_data)
    leve_distance_features = LevenshteinDistance(all_data)
    assert len(word_match_share_features) == len(tf_idf_wordMatch_share_features) == len(length_features) == len(
        length_diff_features) == len(length_diff_rate_features), "基于统计的文本特征数量没有对齐"
    assert len(word_match_share_features) == len(powerful_word_double_side_features) == len(
        powerful_word_one_side_features) == len(powerful_word_double_rate_features) == len(
        powerful_Word_One_rate_features), "基于PowerfulWord features文本特征数量没有对齐"
    assert len(mean_tf_idf_features) == len(powerful_word_double_side_features), "基于mean TF-IDF文本特征数量没有对齐"
    assert len(powerful_word_double_side_features) == len(Jaccard_features) == len(dice_features) == len(
        distance_features) == len(leve_distance_features), "基于Distance features文本特征数量没有对齐"
    print("开始保存到文件......")
    # features_file_object = open(all_static_features_file, mode='w', encoding="UTF-8")
    all_statical_features = [word_match_share_features, tf_idf_wordMatch_share_features, length_features,
                             length_diff_features, length_diff_rate_features,
                             powerful_word_double_side_features, powerful_word_one_side_features,
                             powerful_word_double_rate_features, powerful_Word_One_rate_features,
                             mean_tf_idf_features, Jaccard_features, dice_features, distance_features,
                             leve_distance_features]
    print("总计数据量: ", len(leve_distance_features))
    static_features_length = 135

    lcqmc_statical_features = []
    for index in tqdm(range(len(leve_distance_features))):
        single_line_features = []
        for features in all_statical_features:
            single_line_features += features[index]
        # print("特征长度：", len(single_line_features))
        assert static_features_length == len(single_line_features), "统计特征数量出错"
        lcqmc_statical_features.append(single_line_features)
        # features_file_object.write(" ".join([str(num) for num in single_line_features]) + "\n")
    features_file_object_pk = open(all_static_features_file, mode='wb')
    pk.dump(lcqmc_statical_features, features_file_object_pk)
    features_file_object_pk.close()
    print("基于统计的文本特征长度: ", static_features_length)
    print("统计特征写入完成")
    # features_file_object.close()


def get_SRPI_statistics_features():
    SRPI_floder = "E:/2020-GD/SRPI_features_engineering"
    SRPI_file = "E:/2020-GD/评价数据/sow_reap_paraphrase_identification_token.txt"
    stop_file = SRPI_floder + "/cn_stopwords.txt"
    all_static_features_file = SRPI_floder + "/SRPI_static_features.txt"
    # train_file_for_power = SRPI_floder + "/SRPI_train_forPower.txt"
    train_file_for_power = SRPI_file
    words_power_file = SRPI_floder + "/SRPI_train_power_file.txt"
    all_data = load_token_file(SRPI_file)
    stop_list = stopwords(stop_file)
    tf_idf = generate_idf(all_data)
    print("开始生成基于统计的文本特征.......")
    word_match_share_features = WordMatchShare(all_data, stop_list)
    tf_idf_wordMatch_share_features = TFIDFWordMatchShare(all_data, tf_idf)
    length_features = Length(all_data)
    length_diff_features = LengthDiff(all_data)
    length_diff_rate_features = LengthDiffRate(all_data)
    print("PowerfulWord features........")
    words_power = PowerfulWord().generate_powerful_word(train_file_for_power)
    PowerfulWord().save_powerful_word(words_power, words_power_file)
    powerful_word_double_side_features = PowerfulWordDoubleSide(words_power_file, 500, 0.3).extract_all_features(
        all_data)
    powerful_word_one_side_features = PowerfulWordOneSide(words_power_file, 800, 0.3).extract_all_features(all_data)
    powerful_word_double_rate_features = PowerfulWordDoubleSideRate(words_power_file).extract_all_features(all_data)
    powerful_Word_One_rate_features = PowerfulWordOneSideRate(words_power_file).extract_all_features(all_data)
    print("mean TF-IDF......")
    mean_tf_idf_features = TFIDF(SRPI_file, stop_list).extract_features(all_data)
    print("Distance features.........")
    Jaccard_features = NgramJaccardCoef(all_data)
    dice_features = NgramDiceDistance(all_data)
    distance_features = NgramDistance(all_data)
    leve_distance_features = LevenshteinDistance(all_data)
    assert len(word_match_share_features) == len(tf_idf_wordMatch_share_features) == len(length_features) == len(
        length_diff_features) == len(length_diff_rate_features), "基于统计的文本特征数量没有对齐"
    assert len(word_match_share_features) == len(powerful_word_double_side_features) == len(
        powerful_word_one_side_features) == len(powerful_word_double_rate_features) == len(
        powerful_Word_One_rate_features), "基于PowerfulWord features文本特征数量没有对齐"
    assert len(mean_tf_idf_features) == len(powerful_word_double_side_features), "基于mean TF-IDF文本特征数量没有对齐"
    assert len(powerful_word_double_side_features) == len(Jaccard_features) == len(dice_features) == len(
        distance_features) == len(leve_distance_features), "基于Distance features文本特征数量没有对齐"
    print("开始保存到文件......")
    features_file_object = open(all_static_features_file, mode='w', encoding="UTF-8")
    all_statical_features = [word_match_share_features, tf_idf_wordMatch_share_features, length_features,
                             length_diff_features, length_diff_rate_features,
                             powerful_word_double_side_features, powerful_word_one_side_features,
                             powerful_word_double_rate_features, powerful_Word_One_rate_features,
                             mean_tf_idf_features, Jaccard_features, dice_features, distance_features,
                             leve_distance_features]
    print("总计数据量: ", len(leve_distance_features))
    static_features_length = 125

    lcqmc_statical_features = []
    for index in tqdm(range(len(leve_distance_features))):
        single_line_features = []
        for features in all_statical_features:
            single_line_features += features[index]
        # print("特征长度：", len(single_line_features))
        assert static_features_length == len(single_line_features), "统计特征数量出错"
        lcqmc_statical_features.append(single_line_features)
        features_file_object.write(" ".join([str(num) for num in single_line_features]) + "\n")
    all_static_features_file_pk = SRPI_floder + "/SRPI_static_features.pk"
    features_file_object_pk = open(all_static_features_file_pk, mode='wb')
    pk.dump(lcqmc_statical_features, features_file_object_pk)
    print("基于统计的文本特征长度: ", static_features_length)
    print("统计特征写入完成")
    features_file_object.close()


if __name__ == '__main__':
    get_lcqmc_statistics_features()
    # get_afqmc_statistics_features()




