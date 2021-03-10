# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: test.py 
@time: 2020年11月07日13时10分 
"""

import pandas as pd
# "E:\2020-GD\评价数据\\afqmc_train.txt"
df = pd.read_csv("E:/实验三/afqmc_train_positive_negtive_augmentation.tsv", sep='\t', header=None)
df.columns = ["sentence_1", "sentence_2", "label"]
print(df.info())
print(df["label"].value_counts())
# import random
# print(random.sample(range(1, 34), 6))

# nlp = StanfordCoreNLP("C:/Users/gaojiaming/Desktop/2020-GD/生成/sow-reap-paraphrasing-master/stanford-corenlp-full-2018-02-27", lang='zh')
# 这里改成你stanford-corenlp所在的目录
# sentence = "我爱北京天安门"
# print('Tokenize:', nlp.word_tokenize(sentence))
# print('Part of Speech:', type(nlp.pos_tag(sentence)))
# print('Named Entities:', nlp.ner(sentence))
# print('Constituency Parsing:', nlp.parse(sentence))
# print('Dependency Parsing:', type(nlp.dependency_parse(sentence)))
# nlp.close()  # Do not forget to close! The backend server will consume a lot memery.


# from elmoformanylangs import Embedder
# #
# e = Embedder('C:/Users/gaojiaming/Desktop/2020-GD/生成/sow-reap-paraphrasing-master/sow-reap-paraphrasing-master/chinese_glove_zhs.model')
#
# sents = [['今', '天', '天氣', '真', '好', '阿'],
# ['潮水', '退', '了', '就', '知道', '誰', '沒', '穿', '褲子']]
# # the list of lists which store the sentences
# # after segment if necessary.
#
# matrix = e.sents2elmo(sents, output_layer=-2)  # 1024维度
#
# print(len(matrix))
# print(len(matrix[0]))
# print(len(matrix[1]))
# print(len(matrix[1][1]))
# print(matrix[1][1][1][1])
# print(len(matrix[0][0]))
