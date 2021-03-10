# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: test.py 
@time: 2020年11月25日13时57分 
"""
import pickle as pk
# import tensorflow as tf
import pprint
import numpy as np
import pandas as pd
import re


def load_tensor(init_checkpoint, tensor_name):
    reader = tf.compat.v1.train.NewCheckpointReader(init_checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # 输出权重tensor名字和值
    for key in var_to_shape_map:
        # if "11" in tensor_name or '10' in key:
        print("tensor_name: ", key)
    print("筛选过后的")
    for key in var_to_shape_map:
        if "layer_11" in key or 'layer_10' in key:
            print("tensor_name: ", key)
    # word_embeddings = reader.get_tensor(tensor_name)
    # # 21128 768
    # print(tensor_name, " tensor shape:", {word_embeddings.shape})

#
# chinese_bert_checkpoint = "E:/2020-GD/chinese_L-12_H-768_A-12/bert_model.ckpt"
# load_tensor(init_checkpoint=chinese_bert_checkpoint, tensor_name="dense_1/kernel")

# features_len = {"lcqmc_chinese_elmo_embedding": 2048, "lcqmc_glove_AveVec": 600, "lcqmc_glove_cosvalue": 1,
#                 "lcqmc_glove_tfidf_AveVec": 600, "lcqmc_glove_tfidf_cosvalue": 1, "lcqmc_POStag": 140,
#                 "lcqmc_static": 110, "lcqmc_LDA": 340}
line = "{'src_text':0,'tgt_text':1}"
sample = eval(line.strip())
# src_tk = tokenizer.tokenize(sample["src_text"])
# tgt_tk = tokenizer.tokenize(sample["tgt_text"])
src_tk = sample["src_text"]
tgt_tk = sample["tgt_text"]