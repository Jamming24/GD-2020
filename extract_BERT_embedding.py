# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: extract_BERT_embedding.py 
@time: 2021年02月28日19时01分 
"""

import json
from tqdm import tqdm


def load_BERT_embedding(BERT_json_file, BERT_embedding_file, save_flag=False):
    print("加载BERT json 文件")
    json_f = open(BERT_json_file)
    BERT_CLS_info_list = []
    # 将json格式的数据映射成list的形式
    for json_line in tqdm(json_f.readlines()):  # 按行读取json文件，每行为一个字符串
        data = json.loads(json_line)  # 将字符串转化为列表
        # index = data["linex_index"]
        # 返回CLS向量 768维度
        CLS_info = data["features"][0]["layers"][0]["values"]
        BERT_CLS_info_list.append(CLS_info)
    json_f.close()
    BERT_CLS_info = np.array(BERT_CLS_info_list)
    print(BERT_CLS_info.shape)
    print(BERT_CLS_info[:5])
    if save_flag:
        np.savetxt(BERT_embedding_file, BERT_CLS_info, delimiter="\t")
        print("BERT embedding 保存完成")
    return BERT_CLS_info


train_json_file = "E:/afqmc_features/afqmc_train_embedding_-1_maxAuc.json"
test_json_file = "E:/afqmc_features/afqmc_test_embedding_-1_maxAuc.json"
train_BERT_embedding_file = "E:/afqmc_features/afqmc_train_embedding_maxAuc.txt"
test_BERT_embedding_file = "E:/afqmc_features/afqmc_test_embedding_maxAuc.txt"
train_BERT_embedding = load_BERT_embedding(train_json_file, train_BERT_embedding_file, True)
test_BERT_embedding = load_BERT_embedding(test_json_file, test_BERT_embedding_file, True)