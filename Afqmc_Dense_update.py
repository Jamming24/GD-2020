# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: Afqmc_Dense_update.py 
@time: 2021年03月01日10时59分 
"""

# 读取各类特征文件进行组合 进行特征融合 然后进行训练提高分类效果


def features_merge_one_features():
    # 仅仅加载一种特征 之所以写的这么蠢 是为了怕发生之前实验过程中莫名发生的 实验准确率下降的问题
    pass

def features_merge_Three_features():


def features_merge_two_features(features_names, features_floder):
    # 读取特征文件 并进行组合拼接 不需要返回加载BERT embedding
    static_features = "E:/afqmc_features/afqmc_static_features.pk"  # 0.75
    features_data = pk.load(open(static_features, 'rb'))
    train_features_data = features_data[:34334]
    test_features_data = features_data[34334:]
    print("train shape: ", np.array(train_features_data).shape)
    print("test shape: ", np.array(test_features_data).shape)

    word2vector_features = "E:/afqmc_features/afqmc_glove_AveVec.pk"
    word2vector_features_data = pk.load(open(word2vector_features, 'rb'))
    word2vector_train_features_data = word2vector_features_data[:34334]
    word2vector_test_features_data = word2vector_features_data[34334:]
    print("train shape: ", np.array(word2vector_train_features_data).shape)
    print("test shape: ", np.array(word2vector_test_features_data).shape)

    train_merge_features_list = []
    test_merge_features_list = []
    print("开始合并特征........")
    for index in range(len(train_features_data)):
        train_merge_features_list.append(train_features_data[index] + word2vector_train_features_data[index])
    train_merge_features_matrix = np.array(train_merge_features_list)
    for index in range(len(test_features_data)):
        test_merge_features_list.append(test_features_data[index] + word2vector_test_features_data[index])
    test_merge_features_matrix = np.array(test_merge_features_list)

    print('输入训练数据打印shape:', train_merge_features_matrix.shape)
    print('输入测试数据打印shape:', test_merge_features_matrix.shape)


features_names = "statical,elmo,word2vector"
features_floder = None
features_merge(features_names, features_floder)