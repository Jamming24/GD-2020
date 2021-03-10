# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: Afqmc_Dense.py 
@time: 2021年02月28日09时54分 
"""
import pandas as pd
import numpy as np
import pickle as pk
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn import preprocessing

root_path = "E:/afqmc_features/"
train_path = root_path + "afqmc/afqmc_train_token.txt"
test_path = root_path + "afqmc/afqmc_dev_token.txt"
model_path = "E:/afqmc_features/模型文件"
static_features = "E:/afqmc_features/afqmc_static_features.pk" # 0.75
MLP_size = 64
features_name = "统计特征"

one_hot = preprocessing.OneHotEncoder(sparse=False)
train_csv = pd.read_csv(open(train_path, encoding="UTF-8"), header=None, names=["text1", "text2", "label"], sep="\t")
train_label = np.array(train_csv["label"].tolist())
train_label = pd.get_dummies(train_label)
test_csv = pd.read_csv(open(test_path, encoding="UTF-8"), header=None, names=["text1", "text2", "label"], sep="\t")
test_label = np.array(test_csv["label"].tolist())
# 转换为one-hot
test_label = pd.get_dummies(test_label)
# print(test_label)

train_BERT_embedding_file = "E:/afqmc_features/afqmc_train_embedding_maxAuc.txt"
test_BERT_embedding_file = "E:/afqmc_features/afqmc_test_embedding_maxAuc.txt"
train_CLS_info = np.loadtxt(train_BERT_embedding_file, delimiter="\t")
test_CLS_info = np.loadtxt(test_BERT_embedding_file, delimiter="\t")
print("train shape: ", train_CLS_info.shape)
print("test shape: ", test_CLS_info.shape)

features_data = pk.load(open(static_features, 'rb'))
print("features_data: ",  len(features_data))
print("train shape: ", np.array(features_data).shape)
print("test shape: ", np.array(features_data).shape)

# word2vector_features = "E:/afqmc_features/afqmc_glove_AveVec.pk"
# word2vector_features_data = pk.load(open(word2vector_features, 'rb'))
# word2vector_train_features_data = word2vector_features_data[:34334]
# word2vector_test_features_data = word2vector_features_data[34334:]
# print("train shape: ", np.array(word2vector_train_features_data).shape)
# print("test shape: ", np.array(word2vector_test_features_data).shape)

# cos_features = "E:/afqmc_features/afqmc_glove_cos_value.pk"
# cos_features_data = pk.load(open(cos_features, 'rb'))
# cos_train_features_data = cos_features_data[:34334]
# cos_test_features_data = cos_features_data[34334:]
# print("train shape: ", np.array(cos_train_features_data).shape)
# print("test shape: ", np.array(cos_test_features_data).shape)

# POS_features = "E:/afqmc_features/afqmc_POStag_features.pk"
# POS_features_data = pk.load(open(POS_features, 'rb'))
# POS_train_features_data = POS_features_data[:34334]
# POS_test_features_data = POS_features_data[34334:]
# print("train shape: ", np.array(POS_train_features_data).shape)
# print("test shape: ", np.array(POS_test_features_data).shape)
#
# LDA_features = "E:/afqmc_features/afqmc_LDA_features_features_num-600_.pk"
# LDA_features_data = pk.load(open(LDA_features, 'rb'))
# LDA_train_features_data = LDA_features_data[:34334]
# LDA_test_features_data = LDA_features_data[34334:]
# print("train shape: ", np.array(LDA_train_features_data).shape)
# print("test shape: ", np.array(LDA_test_features_data).shape)

train_merge_features_list = []
test_merge_features_list = []
print("开始合并特征........")
for index in range(len(train_CLS_info)):
    train_merge_features_list.append(features_data[index])
    # + word2vector_train_features_data[index] + cos_train_features_data[index] + POS_train_features_data[index] + LDA_train_features_data[index])
train_merge_features_matrix = np.array(train_merge_features_list)
for index in range(len(test_CLS_info)):
    test_merge_features_list.append(features_data[index])
    # + word2vector_test_features_data[index] + cos_test_features_data[index] + POS_test_features_data[index] + LDA_test_features_data[index])
test_merge_features_matrix = np.array(test_merge_features_list)

print('输入训练数据打印shape:', train_merge_features_matrix.shape)
print('输入测试数据打印shape:', test_merge_features_matrix.shape)

tf.reset_default_graph()
# x表示能够输入任意数量的MNIST图像，每张图都是784维的向量，None表示第一维度可以是任意长度的
x = tf.placeholder(tf.float32, [None, 768])
features_holder = tf.placeholder(tf.float32, [None, 135])
# 数字0~9，共10个类别
y = tf.placeholder(tf.float32, [None, 2])

# 特征映射层 结点数量可以作为超参数调节
out_layer_add_features_1 = tf.layers.dense(inputs=features_holder, units=MLP_size, activation=tf.nn.relu) # 最好一层映射就可以
# 最终分类层
# out_layer_add_features_2 = tf.layers.dense(inputs=out_layer_add_features_1, units=256, activation=tf.nn.relu)
print(out_layer_add_features_1.shape)
output = tf.concat([out_layer_add_features_1, x], 1)
# output = tf.layers.dense(inputs=output, units=256, activation=tf.nn.relu) # 0.7363 # 0.7386 BERT之后再接层性能会下降
classifier_dense = tf.layers.dense(inputs=output, units=2, activation=tf.nn.sigmoid) # 这里使用sigmoid激活函数最好
pred = tf.nn.softmax(classifier_dense)  # softmax分类
# 损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
# 定义参数
learning_rate = 0.01
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
training_epochs = 200
batch_size = 32
display_step = 1
# 模型保存参数
saver = tf.train.Saver(max_to_keep=500)

# # 启动session
with tf.Session() as sess:
    epoch_list = []
    train_cost = []
    test_Accuracy = []
    test_F1_score = []
    init = tf.global_variables_initializer()
    sess.run(init)
    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(train_merge_features_matrix.shape[0] / batch_size) + 1
        # 使用随机批梯度下降循环所以数据集
        for i in range(total_batch):
            add_features = train_merge_features_matrix[i*batch_size:i*batch_size+batch_size]
            bert_embedding = train_CLS_info[i*batch_size:i*batch_size+batch_size]
            batch_y = train_label[i*batch_size:i*batch_size+batch_size]
            # 运行优化器
            c = sess.run([optimizer, cost], feed_dict={x: bert_embedding, features_holder: add_features, y: batch_y})
            avg_cost += c[1] / total_batch
        # 显示训练中的详细信息
        correct_prediction = tf.equal(tf.math.argmax(pred, 1), tf.math.argmax(y, 1))
        reduce_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if (epoch + 1) % display_step == 0:
            # '%04d' % (epoch+1) 表示格式化输出 4表示 总计位数
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            # 测试Model  用tf.arg_max返回onehot编码中数值为1那个元素的下标 这三行代码都是干什么用的呢？？？？？？？
            # 计算准确率
            accuracy = reduce_accuracy.eval({x: test_CLS_info, features_holder: test_merge_features_matrix, y: test_label})
            print("Accuracy:", accuracy)

            acc, y_pred, y_true = sess.run([reduce_accuracy, tf.argmax(pred, 1), tf.math.argmax(y, 1)], feed_dict={x: test_CLS_info, features_holder: test_merge_features_matrix, y: test_label})
            model_f1_score = f1_score(y_true, y_pred)
            print("model_f1_score: ", model_f1_score)
            # print("model_acc: ", acc)

            epoch_list.append(epoch)
            train_cost.append(avg_cost)
            test_Accuracy.append(accuracy)
            test_F1_score.append(model_f1_score)
            # 保存模型,并将模型保存的路径打印出来
            save_path = saver.save(sess, model_path+"/"+features_name+"_"+str(MLP_size)+"/"+str(epoch)+"_"+str(accuracy)+"_"+str(model_f1_score)+".ckpt")
            print("Model saved in file: %s" % save_path)
    print("Training Finished!")
    # # 创建图形
    # plt.figure(1)
    # # 第一行第一列图形
    # ax1 = plt.subplot(2, 2, 1)
    # # 第一行第二列图形
    # ax2 = plt.subplot(2, 2, 2)
    # # 第二行
    # ax3 = plt.subplot(2, 1, 2)
    #
    # # 选择ax1
    # plt.sca(ax1)
    # plt.title(f'MLP_size: {MLP_size}    train error Result Analysis')
    # plt.plot(epoch_list, train_cost, color='red', label='train error')
    # plt.legend()  # 显示图例
    # plt.xlabel('iteration times')
    # plt.ylabel('rate')
    #
    # # 选择ax2
    # plt.sca(ax2)
    # plt.title(f'MLP_size: {MLP_size}    F1-score Result Analysis')
    # plt.plot(epoch_list, test_F1_score, color='green', label='test F1-score')
    # plt.legend()  # 显示图例
    # plt.xlabel('iteration times')
    # plt.ylabel('rate')
    #
    # # 选择ax3
    # plt.sca(ax3)
    plt.title(f'MLP_size: {MLP_size}    Accuracy Result Analysis')
    # plt.plot(epoch_list, train_cost, color='green', label='training cost', marker='x')
    plt.plot(epoch_list, test_Accuracy, color='skyblue', label='test Accuracy')
    plt.legend()  # 显示图例
    plt.xlabel('iteration times')
    plt.ylabel('rate')
    plt.show()


