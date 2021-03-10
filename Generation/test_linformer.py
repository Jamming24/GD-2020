# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: test_linformer.py 
@time: 2020年11月10日16时15分 
"""

import torch
# from torch.nn.utils.rnn import pack_padded_sequence
# from linformer_pytorch import Linformer
# from linformer_pytorch import LinformerEncDec
# from torch.nn.utils.rnn import pack_padded_sequence
# import numpy as np

#
# encdec = LinformerEncDec(
#     enc_num_tokens=10000,
#     enc_input_size=512,
#     enc_channels=16,
#     dec_num_tokens=10000,
#     dec_input_size=512,
#     dec_channels=16,
# )
#
# x = torch.randint(1,10000,(1,512))
# y = torch.randint(1,10000,(1,512))
#
# output = encdec(x,y)
# print(output)
# inputs = torch.tensor([[1, 3, 4, 6, 5, 2], [1, 3, 4, 5, 2, 0],
#                        [1, 8, 3, 6, 4, 2], [1, 5, 4, 8, 3, 2]])
# hidden_size = 256
# embedder = nn.Embedding(10, 8, padding_idx=0)
# scale_embedding = hidden_size ** 0.5
# print("scale_embedding: ", scale_embedding)
# x = embedder(inputs)
# print("x......")
# print(x)
# print(x.shape)
# x = x.mul_(scale_embedding)

# batch_size = 2
# max_length = 3
# hidden_size = 2
# n_layers = 1

#
# inputs = torch.tensor([[1, 3, 4, 6, 5, 2], [1, 3, 4, 5, 2, 0],
#                        [1, 8, 3, 6, 4, 2], [1, 5, 4, 8, 3, 2]])


# from tensorflow.keras.preprocessing import sequence
# import numpy as np

# a=np.array([[1, 3, 4, 6, 5, 2], [1, 3, 4, 5, 2, 0], [1, 8, 3, 6, 4, 2], [1, 5, 4, 8, 3, 2]])
# a=np.array([[1, 8, 3, 6, 4, 2]])
# ap = sequence.pad_sequences(a, 12, padding="post", truncating="post") # "post"
# torch.zeros(*size,                  => 张量的形状，如(3, 3)
#             out = None,             => 输出的张量
#             dtype = None,           => 内存数据类型
#             layout = torch.strided, => 内存中布局形式，有strided,sparse_coo等
#             device = None,          => 所在设备，gpu/cpu
#             requires_grad  =False)  => 是否需要梯度
# print(ap)
# a = torch.zeros([1, 3, 4])
# print("a = ", a)
# inputs.view(-1, 24)
# print(inputs)

candidate = ['而且 事情 全部 的 真相 ， 我 自己 当时 也 不 知道 。']  # 预测摘要, 可以是列表也可以是句子
reference = ['这 事情 当时 我 是 我 自己 的 事情 。']  # 真实摘要
from rouge import Rouge
rouge = Rouge()
rouge_score = rouge.get_scores(hyps=candidate, refs=reference)
print(rouge_score[0]["rouge-1"])
print(rouge_score[0]["rouge-2"])
print(rouge_score[0]["rouge-l"])