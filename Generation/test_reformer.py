# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: test_reformer.py 
@time: 2020年11月14日12时28分 
"""

import torch
from reformer_pytorch import ReformerEncDec

#
DE_SEQ_LEN = 32
EN_SEQ_LEN = 32

enc_dec = ReformerEncDec(dim=32, enc_num_tokens=200, enc_depth=1, enc_max_seq_len=DE_SEQ_LEN, dec_num_tokens=200,
                         dec_depth=1, dec_max_seq_len=EN_SEQ_LEN).cuda()
train_seq_in = torch.randint(0, 200, (8, 18)).long().cuda()
train_seq_out = torch.randint(0, 200, (8, 16)).long().cuda()
input_mask = torch.ones(8, 16).bool().cuda()

loss = enc_dec(train_seq_in, train_seq_out, return_loss = True, enc_input_mask = input_mask)
loss.backward()

eval_seq_in = torch.randint(0, 200, (1, DE_SEQ_LEN)).long().cuda()
eval_seq_out_start = torch.tensor([[0.]]).long().cuda()  # assume 0 is id of start token
samples = enc_dec.generate(eval_seq_in, eval_seq_out_start, seq_len=16, eos_token=1)  # assume 1 is id of stop token
print(samples.shape) # (1, <= 1024) decode the tokens
print(samples)
save_path = "E:/2020-GD/生成/test.parameter"
torch.save({'state_dict': enc_dec.state_dict(), 'ep_loss': loss / 8, 'train_minibatches': 8}, save_path)

new_model = ReformerEncDec(dim=32, enc_num_tokens=200, enc_depth=1, enc_max_seq_len=DE_SEQ_LEN, dec_num_tokens=200,
                         dec_depth=1, dec_max_seq_len=EN_SEQ_LEN).cuda()
pp_model = torch.load(save_path)
new_model.load_state_dict(pp_model["state_dict"])
samples_test = enc_dec.generate(eval_seq_in, eval_seq_out_start, seq_len=16, eos_token=1)
print("samples_test")
print(samples_test)
for k in samples_test[0]:
    print(k.item())