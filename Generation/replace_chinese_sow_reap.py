# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: replace_chinese_sow_reap.py 
@time: 2020年11月10日11时39分 
"""
# 替换中文sow_reap文件中未token的原中文句子

root = "E:/2020-GD/实验三"
Stanford_chinese = root + "/lcqmc_train_parse_tree_Stanford_Token.txt"
Stanford_chinese_object = open(Stanford_chinese, 'r', encoding='UTF-8')
Stanford_chinese_list = []
for line in Stanford_chinese_object:
    Stanford_chinese_list.append(line)
sow_reap = root + "/lcqmc_train_parse_tree.txt"
sow_reap_object = open(sow_reap, mode='r', encoding='UTF-8')
out_file = open(root + "/lcqmc_train_parse_tree_final.txt", mode='w', encoding="UTF-8")
out_file.write(sow_reap_object.readline())
out_file.write(sow_reap_object.readline())
index = 0
for line in sow_reap_object:
    out_file.write(line)
    if "Sentence" in line:
        sow_reap_object.readline()
        out_file.write(Stanford_chinese_list[index])
        index += 1
out_file.close()
Stanford_chinese_object.close()
sow_reap_object.close()