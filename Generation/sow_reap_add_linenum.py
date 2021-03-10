# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: sow_reap_add_linenum.py 
@time: 2020年11月07日21时37分 
"""
no_line_num_file = "E:/2020-GD/lcqmc_train_parse_tree_noline_num.txt"
sow_reap_file = "E:/2020-GD/lcqmc_train_parse_tree.txt"
file_object = open(no_line_num_file, encoding='UTF-8', mode='r')
out_object = open(sow_reap_file, mode='w', encoding='UTF-8')
index = 1
for line in file_object:
    if "Sentence" in line:
        line = line.replace("#1", "#"+str(index))
        out_object.write(line)
        index += 1
    else:
        out_object.write(line)
file_object.close()
out_object.close()