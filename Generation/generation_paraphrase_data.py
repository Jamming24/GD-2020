# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: generation_paraphrase_data.py
@time: 2020年12月07日14时01分 
"""


def convert_result_tolist(result_file, model_type="sow"):
    result_file_object = open(result_file, encoding="UTF-8", mode='r')
    result_sentence = []
    blank_line = 0
    if model_type == "sow":
        blank_line = 6
    elif model_type == "reap":
        blank_line = 1
    elif model_type == "baseline":
        blank_line = 0
    while True:
        single_sentence = []
        input_sentence = result_file_object.readline().strip()[15:].lstrip(" ").rstrip(" ")
        true_sentence = result_file_object.readline().strip()[22:].lstrip(" ").rstrip(" ")
        single_sentence.append(input_sentence)
        single_sentence.append(true_sentence)
        for i in range(blank_line):
            result_file_object.readline()
        while True:
            generated_sentence = result_file_object.readline().strip()
            if "Generated Sentence" in generated_sentence:
                single_sentence.append(generated_sentence[19:].lstrip(" ").rstrip(" "))
                result_file_object.readline()
            else:
                if "Ground Truth nll" in generated_sentence:
                    single_sentence.append(generated_sentence[17:].lstrip(" ").rstrip(" "))
                    result_sentence.append(single_sentence)
                    result_file_object.readline()
                break
        if input_sentence == '':
            break
    result_file_object.close()
    print(model_type, " senetence nums: ", len(result_sentence))
    return result_sentence


root_file = "E:/2020-GD/生成/result"
sow_reap_out_file = root_file + "/chinese_all_sow_reap_transfomer.out"
sow_reap_paraphrase_file = "E:/2020-GD/评价数据/sow_reap_paraphrase_identification.txt"
sow_list = convert_result_tolist(sow_reap_out_file, model_type="sow") # sow_reap 5818对句子
file_object = open(sow_reap_paraphrase_file, encoding="UTF-8", mode='w')
for line in sow_list:
    file_object.write("".join(line[0].split())+"\t"+"".join(line[1].split())+"\t1\n")
    file_object.write("".join(line[0].split()) + "\t" + "".join(line[-2].split()) + "\t0\n")
file_object.close()
