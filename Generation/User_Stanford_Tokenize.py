# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: User_Stanford_Tokenize.py 
@time: 2020年11月08日19时11分 
"""


def parse_token_line(line):
    line = line.replace("[", "").replace("]", "")
    line = line.split(" ")
    token = line[0].split("=")[1]
    pos = line[3].split("=")[1]
    return token, pos


file_name = "E:/2020-GD/实验三/lcqmc_train_parse_tree.txt"
out_file = "E:/2020-GD/实验三/lcqmc_train_parse_tree_Stanford_Token.txt"
file = open(file_name, mode='r', encoding='UTF-8')

out_file_object = open(out_file, 'w', encoding='UTF-8')
file.readline()  ### READ TWO EXTRANEUOUS LINE OF FILE
file.readline()
line_num = 0
### READ TOKENS
assert file.readline().strip().startswith("Sentence"), "new Sentence"
while True:
    if line_num % 10000 == 0:
        print(line_num)
    if line_num == 925891:
        out_file_object.flush()
        break
    sentence = file.readline()
    # print(sentence.strip())
    file.readline()
    assert file.readline().strip().startswith("Tokens"), "parsing error tokens"
    tokens = []
    while True:
        line = file.readline().strip()
        if line == "":
            break
        token, pos = parse_token_line(line)
        tokens.append(token)
    # print(tokens)
    # print()
    out_file_object.write(" ".join(tokens) + "\n")
    line = file.readline()
    line_num += 1
    while True:
        if "Sentence" in line:
            break
        else:
            line = file.readline()
    # if line == "EOF":  ### REACHED END OF FILE
    #     break
    out_file_object.flush()

file.close()
out_file_object.close()
print("处理完成")
