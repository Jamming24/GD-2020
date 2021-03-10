# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: Create_Chinese_BPE.py 
@time: 2021年01月06日16时50分 
"""
import pickle as pk
import codecs

# "bpe.codes"
# "parse_vocab.pkl"
# "parse_vocab_rules.pkl"
# "pos_vocab.pkl"
# "vocab.txt"

root = "E:/2020-GD/实验三/lcqmc_afqmc_chinese_resources"


def create_parse_vocab():
    chinese_BPE_vocab = open(root + "/lcqmc_afqmc_vocab.txt", 'r', encoding='UTF-8')
    chinese_vocab_dict = {"PAD": 0, "BOS": 1, "EOS": 2}
    re_chinese_vocab_dict = {0: "PAD", 1: "BOS", 2: "EOS"}
    index = 3
    for line in chinese_BPE_vocab:
        lines = line.strip().split(" ")
        chinese_vocab_dict[lines[0]] = index
        re_chinese_vocab_dict[index] = lines[0]
        index += 1

    file_obj = open(root + '/lcqmc_afqmc_parse_vocab.pkl', 'wb')
    add_tuple = (chinese_vocab_dict, re_chinese_vocab_dict)
    pk.dump(add_tuple, file_obj)
    chinese_BPE_vocab.close()
    pp_vocab, rev_pp_vocab = pk.load(open(root + '/lcqmc_afqmc_parse_vocab.pkl', 'rb'))
    print(pp_vocab)
    # print(rev_pp_vocab)
    # PAD:0   BOS:1  EOS:2


def create_parse_vocab_rules():
    chinese_BPE_vocab = open(root + "/lcqmc_afqmc_vocab.txt", 'r', encoding='UTF-8')
    chinese_parse_vocab_rules = {"PAD": 0, "BOS": 1, "EOS": 2, 'X': 3, 'Y': 4}
    re_chinese_parse_vocab_rules = {0: "PAD", 1: "BOS", 2: "EOS", 3: 'X', 4: 'Y'}
    index = 5
    for line in chinese_BPE_vocab:
        lines = line.strip().split(" ")
        chinese_parse_vocab_rules[lines[0]] = index
        re_chinese_parse_vocab_rules[index] = lines[0]
        index += 1
    chinese_BPE_vocab.close()
    file_obj = open(root + '/lcqmc_afqmc_parse_vocab_rules.pkl', 'wb')
    add_tuple = (chinese_parse_vocab_rules, re_chinese_parse_vocab_rules)
    pk.dump(add_tuple, file_obj)
    print(add_tuple)
    print(len(chinese_parse_vocab_rules))
    # 'PAD': 0, 'BOS': 1, 'EOS': 2, 'X': 3, 'Y': 4,


def create_pos_vocab():
    # 英文
    # NON_TERMINALS = ["S", "SBAR", "SQ", "SBARQ", "SINV",
    #                  "ADJP", "ADVP", "CONJP", "FRAG", "INTJ", "LST",
    #                  "NAC", "NP", "NX", "PP", "PRN", "QP",
    #                  "RRC", "UCP", "VP", "WHADJP", "WHAVP", "WHNP", "WHPP", "WHADVP",
    #                  "X", "ROOT", "NP-TMP", "PRT"]
    # 中文
    # NON_TERMINALS = ["S", "IP", "NP", "DP", "VP", "CP", "CLP", "LCP", "DVP", "DNP",
    # "ADJP", "VSB", "VRD", "QP", "PP", "ADVP", "ROOT", "FLR", "INTJ",
    # "UCP", "VNV", "PRN", "DFL", "VCD", "VPT", "VCP", "FRAG", "LST",
    # "INC", "WHPP"]

    # {0: 'NONE', 1: 'IN', 2: 'PRP', 3: 'VBD', 4: 'RB', 5: 'NP', 6: ':', 7: 'VBN', 8: 'DT', 9: 'JJ', 10: 'NNS', 11: 'PRP$', 12: 'NN', 13: 'VBG', 14: 'PP', 15: "''", 16: '.', 17: 'S', 18: ',', 19: 'FW', 20: 'VBP', 21: 'VP', 22: 'VBZ', 23: 'CC', 24: 'EX', 25: 'MD', 26: 'LS', 27: 'RBS', 28: 'JJS', 29: 'TO', 30: 'VB', 31: 'POS', 32: 'CD', 33: 'WDT', 34: 'SINV', 35: 'NNP', 36: 'SBAR', 37: 'CONJP', 38: 'RBR', 39: 'JJR', 40: 'ADJP', 41: 'FRAG', 42: 'WP', 43: 'PDT', 44: 'ADVP', 45: 'WHNP', 46: 'UH', 47: 'SYM', 48: 'WRB', 49: 'SQ', 50: 'QP', 51: 'PRN', 52: 'WHADJP', 53: 'RP', 54: 'NP-TMP', 55: 'NAC', 56: 'X', 57: '$', 58: 'WHADVP', 59: 'UCP', 60: '-LRB-', 61: '-RRB-', 62: '#', 63: 'RRC', 64: 'NX', 65: 'SBARQ', 66: 'NNPS', 67: 'INTJ', 68: 'WP$', 69: 'WHPP', 70: 'LST'}
    pos_dict = {'FW', 'VA', 'FLR', 'AS', 'VRD', 'VNV', 'LC', 'AD', 'VE', 'VSB', 'DP', 'JJ', 'UCP', 'LB', 'DFL', 'VPT',
                'NP', 'PU', 'PN', 'CS', 'VC', 'MSP', 'PRN', 'ADVP', 'DVP', 'CP', 'ROOT', 'M', 'SP', 'VCD', 'ETC', 'VP',
                'VV', 'NN', 'FRAG', 'DNP', 'CC', 'PP', 'INC', 'IP', 'WHPP', 'QP', 'DT', 'LCP', 'INTJ', 'OD', 'DEC',
                'VCP', 'LST', 'ADJP', 'CD', 'NR', 'DEG', 'CLP', 'NT', 'DER', 'SB', 'ON', 'IJ', 'DEV', 'BA', 'P'}
    pos_set = set()
    for p in pos_dict:
        pos_set.add(p)
    zh_pos_dict = dict()
    re_zh_pos_dict = dict()
    index = 0
    for p in pos_set:
        zh_pos_dict[p] = index
        re_zh_pos_dict[index] = p
        index += 1
    file_obj = open(root + '/lcqmc_afqmc_pos_vocab.pkl', 'wb')
    add_tuple = (zh_pos_dict, re_zh_pos_dict)
    pk.dump(add_tuple, file_obj)
    file_obj.close()
    # 其实所有词表文件都是基于BPE生成的词表改进来的 不知道不用BPE词表可不可以


# create_parse_vocab()
# create_parse_vocab_rules()
create_pos_vocab()
