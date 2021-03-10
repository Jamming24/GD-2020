# -*- coding:utf-8 -*-
""" 
@author:gaojiaming 
@file: Visualize_the_embeddings.py 
@time: 2021年01月30日09时30分 
"""
import os
from tempfile import gettempdir

# Step 6: Visualize the embeddings.

# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.


def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(20, 20))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.savefig(filename)


import numpy as np


embedding_file = "C:/Users/gaojiaming/Desktop/PAPER_word2vector_word_add_component_vector.txt"
word_vector_object = open(embedding_file, encoding="UTF-8", mode='r')
word_embeddings = []
labels = []
size = 0
for line in word_vector_object:
    ll = line.strip("\n").split(" ")
    character = ll[0]
    if len(ll) == 2:
        size = int(ll[1])
    else:
        vector = ll[1:-1]
        vector = np.array(list(map(lambda x: float(x), vector)))
        labels.append(character)
        word_embeddings.append(vector)
word_vector_object.close()
print(size)
try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'STSong'
    plot_only = 500
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    low_dim_embs = tsne.fit_transform(word_embeddings[:plot_only])
    # labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels[:plot_only], 'C:/Users/gaojiaming/Desktop/tsne_500.png')
    plt.show()

except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)
