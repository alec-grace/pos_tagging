# File: embeddings.py
# Author: Alec Grace
# Created: 20 Feb 2024
# Purpose:
#   Create word embeddings from a given dataset

import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
import pickle
import os


# def training(train_file: str) -> list:
#     corpus = ''
#     with open(train_file, 'r') as infile:
#         lines = infile.readlines()
#     infile.close()
#
#     for line in lines:
#         corpus = corpus + line.replace("\n", " ")
#
#     data = []
#     sents = sent_tokenize(corpus)
#     for sent in sents:
#         temp = []
#         for word in word_tokenize(sent):
#             temp.append(word)
#         data.append(temp)
#     return data
#
#
# def text2vec(corpus):
#     mod_min = 1
#     vec_size = 300
#     window = 10
#     sg = 0
#     model = Word2Vec.load("archive/GoogleNews-vectors-negative300.bin")


# paperswithcode tend to have datasets with research papers
def main():
    # model = Word2Vec.load("GoogleNews-vectors-negative300.bin")
    # print(model.wv['hello'])
    if os.path.exists("goog_news.model"):
        wv = Word2Vec.load("goog_news.model")
    else:
        wv = api.load('word2vec-google-news-300')
        wv.save("goog_news.model")

    print(wv['hello'])


if __name__ == "__main__":
    main()
