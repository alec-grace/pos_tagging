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
from gensim.models import KeyedVectors
import os


# convert a text file into a list of sentences that are tokenized
def tokenize(data):
    with open(data, 'r') as data_file:
        lines = data_file.readlines()
    data_file.close()
    token_list = []
    for line in lines:
        # replace hyphens and periods because the model does not recognize them
        line = line.replace("-", " ")
        line = line.replace(".", " ")
        for sent in sent_tokenize(line):
            token_list.append(word_tokenize(sent))
    return token_list


# convert text file into a numpy array of the word vectors
def get_nparray(wv, file):
    tokens = tokenize(file)
    words = []
    non_words = {}
    for token in tokens:
        for word in token:
            try:
                words.append(wv[word])
            except KeyError:
                non_words[word] = 1 if word not in non_words.keys() else non_words[word] + 1
    words = np.array(words)
    return words


# paperswithcode tend to have datasets with research papers
def main():
    # check if the model is already saved, if not load it from gensim downloader
    if os.path.exists("goog_news.wordvectors"):
        wv = KeyedVectors.load("goog_news.wordvectors", mmap='r')
    else:
        wv = api.load('word2vec-google-news-300')
        wv.save("goog_news.wordvectors")

    # store each transcript as numpy array of word vectors
    # planning on generalizing directories and dictionaries with config file eventually
    ww_scripts = {}
    for script in os.listdir("ww_text/"):
        ww_scripts[script] = get_nparray(wv, "ww_text/" + script)
    dn_scripts = {}
    for script in os.listdir("dn_text/"):
        dn_scripts[script] = get_nparray(wv, "dn_text/" + script)

    # print for testing purposes
    print(dn_scripts["episode1.txt"])
    print(ww_scripts["101.txt"])


if __name__ == "__main__":
    main()
