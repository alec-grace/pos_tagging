# File: embeddings.py
# Author: Alec Grace
# Created: 20 Feb 2024
# Purpose:
#   Create word embeddings from a given dataset - needs work

import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim.downloader as api
from gensim.models import KeyedVectors
import os
import torch


class Embeddings:

    def __init__(self):
        """
        Constructor for a class to utilize embeddings from
        Word2Vec's Google News 300 dataset
        """
        # check if the embedding model is already saved, if not load it from gensim downloader
        if os.path.exists("goog_news.wordvectors"):
            self.wv = KeyedVectors.load("goog_news.wordvectors", mmap='r')
        else:
            self.wv = api.load('word2vec-google-news-300')
            self.wv.save("goog_news.wordvectors")

    # convert a text file into a list of sentences that are tokenized
    def __tokenize_file(self, data):
        """
        Tokenizes a whole file
        :param data: text file
        :return: the file as a list of individual tokens
        """
        with open(data, 'r') as data_file:
            lines = data_file.readlines()
        data_file.close()
        token_list = []
        for line in lines:
            # replace hyphens and periods because the model does not recognize them
            line = line.replace("-", " ")
            line = line.replace(".", " ")
            # add each sentence with tokenized words to the continuous list
            for sent in sent_tokenize(line):
                token_list.append(word_tokenize(sent))
        return token_list

    def __tokenize_list(self, data):
        """
        Tokenizes sentences in a list
        :param data: list of sentences
        :return: list of tokenized sentences
        """
        token_list = []
        for item in data:
            # replace hyphens and periods because the model does not recognize them
            item = item.replace("-", " ")
            item = item.replace(".", " ")
            # add each sentence with tokenized words to the continuous list
            token_list.append(word_tokenize(item))
        return token_list

    # convert text file into a list of Pytorch Tensors
    def get_tensor_embeddings(self, data, file: bool):
        """
        Arguments:
        :param data: list of words or csv file of words
        :param file: True if data is file, False if list
        :return: tensor values of input words after being embedded through
        Word2Vec Google-News-Negative300
        """
        if file:
            tokens = self.__tokenize_file(data)
        else:
            tokens = self.__tokenize_list(data)
        words = []
        non_words = {}
        for token in tokens:
            for word in token:
                try:
                    words.append(self.wv[word])
                # catch any "key not found" errors for words not in training data
                # storing and writing out just for testing
                except KeyError:
                    non_words[word] = 1 if word not in non_words.keys() else non_words[word] + 1
        # words = np.array(words)
        for i in range(len(words)):
            word = torch.Tensor(words[i])
            words[i] = word
        # # testing code
        # with open('non_word_data.txt', 'w') as outfile:
        #     for key in non_words.keys():
        #         outfile.write(key + " : " + str(non_words[key]))
        # outfile.close()
        return words

    def get_embedding(self, word: str):
        """
        Single word embeddings
        :param word: word
        :return: embedded word vector
        """
        word = np.array(self.wv[word])
        return torch.Tensor(word)
