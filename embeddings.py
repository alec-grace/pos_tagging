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


# paperswithcode tend to have datasets with research papers
def main():
    if os.path.exists("goog_news.wordvectors"):
        wv = KeyedVectors.load("goog_news.wordvectors", mmap='r')
    else:
        wv = api.load('word2vec-google-news-300')
        wv.save("goog_news.wordvectors")

    print(wv.most_similar('twitter'))


if __name__ == "__main__":
    main()
