# File: embeddings.py
# Author: Alec Grace
# Created: 20 Feb 2024
# Purpose:
#   Create word embeddings from a given dataset

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
from gensim.models import Word2Vec


def main():

    with open("test_corpus.txt", 'r') as infile:
        lines = infile.readlines()
    infile.close()
    corpus = ''
    for line in lines:
        corpus = corpus + line
    corpus.replace("\n", " ")
    corpus.replace("*", " ")
    sents = sent_tokenize(corpus)
    data = []
    for sent in sents:
        temp = []
        for word in word_tokenize(sent):
            temp.append(word)
        data.append(temp)

    with open("test_corpus2.txt", 'r') as infile:
        lines = infile.readlines()
    infile.close()

    corpus2 = ''
    for line in lines:
        corpus2 = corpus2 + line.replace("\n", " ")
    corpus2.replace("\n", " ")
    corpus2.replace("*", " ")
    sents2 = sent_tokenize(corpus2)
    data2 = []
    for sent in sents2:
        temp = []
        for word in word_tokenize(sent):
            temp.append(word)
        data2.append(temp)

    test_word1 = "heart"
    test_word2 = "organ"
    test_word3 = "guts"
    test_word4 = "OpenAI"
    test_word5 = "tech"

    mod_min = 1
    vec_size = 5
    window = 5
    sg = 1
    model = gensim.models.Word2Vec(data, min_count=mod_min, vector_size=vec_size, window=window)
    model2 = gensim.models.Word2Vec(data, min_count=mod_min, vector_size=vec_size, window=window, sg=sg)

    model3 = gensim.models.Word2Vec(data2, min_count=mod_min, vector_size=vec_size, window=window)
    model4 = gensim.models.Word2Vec(data2, min_count=mod_min, vector_size=vec_size, window=window, sg=sg)

    with open('test_results.txt', 'a') as outfile:
        outfile.write("\n\nmin_count = " + str(mod_min) + "\nvector_size = " + str(vec_size) +
                      "\nwindow = " + str(window) + "\nskip gram = " + str(sg))
        outfile.write("\nbow similarity between " + test_word1 + " and " + test_word2 +
                      ": " + str(model.wv.similarity(test_word1, test_word2)))
        outfile.write("\nsg similarity between " + test_word1 + " and " + test_word2 +
                      ": " + str(model2.wv.similarity(test_word1, test_word2)))
        outfile.write("\nbow similarity between " + test_word1 + " and " + test_word3 +
                      ": " + str(model.wv.similarity(test_word1, test_word3)))
        outfile.write("\nsg similarity between " + test_word1 + " and " + test_word3 +
                      ": " + str(model2.wv.similarity(test_word1, test_word3)))
        outfile.write("\nbow similarity between " + test_word4 + " and " + test_word5 +
                      ": " + str(model3.wv.similarity(test_word4, test_word5)))
        outfile.write("\nsg similarity between " + test_word4 + " and " + test_word5 +
                      ": " + str(model4.wv.similarity(test_word4, test_word5)))
    outfile.close()


if __name__ == "__main__":
    main()
