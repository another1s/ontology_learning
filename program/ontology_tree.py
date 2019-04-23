import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from program.analyse import Daedalus, Paper
import re


def readfile(filename, path = '../dataset/pubmed_data/'):
    csvfile = open(path + filename, 'r', encoding='utf-8 ')
    dict_reader = csv.DictReader(csvfile)
    return dict_reader


def filtering_words(text):
    s = re.sub("[\'\\\\/\"\\n\]\[]", "", text)
    return s


class FeatureExtraction:
    def c_value(self, corpus0):

        return

    def tfidf(self, corpus0):
        vectorizer = TfidfVectorizer()
        result = vectorizer.fit_transform(corpus0)
        name = vectorizer.get_feature_names()
        return result, name
    def okapi(self, corpus0):

        return


def killerqueen():
    # some constant
    keywords_per_paper = 50
    paper = Paper()
    corpus = Daedalus()
    papers = readfile('file2.csv')
    for row in papers:
        paper.define(row['title'], filtering_words(row['abstract']), filtering_words(row['authorlist']), filtering_words(row['features']), None)
        corpus.add_paper(paper)
        corpus.add_author(paper)

    FE = FeatureExtraction()
    plaintext = list()
    for paper in corpus.corpus_paper_list:
        t = re.sub("[0-9]", "", paper.abstract)
        plaintext.append(t)
    abstract_tfidf, words = FE.tfidf(plaintext)
    print(abstract_tfidf)
    print(words)
    keywords = list()
    vectorized_keywords = list()
    tfidf = abstract_tfidf.A
    for row in tfidf:
        k = list()
        seq = np.argsort(-row)
        k_words_seq = seq[:keywords_per_paper]
        vectorized_keywords.append(k_words_seq)
        _keywords = list()
        for element in k_words_seq:
            _keywords.append(words[element])
        keywords.append(_keywords)
    print(keywords)

    return keywords, vectorized_keywords, tfidf, corpus

