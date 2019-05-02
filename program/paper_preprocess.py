import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from program.analyse import Daedalus, Paper
import re
from nltk.corpus import stopwords
import pickle

def readfile(filename, path = '../dataset/pubmed_data/'):
    csvfile = open(path + filename, 'r', encoding='utf-8 ')
    dict_reader = csv.DictReader(csvfile)
    return dict_reader

def save_to_local(keywords, filename):
    with open(filename, 'a') as f:
        for keyword in keywords:
            f.writelines(str(keyword))
            f.write('\n')
        f.close()

def save_all(paperlist, filename):
    with open(filename, 'a') as f:
        for paper in paperlist:
            f.writelines([paper.label, str(paper.abstract), str(paper.index), str(paper.keywords), str(paper.vectorized_keywords)])
        f.write('\n')
        f.close()

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
        with open('tfidf.pickle','wb+') as f:
            pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        return result, name
    def reload(self, corpus0, vectorizer):
        result = vectorizer.fit_transform(corpus0)
        name = vectorizer.get_feature_names()
        return result, name

def killerqueen():
    # some constant
    keywords_per_paper = 50
    paper = Paper()
    corpus = Daedalus()
    papers = readfile('training2.csv')
    words = stopwords.words('english')
    print(words)
    for row in papers:
        #a = row['\ufefflabel']
        paper.define(row['\ufefflabel'], row['title'], filtering_words(row['abstract']), filtering_words(row['features']), filtering_words(row['authorlist']), row['index'], None)
        filtered_words = [word for word in paper.abstract.split() if word not in stopwords.words('english')]
        d =''
        for r in filtered_words:
            d = d + ' ' + r
        paper.abstract = d
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
    save_to_local(keywords, "keywords.csv")
    save_to_local(vectorized_keywords, "vectorized_keywords")
    for paper, keyword, vectorized_keyword in zip(corpus.corpus_paper_list, keywords, vectorized_keywords):
        paper.computed_features(keywords=keyword, vectorized_keywords=vectorized_keyword)
    save_all(corpus.corpus_paper_list, "comprehensive.csv")
    return keywords, vectorized_keywords, tfidf, corpus

def killerqueen_release(fname, mname):
    keywords_per_paper = 50
    paper = Paper()
    corpus = Daedalus()
    papers = readfile(fname)
    words = stopwords.words('english')
    print(words)
    for row in papers:
        # a = row['\ufefflabel']
        paper.define(row['\ufefflabel'], row['title'], filtering_words(row['abstract']),
                     filtering_words(row['features']), filtering_words(row['authorlist']), None)
        filtered_words = [word for word in paper.abstract.split() if word not in stopwords.words('english')]
        d = ''
        for r in filtered_words:
            d = d + ' ' + r
        paper.abstract = d
        corpus.add_paper(paper)
        corpus.add_author(paper)

    with open(mname + '.pickle', 'rb') as f:
        vectorizer = pickle.load(f)

    FE = FeatureExtraction()
    plaintext = list()
    for paper in corpus.corpus_paper_list:
        t = re.sub("[0-9]", "", paper.abstract)
        plaintext.append(t)
    abstract_tfidf, words = FE.reload(plaintext, vectorizer)
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
    save_to_local(keywords, "keywords.csv")
    save_to_local(vectorized_keywords, "vectorized_keywords")
    for paper, keyword, vectorized_keyword in zip(corpus.corpus_paper_list, keywords, vectorized_keywords):
        paper.computed_features(keywords=keyword, vectorized_keywords=vectorized_keyword)
    return keywords, vectorized_keywords, tfidf, corpus

keywords, vectorized_keywords, tfidf, corpus = killerqueen()