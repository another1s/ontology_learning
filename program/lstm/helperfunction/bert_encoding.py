from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer
from bert_serving.client import BertClient
from program.paper_preprocess import killerqueen
import re
model_path = 'D:/cs8740/model_saved/word_embedding/uncased_L-24_H-1024_A-16/uncased_L-24_H-1024_A-16/'

def bert(corpus):
    bc = BertClient()
    result = list()
    pattern = r', |\. '
    for paper in corpus.corpus_paper_list:
        abstract = re.split(pattern, paper.abstract)
        while '' in abstract:
            abstract.remove('')
        word_encodings = bc.encode(abstract)
        result.append(word_encodings)
    return result

keywords, vectorized_keywords, tfidf, corpus = killerqueen()
result = bert(corpus)
print(result)

