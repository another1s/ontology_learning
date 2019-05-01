from program.paper_preprocess import killerqueen_release
from program.lstm.testing import classifier
from program.rf.bitethedust import *
from program.ontology_tree import load_tree
# a finalized pipeline model including four 3 parts
# a general LSTM-based classifier that will distinguish which category that paper belong tp according to its abstract
# then with the predicted category. the computed tfidf keywords vector will be used as the input of
# corresponding random forest model by which the sub-category of the paper can be determined.

MODEL_PATH = '../model_saved/ml_model/'
DATA_PATH = '../dataset/pubmed_data/'

def main(file_name, model_name):

    keywords, vectorized_keywords, tfidf, corpus = killerqueen_release(file_name, model_name)
    result, categories = classifier(MODEL_PATH, DATA_PATH)
    # category should be 0-6, loading random forest model
    for paper, category in zip(corpus.corpus_paper_list, categories):
        predicted_depth = rd_release(paper, category)
        # according to the depth, load corresponding tree
        tree = load_tree(category)
        tree.compare(paper, predicted_depth)

