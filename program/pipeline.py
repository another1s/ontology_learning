from program.paper_preprocess import killerqueen_release, get_whole_corpus
from program.lstm.testing import classifier
from program.rf.bitethedust import rd_release
from program.ontology_tree import diease, basic_meshtree, initial_tree_data
# a finalized pipeline model including four 3 parts
# a general LSTM-based classifier that will distinguish which category that paper belong tp according to its abstract
# then with the predicted category. the computed tfidf keywords vector will be used as the input of
# corresponding random forest model by which the sub-category of the paper can be determined.

MODEL_PATH = '../model_saved/ml_model/'
DATA_PATH = '../dataset/finalized_result/'

def main(file_name, model_name):
    # calling tfidf tokenizer and generate vectorized keywords representation of the papers
    keywords, vectorized_keywords, tfidf, paper_set = killerqueen_release(file_name, model_name)
    # get the classification result
    result, categories = classifier(MODEL_PATH, DATA_PATH)
    t = list()
    for c in categories:
        t = t +list(c)

    # category should be 0-6, loading random forest model
    # initialize ontology-terms tree
    corpus = get_whole_corpus('D:/cs8740/dataset/pubmed_data/training2.csv')
    forest = basic_meshtree('D:/cs8740/dataset/pubmed_data/data.csv')
    forest = initial_tree_data(filename='../dataset/pubmed_data/dataDepth.csv', forest=forest,corpus=corpus)
    for paper, category in zip(paper_set.corpus_paper_list, t):
        if category == 2 or category == 6:
            predicted_depth = rd_release(paper, category)
        # according to the depth, load corresponding tree
            forest[diease(category).name].compare(paper, predicted_depth)
    return forest

forest = main('../dataset/finalized_result/test2.csv','tfidf')
for tree in forest.values():
    print(tree.node_list)