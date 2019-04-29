import tensorflow as tf
from program.lstm.testing import classifier

MPATH = '../model_saved/ml_model/'
DPATH = '../dataset/'


class Node:
    father = None
    children_list = list()
    node_name = None

    def __init__(self, father, children, word):
        self.father = father
        self.children_list.append(children)
        self.node_name.append(word)

# algorithm: travel through the tree and exploring the name
# of each nodes. choosing the node which paper_keywords contain at the next node.
# if none of the them is included, comparing tfidf vector distance between papers and calculating an average depth
# then create an new node as a child node at that level of tree

class Tree(Node):
    root = None
    node_list = list()
    def merge(self):

        return
    def insert(self):

        return
    def tree_dfs(tree, paper_keywords):
        return

# receiving
def sheerheartattack(paper, tree):
    tree.tree_dfs(tree, paper.keywords)

    return tree

