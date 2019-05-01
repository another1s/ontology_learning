import numpy as np
MPATH = '../model_saved/ml_model/'
DPATH = '../dataset/'


class Node:
    father = None
    children_list = []
    node_name = None
    depth = None
    paper_collection = list()

    def __init__(self, father, word, depth, papers):
        self.father = father
        self.children_list = []
        self.node_name = word
        self.depth = depth
        self.paper_collection = papers
        return


# algorithm: travel through the tree and exploring the name
# of each nodes. choosing the node which paper_keywords contain at the next node.
# if none of the them is included, comparing tfidf vector distance between papers and calculating an average depth
# then create an new node as a child node at that level of tree

class Tree:
    root = None
    node_list = []
    flag = 0

    def __init__(self, root):
        self.root = root
        self.node_list = []
        return

    def merge(self, paper, node):
        node.paper_collection.append(paper)
        return len(node.paper_collection)

    def insert(self, paper, node):
        new_son = Node(father=node, word=paper.keywords[0], depth=node.depth+1, papers=[paper])
        node.children_list.append(new_son)
        return "new branches created"

    def compare(self, target, depth, threshold=0.7):
        similarity = list()
        p = None
        for node in self.node_list:
            if node.depth == depth:
                for paper in node.paper_collection:
                    x = paper.vectorized_keywords
                    y = target.vectorized_keywords
                    similarity.append(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)))
                p = node.paper_collection

        if np.max(similarity) > threshold:
            index = similarity.index(np.max(similarity))
            correspond_node = p[index].node
            print("paper belong to this node" + correspond_node.node_name + ":" + self.merge(target, correspond_node))

        else:
            index = similarity.index(np.max(similarity))
            correspond_node = p[index].node.father
            print(correspond_node.node_name)
            print(self.insert(target, correspond_node))

    def search(self, node, paper_keywords):
        if node.node_name == paper_keywords:
            self.flag = 1
            return
        if len(node.children_list) > 0:
            for childNode in node.children_list:
                self.search(childNode, paper_keywords)
                if self.flag == 1:
                    return
        return

    def tree_dfs(self, paper_keywords):
        self.flag = 0
        self.search(self.root, paper_keywords)
        if (self.flag == 1):
            return True
        else:
            return False


def save_tree(t):

    return

def load_tree(root):
    tree = root
    return tree