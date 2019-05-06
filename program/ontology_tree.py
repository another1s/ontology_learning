import numpy as np
import csv
import json
from program.paper_preprocess import Paper
from enum import Enum
MPATH = '../model_saved/ml_model/'
DPATH = '../dataset/'


class Node:
    father = None
    children_list = []
    node_name = None
    depth = None
    paper_collection = list()

    def __init__(self, father, children, word, depth, papers):
        self.father = father
        self.children_list = children
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

    def find_node(self, name):
        for node in self.node_list:
            if name == node.node_name:
                return node
        return 0

    def add_son(self, father, son):
        counter = 0
        f_index = 0
        s_index = 0
        for node in self.node_list:
            if node.node_name == father:
                f_index = counter
            if node.node_name == son:
                s_index = counter
            counter = counter + 1
        self.node_list[f_index].children_list.append(self.node_list[s_index])

    def add_node(self, node):
        self.node_list.append(node)
        return

    def merge(self, paper, node):
        node.paper_collection.append(paper)
        return len(node.paper_collection)

    def insert(self, paper, node):
        new_son = Node(father=node, word=paper.keywords[0], children=[] ,depth=node.depth+1, papers=[paper])
        node.children_list.append(new_son.node_name)
        self.node_list.append(new_son)
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
    '''
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
    '''
def save_tree(root, tree):
    t = dict()
    t['root'] = tree.root
    t['node_list'] = list()
    for node in tree.node_list:
        detail =dict()
        detail['father'] = node.father
        detail['node_name'] = node.node_name
        detail['depth'] = node.depth
        l1 = list()
        l2 = list()
        for i in node.paper_collection:
            l1.append(i.index)
        for j in node.children_list:
            l2.append(j.node_name)
        detail['paper_collection'] = l1
        detail['children_list'] = l2
        t['node_list'].append(detail)
    js_str = json.dumps(t)
    with open(root + '_tree.json', 'w', encoding='utf-8') as f:
        f.write(js_str)
    return "saving object successfully"

def load_paper_data(m):
    with open("comprehensive.csv",'r') as f:
        dict_reader = csv.DictReader(f)
    i = dict_reader['index'].index(m)
    p = Paper()
    p.title = dict_reader['title'][i]
    p.abstract = dict_reader['abstract'][i]
    p.author_list = dict_reader['author_list'][i]
    p.word_dict = dict_reader['word_dict'][i]
    p.keywords = dict_reader['keywords'][i]
    p.vectorized_keywords = dict_reader['vectorized_keywords'][i]
    p.label = dict_reader['label'][i]
    p.index = dict_reader['index'][i]
    return p

def load_tree(root):
    t = Tree()
    with open(root + '_tree.json', 'w', encoding='utf-8') as f:
        load_dict = json.loads(f)
    t.root = load_dict['root']
    for n in load_dict['node_list']:
        father = n['father']
        node_name = n['node_name']
        depth = n['depth']
        children_list = n['children_list']
        paper_collection = n['paper_collection']
        node = Node(father=father, word=node_name, depth=depth,children=children_list,papers=paper_collection)
        t.node_list.append(node)

    # reconstruct
    for n in t.node_list:
        w = list()
        for m in n.paper_collection:
            paper = load_paper_data(m)
            w.append(paper)
        n.paper_collection = w
    return t



def initial_tree_data(filename,forest,corpus):
    f = open(filename, 'r', encoding='utf-8')
    csvfile = csv.DictReader(f)
    for row in csvfile:
        father_node = row['father node']
        index = row['index']
        depth = row['depth']
        label = row['label']
        paper = corpus.find_paper(index)
        tree_name = diease(int(label)).name

        node = forest[tree_name].find_node(father_node)
        if not node:
            node.paper_collection.append(paper)

    return forest

class diease(Enum):
    BM = 2
    VI = 0
    PA = 1
    NE = 3
    MD = 4
    DS = 5
    ST = 6
def basic_meshtree(filename):
    forest = {'BM': [], 'VI': [], 'PA': [], 'NE': [], 'MD': [], 'DS': [], 'ST': []}
    for key in forest.keys():
        forest[key] = Tree(key)
    f = open(filename, 'r', encoding='utf-8')
    csvfile = csv.DictReader(f)
    for row in csvfile:
        label = row['label']
        group = diease(int(label)).name
        nodename = row['node_name']
        depth = row['depth']
        father = row['father']
        N = Node(father=father,children=[], depth=depth, word=nodename, papers=[])
        forest[group].add_node(N)

    f = open(filename, 'r', encoding='utf-8')
    csvfile = csv.DictReader(f)
    for row in csvfile:
        father = row['father']
        nodename = row['node_name']
        label = row['label']
        group = diease(int(label)).name
        forest[group].add_son(father=father, son=nodename)

    return forest



ontology_forest = basic_meshtree('data.csv')
