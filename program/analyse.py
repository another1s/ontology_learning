import pandas

'''
features of papers used including:
    1.main title
    2.abstract contents
    3.author
    4.keywords 
    for both main title and abstract contents, dictionary and word counts need to be attained
expected result: a dict containing text, word count, and a dictionary 
for authors, authors of each paper need to be ranked by their contribution(just weight them in order)
expected result: a list of authors
'''
DATA_PATH = ""
DATA_NAME = ""


class Author:
    author_name = None
    author_publication = list()


class Paper:
    title = None
    abstract = None
    author_list = list()
    word_dict = dict()
    keywords = list()
    label = None
    def define(self, label, title, abstract,keywords, author_list, word_dict=None):
        self.label = label
        self.title = title
        self.abstract = abstract
        self.author_list = author_list
        self.keywords = keywords
        self.word_dict = word_dict


class Daedalus:
    corpus_word_dictionary = dict()
    corpus_author_list = dict()
    corpus_paper_list = list()

    def query_something(self, subject, target, sign):
        author_profile = Author()
        if sign:
            if any(target):
                for word in target:
                    if word == subject:
                        return 1
                return 0
            else:
                return 0
        elif not sign:
            b = any(target.values())
            if any(target.values()):
                for element in target.values():
                    if type(element) == type(author_profile):
                        if element.author_name == subject:
                            return subject
                return 0
            else:
                return 0
        else:
            return 0

    def add_paper(self, paper):
        p = Paper()
        paper_title = paper.title.split()
        word_dict = dict()
        for word in paper_title:
            t1 = self.query_something(word, word_dict.keys(), 1)
            if t1 == 1:
                word_dict[word] = word_dict[word] + 1
            if t1 == 0:
                word_dict[word] = 1
            t2 = self.query_something(word, self.corpus_word_dictionary.keys(), 1)
            if t2 == 1:
                self.corpus_word_dictionary[word] = self.corpus_word_dictionary[word] + 1
            if t2 == 0:
                self.corpus_word_dictionary[word] = 1

        paper_abstract = paper.abstract.split()
        for word in paper_abstract:
            t1 = self.query_something(word, word_dict.keys(), 1)
            if t1 == 1:
                word_dict[word] = word_dict[word] + 1
            if t1 == 0:
                word_dict[word] = 1
            t2 = self.query_something(word, self.corpus_word_dictionary.keys(), 1)
            if t2 == 1:
                self.corpus_word_dictionary[word] = self.corpus_word_dictionary[word] + 1
            if t2 == 0:
                self.corpus_word_dictionary[word] = 1
        p.define(paper.label, paper.title, paper.abstract, paper.author_list, paper.keywords, word_dict)
        self.corpus_paper_list.append(p)
    # fetch the authors of each paper
    # rank is defined as average author rank

    def add_author(self, paper):
        paper_authors = paper.author_list.split(',')
        person = Author()
        for author in paper_authors:
            one = self.query_something(author, self.corpus_author_list, 0)
            if one is 0:
                person.author_name = author
                person.author_publication.append(paper)
                self.corpus_author_list[author] = person
            else:
                self.corpus_author_list[one].author_publication.append(paper)

#    def whole_paper(self, paper):

