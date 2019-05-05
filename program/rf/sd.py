import pymongo, os

# 检索作者名兼去重


total = set()

def add_total(author_list):
    for author in author_list:
        if author in total:
            continue
        else:
            total.add(author)

client = pymongo.MongoClient('127.0.0.1', 27017)
db = client.test
db_list = ['Analytical','Anatomy','Anthropology','Chemicals','Disciplines', 'Diseases','Health',
'Humanities','Information','Organisms', 'Phenomena', 'Psychiatry', 'Technology']
for name in db_list:
    print(name)
    coll = db['%s_publication'%name]
    coll_list = list(coll.find({},{'authors':1}))
    print(len(coll_list))
    for author in coll_list:
        add_total(author['authors'])

# write into txt
with open('author_list.txt',mode='w',encoding='utf-8') as file_handle:
    for name in total:
        file_handle.write(name)
        file_handle.write('\n')

# read from txt, name_list is result
name_list = []
rs = os.path.exists('author_list.txt')
if rs == True:
    with open('author_list.txt',mode='r', encoding='utf-8') as file_handle:
        contents = file_handle.readlines()
        for name in contents:
            name = name.strip('\n')
            name_list.append(name)

print(name_list)


class Author:
    name = None
    id = None
    email = None
    information = None
    imageUrl = None
    career = list()
    organization = list()

    def define(self, profile):
        self.name = profile['name']
        self.id = profile['id']
        self.email = profile['email']
        self.information = profile['information']
        self.imageUrl = profile['imageUrl']
        self.career = profile['career']
        self.organization = profile['organization']



