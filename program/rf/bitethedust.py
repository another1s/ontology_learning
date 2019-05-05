from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
import csv
from enum import Enum
# 0. Bacterial Infections and Mycoses, 1. virus, 2. Parasitic, 3. Neoplasms, 4. Musculoskeletal Diseases, 5. Digestive System Diseases, 6. Stomatognathic Diseases
# saving 7 seven individual random forest
# the placement will indicate the depth of the paper

class diease(Enum):
    BM = 2
    VI = 0
    PA = 1
    NE = 3
    MD = 4
    DS = 5
    ST = 6



def rd_training(data, placement, group):
    clf = RandomForestClassifier(n_estimators=50)

    clf.fit(data, placement)
    with open(group + '.pickle', 'wb') as handle:
        pickle.dump(clf, handle)

def rd_release(data, group):
    with open(group + '.pickle', 'r') as f:
        clf = pickle.load(f)
    result = clf.predict(data)
    return result

def readcsv(filename):
    f = open(filename, 'r', encoding='utf-8')
    csvfile = csv.DictReader(f)
    return csvfile

def vec_process(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        csvfile = csv.DictReader(f)
        count = 1
        vec = list()
        l = ''
        for row in csvfile:
            count = count + 1
            d = row['data']
            l = l + d
            if count % 5 == 1:
                count = 1
                l = l.split(' ')
                x = list()
                for part in l:
                    if len(part):
                        x.append(int(part))
                vec.append(np.array(x))
                l = ''
    return vec

w = vec_process('vectorized_keywords.csv')


label_data = readcsv('dataDepth.csv')
indices = list()
depth_ = list()
label_ = list()
group = {'BM':[], 'VI':[],'PA':[] ,'NE' :[],'MD':[] , 'DS' :[],'ST':[]}
for row in label_data:

    index = row['index']
    depth = row['depth']
    label = row['label']

    indices.append(index)
    depth_.append(depth)
    label_.append(label)
x= list()
y= list()
for i, j ,z in zip(indices,depth_, label_):
    x.append(w[int(i)])
    y.append(j)
    keys = ['data', 'label']
    value = [w[int(i)], j]
    dat = dict(zip(keys, value))
    s = diease(int(z)).name
    group[s].append(dat)



for die in diease:
    xtrain = list()
    ytrain = list()
    m = die.name
    s = group[m]
    for item in s:
        x = item['data']
        y = item['label']
        xtrain.append(x)
        ytrain.append(y)
    if len(xtrain):
        rd_training(xtrain, ytrain, m)
