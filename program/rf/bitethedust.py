from sklearn.ensemble import RandomForestClassifier
import pickle

# 0. Bacterial Infections and Mycoses, 1. virus, 2. Parasitic, 3. Neoplasms, 4. Musculoskeletal Diseases, 5. Digestive System Diseases, 6. Stomatognathic Diseases
# saving 7 seven individual random forest
# the placement will indicate the depth of the paper
def rd_training(data, placement, group):
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(data, placement)
    with open(group + '.pickle', 'w') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

def rd_release(data, group):
    with open(group + '.pickle', 'r') as f:
        clf = pickle.load(f)
    result = clf.predict(data)
    return result