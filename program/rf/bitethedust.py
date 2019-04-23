from sklearn.ensemble import RandomForestClassifier
import pickle

# 0. Bacterial Infections and Mycoses, 1. virus, 2. Parasitic, 3. Neoplasms, 4. Musculoskeletal Diseases, 5. Digestive System Diseases, 6. Stomatognathic Diseases
def training(data, placement):
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(data, placement)
    with open(placement + '.pickle', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
