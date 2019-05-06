from program.lstm.main import load_data
from program.lstm.helperfunction.prepare_data import data_preprocessing_v2, fill_feed_dict
import pickle
from keras_preprocessing.sequence import pad_sequences
import numpy as np
def make_test_feed_dict0(x, label, keep_prob, batch):
    a = batch[1]
    feed_dict = {x: batch[0],
                 label: batch[1],
                 keep_prob: 1.0}
    return feed_dict

def run_eval_step0(x, label, keep_prob, prediction, sess, batch):
    feed_dict = make_test_feed_dict0(x, label, keep_prob, batch)
    prediction = sess.run(prediction, feed_dict=feed_dict)
    acc = np.sum(np.equal(prediction, batch[1])) / len(prediction)
    return acc


config = {
    "max_len": 32,
    "hidden_size": 72,
    "vocab_size": 50002,
    "embedding_size": 128,
    "n_class": 7,
    "learning_rate": 1e-3,
    "batch_size": 4,
    "train_epoch": 20
}


max_len = 32
x_train, y_train = load_data("D:/cs8740/dataset/pubmed_data/training.csv", sample_ratio=1, one_hot=False)
x_test0, y_test = load_data("D:/cs8740/dataset/pubmed_data/test.csv", one_hot=False)
x_train, x_test, vocab_size, train_words, test_words, tokenizer = data_preprocessing_v2(x_train, x_test0, max_len=32, max_words=60000)
with open('D:/cs8740/model_saved/word_embedding/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('D:/cs8740/model_saved/word_embedding/tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)

test_idx = tokenizer1.texts_to_sequences(x_test0)
test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
print(test_padded)
print(x_test)
