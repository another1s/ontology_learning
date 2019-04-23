import tensorflow as tf
from program.lstm.helperfunction.prepare_data import load_data
from program.lstm.helperfunction.prepare_data import fill_feed_dict, data_preprocessing_release
import pickle
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
    return acc, prediction

def classifier(modelpath, datapath):
    config = {
        "max_len": 32,
        "hidden_size": 72,
        "vocab_size": 60002,
        "embedding_size": 128,
        "n_class": 7,
        "learning_rate": 1e-3,
        "batch_size": 4,
        "train_epoch": 20
    }
    with open(modelpath + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    max_len = 32
    #x_test, y_test = load_data(datapath + "dbpedia_data/dbpedia_csv/test.csv", one_hot=False)
    #x_test, vocab_size = data_preprocessing_release(x_test, tokenizer, max_len=32, max_words=50000)
    x_test, y_test = load_data(datapath + "pubmed_data/test.csv", one_hot=False)
    x_test, vocab_size = data_preprocessing_release(x_test, tokenizer, max_len=32, max_words=60002)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(modelpath + 'model.ckpt.meta')
        saver.restore(sess, modelpath + "model.ckpt")
        graph = tf.get_default_graph()
        name = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print([n.name for n in tf.get_default_graph().as_graph_def().node])
        x = graph.get_tensor_by_name('x:0')
        label = graph.get_tensor_by_name('label:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        prediction = graph.get_tensor_by_name('prediction:0')
        cnt = 0
        test_acc = 0
        predict = list()
        for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
            acc, pre= run_eval_step0(x, label, keep_prob, prediction, sess, (x_batch, y_batch))
            test_acc += acc
            predict.append(pre)
            cnt += 1
    result = ("Test accuracy : %f %%" % (test_acc / cnt * 100))
    print(result)
    return result, predict