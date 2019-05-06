from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from program.lstm.helperfunction.prepare_data import *
from tensorflow.contrib.rnn import LSTMCell
FLAGS = None


class ABLSTM(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len], name='x')
        self.label = tf.placeholder(tf.int32, [None], name='label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def build_graph(self):
        print("building graph")
        # Word embedding
        embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True, name='embeddings_var')
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x, name='batch_embedded')

        rnn_outputs, _ = bi_rnn(LSTMCell(self.hidden_size),
                                LSTMCell(self.hidden_size),
                                inputs=batch_embedded, dtype=tf.float32)

        fw_outputs, bw_outputs = rnn_outputs

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)

        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len), name='alpha'))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1], name='r0'))
        r = tf.squeeze(r, name='r1')
        h_star = tf.tanh(r, name='r2')  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, self.keep_prob, name='h_drop')

        # Fully connected layer（dense layer)
        FC_W = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1), name='FC_W')
        FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]), name='FC_b')
        y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b, name='y_hat')

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label), name='loss')

        # prediction
        self.prediction = tf.argmax(tf.nn.softmax(y_hat), 1, name='prediction')

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0, name='gradients')

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")
