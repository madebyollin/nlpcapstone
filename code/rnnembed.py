"""Goal: use embeddings of words to do a simple RNN language model."""
import numpy as np
import tensorflow as tf
import spacy.en
import en_core_web_md

VERBOSE     = True
CORPUS_LOC  = "/homes/iws/brandv2/nlp/corpus/reuters.txt"

def log(*args):
    if VERBOSE: print(*args)

log("Loading spacy English...")
spacy_en = en_core_web_md.load()

log("Reading corpus...")
with open(CORPUS_LOC, encoding="iso-8859-1") as f:
    corpus = spacy_en(f.read()).to_array([spacy.attrs.ID])
    corpus = np.ravel(corpus) # Flatten to 1-d array

NUM_WORDS   = len(corpus)
VOCAB_SIZE  = len(spacy_en.vocab)
EMBED_SIZE  = spacy_en.vocab.vectors_length
SEQ_LEN     = 10
HIST_LEN    = SEQ_LEN-1
LSTM_SIZE   = 300
NUM_SAMPLES = NUM_WORDS - SEQ_LEN
BATCH_SIZE  = 20
EPOCHS      = 1000

log("Loaded corpus: vocab size is", VOCAB_SIZE,
    "max seq length is", SEQ_LEN)

def embeddings(xs):
    """
    Input:  an iterable of integer lexeme ids
    Return: a matrix of word embeddings with dims [num_words, embed_size]
    """
    output = np.empty([len(xs), EMBED_SIZE])
    for i, lexid in enumerate(xs):
        output[i] = spacy_en.vocab[lexid].vector
    return output

def yield_all_pairs():
    i = 0
    while True:
        hist_indices = (i + np.arange(HIST_LEN)) % NUM_SAMPLES
        batch_x = corpus[hist_indices]
        batch_y = corpus[i+HIST_LEN]
        i = (i+1) % NUM_SAMPLES
        yield embeddings(batch_x), batch_y

sample_iter = yield_all_pairs()

def get_batch(size):
    x = np.empty([size, HIST_LEN, EMBED_SIZE], dtype=np.float32)
    y = np.empty([size], dtype=np.int)
    for i in range(size):
        x[i], y[i] = next(sample_iter)
    return x, y

def run_model():
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32,
                                [None, HIST_LEN, EMBED_SIZE],
                                name="inputs")

        labels = tf.placeholder(tf.int64,
                                [None], 
                                name="labels")

        cell    = tf.contrib.rnn.LSTMCell(LSTM_SIZE)
        # in_size = inputs.get_shape()[0]
        # init_state = cell.zero_state(in_size, tf.float32)
        states, final_state_tuple = tf.nn.dynamic_rnn(
                cell,
                inputs,
                dtype=tf.float32)

        def final_proj(x):
            init_w = tf.truncated_normal([LSTM_SIZE, VOCAB_SIZE], stddev=0.3)
            w = tf.Variable(init_w)
            b = tf.Variable(tf.zeros([VOCAB_SIZE]))
            return tf.nn.xw_plus_b(x, w, b)

        final_states = states[:, -1, :]
        logits = final_proj(final_states)
        loss   = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits))

        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss)

        correct = tf.equal(labels, tf.argmax(tf.nn.softmax(logits), 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        saver = tf.train.Saver()

    # config = tf.ConfigProto(log_device_placement=True)
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        avg_loss = 0
        avg_steps = 10
        eval_x, eval_y = get_batch(1000)
        for t in range(EPOCHS):
            print("Step", t)

            batch_inputs, batch_labels = get_batch(BATCH_SIZE)
            train_feed = {
                inputs: batch_inputs,
                labels: batch_labels
            }
            _, cur_loss = sess.run([train_step, loss], train_feed)

            avg_loss += cur_loss
            if t % avg_steps == 0:
                if t > 0: avg_loss /= avg_steps
                print("\tAverage loss:\t", avg_loss)
                acc = sess.run([accuracy], {inputs: eval_x, labels: eval_y})
                print("\tTrain acc:\t", acc)
        path = saver.save(sess, "model.ckpt")
        print("Saved model to ", path)

if __name__ == "__main__":
    run_model()

def test_batch():
    for _ in range(2):
        xs, ys = get_batch(3)
        print(xs, "=>", ys)
        print("xs shape", xs.shape)
        print("ys shape", ys.shape)
