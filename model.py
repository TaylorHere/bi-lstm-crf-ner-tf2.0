# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/12/3 7:15 下午
# @Author: wuchenglong


import tensorflow as tf
import tensorflow_addons as tf_ad


class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size):
        super(NerModel, self).__init__()
        self.num_hidden = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.transition_params = None

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.biLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden_num, return_sequences=True))
        self.dense = tf.keras.layers.Dense(label_size)

        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)),
                                             trainable=False)
        self.dropout = tf.keras.layers.Dropout(0.5)

    # @tf.function
    def call(self, text,labels=None,training=None):
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        # -1 change 0
        inputs = self.embedding(text)
        inputs = self.dropout(inputs, training)
        logits = self.dense(self.biLSTM(inputs))

        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = tf_ad.text.crf_log_likelihood(logits, label_sequences, text_lens)
            self.transition_params = tf.Variable(self.transition_params, trainable=False)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens


    def train_one_step(self, text_batch, labels_batch):
        with tf.GradientTape() as tape:
            logits, text_lens, log_likelihood = self.model(text_batch, labels_batch,training=True)
            loss = - tf.reduce_mean(log_likelihood)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, logits, text_lens


    def get_acc_one_step(self, logits, text_lens, labels_batch):
        paths = []
        accuracy = 0
        for logit, text_len, labels in zip(logits, text_lens, labels_batch):
            viterbi_path, _ = tf_ad.text.viterbi_decode(logit[:text_len], self.model.transition_params)
            paths.append(viterbi_path)
            correct_prediction = tf.equal(
                tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path], padding='post'),
                                    dtype=tf.int32),
                tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]], padding='post'),
                                    dtype=tf.int32)
            )
            accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
        accuracy = accuracy / len(paths)
        return accuracy
