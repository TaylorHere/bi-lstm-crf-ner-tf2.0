# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/12/3 7:28 下午
# @Author: wuchenglong


from utils import tokenize,build_vocab,read_vocab
import tensorflow as tf
from model import NerModel
import tensorflow_addons as tf_ad
import os
import numpy as np
from args_help import args
from logger import logger

logger.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

if not (os.path.exists(args.vocab_file) and os.path.exists(args.tag_file)):
    logger.info("building vocab file")
    build_vocab([args.train_path], args.vocab_file, args.tag_file)
else:
    logger.info("vocab file exits!!")


vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
text_sequences ,label_sequences= tokenize(args.train_path,vocab2id,tag2id)


train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences, label_sequences))
train_dataset = train_dataset.shuffle(len(text_sequences)).batch(args.batch_size, drop_remainder=True)

logger.info("hidden_num:{}, vocab_size:{}, label_size:{}".format(args.hidden_num, len(vocab2id), len(tag2id)))

model = NerModel(
    hidden_num=args.hidden_num,
    vocab_size=len(vocab2id),
    label_size=len(tag2id),
    embedding_size=args.embedding_size)

model.compile(
    optimizer=tf.keras.optimizers.Adam(args.lr),
    # loss=
)


ckpt = tf.train.Checkpoint(optimizer=model.optimizer, model=model)
ckpt.restore(tf.train.latest_checkpoint(args.output_dir))
ckpt_manager = tf.train.CheckpointManager(ckpt,
                                          args.output_dir,
                                          checkpoint_name='model.ckpt',
                                          max_to_keep=3)

model.fit(train_dataset, batch_size=args.batch_size, epoch=args.epoch)
# @tf.function


# best_acc = 0
# step = 0
# for epoch in range(args.epoch):
#     for _, (text_batch, labels_batch) in enumerate(train_dataset):
#         step = step + 1
#         loss, logits, text_lens = train_one_step(text_batch, labels_batch)
#         if step % 20 == 0:
#             accuracy = get_acc_one_step(logits, text_lens, labels_batch)
#             logger.info('epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
#             if accuracy > best_acc:
#               best_acc = accuracy
#               ckpt_manager.save()
#               logger.info("model saved")


logger.info("finished")