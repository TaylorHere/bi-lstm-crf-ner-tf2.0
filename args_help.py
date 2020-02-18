# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/12/3 1:44 下午
# @Author: wuchenglong

import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_path", type=str, default="/home/jovyan/shared/data/train.txt",help="train file")
parser.add_argument("--test_path", type=str, default="/home/jovyan/shared/data/test.txt",help="test file")
parser.add_argument("--output_dir", type=str, default="/home/jovyan/shared/data/checkpoints/",help="output_dir")
parser.add_argument("--log_dir", type=str, default="/home/jovyan/shared/log/",help="logs")
parser.add_argument("--vocab_file", type=str, default="/home/jovyan/shared/data/vocab.txt",help="vocab_file")
parser.add_argument("--tag_file", type=str, default="/home/jovyan/shared/data/tags.txt",help="tag_file")
parser.add_argument("--batch_size", type=int, default=32,help="batch_size")
parser.add_argument("--hidden_num", type=int, default=512,help="hidden_num")
parser.add_argument("--embedding_size", type=int, default=300,help="embedding_size")
parser.add_argument("--embedding_file", type=str, default=None,help="embedding_file")
parser.add_argument("--epoch", type=int, default=100,help="epoch")
parser.add_argument("--lr", type=float, default=1e-3,help="lr")
parser.add_argument("--require_improvement", type=int, default=100,help="require_improvement")
args = parser.parse_args()
