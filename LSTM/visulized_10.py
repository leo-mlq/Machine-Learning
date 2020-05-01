from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_lines(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split("<eos>")
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split("<eos>")


def _build_vocab(line):
  data = line.split();
  counter = collections.Counter(data)
  #count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  count_pairs = counter.items()
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id




ten_lines = _read_lines("./simple-examples/data/ptb.valid.txt")[:10]
for i in range(10):
  print('"'+ten_lines[i]+'"'+" where ")
  print(_build_vocab(ten_lines[i]))
  print('\n')
