# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Changes by Babylon Partners
#   - Modified skip_thoughts/vocabulary_expansion.py to
#     prepare pretrained embeddings
# ==============================================================================


import os
import collections
import tensorflow as tf
import numpy as np
import gensim


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("vocab_file", None,
                       "Existing vocab file."
                       "The file format is a list of newline-separated words, "
                       "where the word id is the corresponding 0-based index "
                       "in the file. The 0th word must correspond to the <eos> "
                       "token and the 1st word to the <unk> token.")
tf.flags.DEFINE_string("word2vec_model", None,
                       "File containing a word2vec model in binary format.")
tf.flags.DEFINE_string("output_dir", None, "Output directory.")

tf.logging.set_verbosity(tf.logging.INFO)


def _get_vocabulary():
  """Loads the model vocabulary.


  Returns:
    vocab: A dictionary of word to id.
  """
  if not FLAGS.vocab_file:
    raise RuntimeError("No vocab file specified.")

  tf.logging.info("Loading existing vocab file.")
  vocab = collections.OrderedDict()
  with tf.gfile.GFile(FLAGS.vocab_file, mode="r") as f:
    for i, line in enumerate(f):
      word = line.strip().decode("utf-8")
      assert word not in vocab, "Attempting to add word twice: %s" % word
      vocab[word] = i
  tf.logging.info("Read vocab of size %d from %s",
                  len(vocab), FLAGS.vocab_file)
  return vocab


def main(unused_argv):
    vocab = _get_vocabulary()
    tf.logging.info("Loading word2vec model.")
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
      FLAGS.word2vec_model, binary=True)
    tf.logging.info("Loaded word2vec model.")

    vocab_size = len(vocab)
    init_embeddings = np.zeros(shape=(vocab_size, w2v_model.vector_size),
                               dtype=np.float32)
    eos_vector = np.zeros(shape=(1, w2v_model.vector_size),
                          dtype=np.float32)
    eos_vector[0][0] = -1
    unk_vector = np.zeros(shape=(1, w2v_model.vector_size),
                          dtype=np.float32)
    unk_vector[0][-1] = -1

    tf.logging.info("Building embedding matrix.")
    for word, idx in vocab.items():
      if word in w2v_model:
        init_embeddings[idx] = w2v_model[word]
      else:
        init_embeddings[idx] = unk_vector
    init_embeddings[0] = eos_vector
    init_embeddings[1] = unk_vector
    embeddings_file = os.path.join(FLAGS.output_dir, "init_embeddings.npy")
    if not os.path.exists(FLAGS.output_dir):
      os.makedirs(FLAGS.output_dir)
    np.save(embeddings_file, init_embeddings)


if __name__ == "__main__":
  tf.app.run()
