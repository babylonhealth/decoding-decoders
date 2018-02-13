# Copyright 2018 Babylon Partners. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cPickle
import sys

from skip_thoughts import configuration
from skip_thoughts import decode
from skip_thoughts import encoder_manager
from skip_thoughts import experiments

import numpy as np
import tensorflow as tf

import logging

from sent_eval.examples.exutil import dotdict


FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string("model_dir", None,
                       "Directory for saving and loading checkpoints.")
tf.flags.DEFINE_string("output_results_path", None,
                       "Path to save pickled results to.")
tf.flags.DEFINE_bool("use_eos", True,
                     "If to use the eos token during encoder unroll.")
tf.flags.DEFINE_integer("unroll_length", None,
                        "If to use the eos token during encoder unroll.")
tf.flags.DEFINE_string("decoder_type", None,
                       "If to use the eos token during encoder unroll.")

if not FLAGS.model_dir:
  raise ValueError("--model_dir is required.")
if not FLAGS.output_results_path:
  raise ValueError("--output_results_path is required.")
if not FLAGS.unroll_length:
  raise ValueError("--unroll_length is required.")

decoder_types = ['mean', 'concat']
if FLAGS.decoder_type not in decoder_types:
  raise ValueError("--decoder_type must be one of {t}".format(t=decoder_types))

# Set paths to the model.
VOCAB_FILE = os.path.join(FLAGS.model_dir, "vocab.txt")
EMBEDDING_MATRIX_FILE = os.path.join(FLAGS.model_dir, "embeddings.npy")
CHECKPOINT_PATH = FLAGS.model_dir
FLAGS_PICKLE_PATH = os.path.join(FLAGS.model_dir, "flags.pkl")

# Load the configuration used to make the model
with open(FLAGS_PICKLE_PATH, 'r') as f:
  model_flags = cPickle.load(f)

decoder_config = experiments.get_decoder_config(flags=model_flags)
model_config = configuration.model_config(
  input_file_pattern=model_flags.input_file_pattern,
  vocab_size=model_flags.vocab_size,
  batch_size=model_flags.batch_size,
  word_embedding_dim=model_flags.word_dim,
  encoder_dim=model_flags.encoder_dim,
  skipgram_encoder=model_flags.skipgram_encoder,
  sequence_decoder_pre=decoder_config.sequence_decoder_pre,
  sequence_decoder_cur=decoder_config.sequence_decoder_cur,
  sequence_decoder_post=decoder_config.sequence_decoder_post,
  skipgram_decoder_pre=decoder_config.skipgram_decoder_pre,
  skipgram_decoder_cur=decoder_config.skipgram_decoder_cur,
  skipgram_decoder_post=decoder_config.skipgram_decoder_post,
  share_weights_logits=model_flags.share_weights_logits,
  normalise_decoder_losses=model_flags.normalise_decoder_losses,
  skipgram_prefactor=model_flags.skipgram_prefactor,
  sequence_prefactor=model_flags.sequence_prefactor)

# Set up the encoder. Here we are using a single unidirectional model.
# To use a bidirectional model as well, call load_model() again with
# configuration.model_config(bidirectional_encoder=True) and paths to the
# bidirectional model's files. The encoder will use the concatenation of
# all loaded models.
encoder = encoder_manager.EncoderManager()
encoder.load_model(model_config=model_config,
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH,
                   mode='encode-decode')

# Build the decoder
g = encoder.graph
sess = encoder.sessions[0]

tensor_names_global = {
    'word_embedding': 'word_embedding:0'}

tensor_names_pre = {
    'logits': 'logits/logits/decoder_pre:0',
    'decoder_output': 'decoder_pre/decoder_output:0',
    'decoder_state': 'decoder_pre/decoder_state:0'}

tensor_names_post = {
    'logits': 'logits_1/logits/decoder_post:0',
    'decoder_output': 'decoder_post/decoder_output:0',
    'decoder_state': 'decoder_post/decoder_state:0'}

decoder_pre = decode.Decoder(
    g=g,
    tensor_names_decoder=tensor_names_pre,
    tensor_names_global=tensor_names_global)

decoder_post = decode.Decoder(
    g=g,
    tensor_names_decoder=tensor_names_post,
    tensor_names_global=tensor_names_global)


# encodings = encoder.encode(data)


# Set PATHs
current_path = os.path.dirname(__file__)
PATH_TO_SENTEVAL = os.path.join(current_path, '../')
PATH_TO_DATA = os.path.join(current_path, '../data/senteval_data')

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# consider the option of lower-casing or not for bow.
def prepare(params, samples):
  params.batch_size = 32
  # set to 10 to be comparable to published results
  params.kfold = 10
  return


def batcher_steps(steps, decoder_type):
  def batcher(params, batch):
    batch = [" ".join(sent) if sent != [] else " ".join(['.'])
             for sent in batch]

    decode_pre_rep, decode_post_rep = decode.decode(
      sess=sess, data=batch,
      encoder=encoder,
      decoder_pre=decoder_pre,
      decoder_post=decoder_post,
      steps=steps,
      use_eos=FLAGS.use_eos)

    decode_rep_concat = np.concatenate(
      (np.array(decode_pre_rep), np.array(decode_post_rep)), axis=1)

    if decoder_type == 'mean':
      return np.mean(decode_rep_concat, axis=1)

    this_batch_size = len(decode_rep_concat)
    return np.reshape(decode_rep_concat, (this_batch_size, -1))

  return batcher


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
params_senteval = dotdict(params_senteval)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

transfer_tasks = [ #'CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC',
                  'SICKRelatedness', 'SICKEntailment'] #, 'STSBenchmark']

similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']

if __name__ == "__main__":
  batcher = batcher_steps(steps=FLAGS.unroll_length,
                          decoder_type=FLAGS.decoder_type)

  se = senteval.SentEval(params_senteval, batcher, prepare)
  tasks = transfer_tasks
  results = se.eval(tasks)
  f = open(FLAGS.output_results_path, 'wb')
  cPickle.dump(results, f)
  f.close()
