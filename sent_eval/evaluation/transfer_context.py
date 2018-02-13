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
from skip_thoughts import encoder_manager

import tensorflow as tf

import logging
import skip_thoughts.experiments as experiments

from sent_eval.examples.exutil import dotdict


FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string("model_dir", None,
                       "Directory for saving and loading checkpoints.")
tf.flags.DEFINE_string("output_results_path", None,
                       "Path to save pickled results to.")
tf.flags.DEFINE_bool("use_eos", True,
                     "If to use the eos token during encoder unroll.")

if not FLAGS.model_dir:
  raise ValueError("--model_dir is required.")
if not FLAGS.output_results_path:
  raise ValueError("--output_results_path is required.")

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
                   checkpoint_path=CHECKPOINT_PATH)

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
  params.batch_size = 128
  # set to 10 to be comparable to published results
  params.kfold = 10
  return


def batcher(params, batch):
  batch = [" ".join(sent) if sent != [] else " ".join(['.'])
           for sent in batch]
  return encoder.encode(batch, use_eos=FLAGS.use_eos)


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
params_senteval = dotdict(params_senteval)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC',
                  'SICKRelatedness', 'SICKEntailment', 'STSBenchmark']

similarity_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']

if __name__ == "__main__":
  se = senteval.SentEval(params_senteval, batcher, prepare)
  tasks = transfer_tasks
  results = se.eval(tasks)
  f = open(FLAGS.output_results_path, 'wb')
  cPickle.dump(results, f)
  f.close()
