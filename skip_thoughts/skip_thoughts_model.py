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
#   - Added code to support sequence and skipgram
#     decoders for previous, current, and next sentences
#   - Added _build_skipgram_decoder function
#   - Added encode-decode mode
#   - Added code to support decoder unrolling
#   - Added summaries
# ==============================================================================
"""Skip-Thoughts model for learning sentence vectors.

The model is based on the paper:

  "Skip-Thought Vectors"
  Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel,
  Antonio Torralba, Raquel Urtasun, Sanja Fidler.
  https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf

Layer normalization is applied based on the paper:

  "Layer Normalization"
  Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
  https://arxiv.org/abs/1607.06450
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from skip_thoughts.ops import gru_cell
from skip_thoughts.ops import input_ops

import skip_thoughts.summaries as summaries


def random_orthonormal_initializer(shape, dtype=tf.float32,
                                   partition_info=None):
  # pylint: disable=unused-argument
  """Variable initializer that produces a random orthonormal matrix."""
  if len(shape) != 2 or shape[0] != shape[1]:
    raise ValueError("Expecting square shape, got %s" % shape)
  _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
  return u


class SkipThoughtsModel(object):
  """Skip-thoughts model."""

  def __init__(self, config, mode="train", input_reader=None):
    """Basic setup. The actual TensorFlow graph is constructed in build().

     Args:
        config: Object containing configuration parameters.
        mode: "train", "eval" or "encode".
        input_reader: Subclass of tf.ReaderBase for reading the input
          serialized tf.Example protocol buffers. Defaults to
          TFRecordReader.

     Raises:
        ValueError: If mode is invalid.
     """
    if mode not in ["train", "eval", "encode", "encode-decode"]:
      raise ValueError("Unrecognized mode: %s" % mode)

    self.config = config
    self.mode = mode
    self.reader = input_reader if input_reader else tf.TFRecordReader()

    # Initializer used for non-recurrent weights.
    self.uniform_initializer = tf.random_uniform_initializer(
      minval=-self.config.uniform_init_scale,
      maxval=self.config.uniform_init_scale)

    # Input sentences represented as sequences of word ids.
    # "encode" is the source sentence,
    # "decode_pre" is the previous sentence and
    # "decode_post" is the next sentence.
    # Each is an int64 Tensor with shape [batch_size, padded_length].
    self.encode_ids = None
    self.decode_pre_ids = None
    self.decode_post_ids = None

    # Boolean masks distinguishing real words (1) from padded words (0).
    # Each is an int32 Tensor with shape [batch_size, padded_length].
    self.encode_mask = None
    self.decode_pre_mask = None
    self.decode_post_mask = None

    # Input sentences represented as sequences of word embeddings.
    # Each is a float32 Tensor with shape
    # [batch_size, padded_length, emb_dim].
    self.encode_emb = None
    self.decode_pre_emb = None
    self.decode_post_emb = None

    # The output from the sentence encoder.
    # A float32 Tensor with shape [batch_size, num_gru_units].
    self.thought_vectors = None

    # The cross entropy losses and corresponding weights of the decoders.
    # Used for evaluation.
    self.target_cross_entropy_losses = []
    self.target_cross_entropy_loss_weights = []

    # The total loss to optimize.
    self.total_loss = None

  tvars = tf.trainable_variables()
  with tf.name_scope('trainable_vars'):
    # Summarise all variables
    for var in tvars:
      summaries.variable_summaries(var, summary_prefix=var.name)

  def build_inputs(self):
    """Builds the ops for reading input data.

    Outputs:
      self.encode_ids
      self.decode_pre_ids
      self.decode_post_ids
      self.encode_mask
      self.decode_pre_mask
      self.decode_post_mask
    """
    if self.mode == "encode":
      # Word embeddings are fed from an external vocabulary which has
      # possibly been expanded (see vocabulary_expansion.py).
      encode_ids = None
      decode_pre_ids = None
      decode_post_ids = None

      encode_mask = tf.placeholder(
        tf.int8, (None, None), name="encode_mask")
      decode_pre_mask = None
      decode_post_mask = None
    elif self.mode == "encode-decode":
      # Word embeddings are fed from an external vocabulary which has
      # possibly been expanded (see vocabulary_expansion.py).
      encode_ids = None
      decode_pre_ids = None
      decode_post_ids = None

      encode_mask = tf.placeholder(
        tf.int8, (None, None), name="encode_mask")
      decode_pre_mask = tf.placeholder(
        tf.int8, (None, None), name="decode_pre_mask")
      decode_post_mask = tf.placeholder(
        tf.int8, (None, None), name="decode_post_mask")
    else:
      # Prefetch serialized tf.Example protos.
      input_queue = input_ops.prefetch_input_data(
        self.reader,
        self.config.input_file_pattern,
        shuffle=self.config.shuffle_input_data,
        capacity=self.config.input_queue_capacity,
        num_reader_threads=self.config.num_input_reader_threads)

      # Deserialize a batch.
      serialized = input_queue.dequeue_many(self.config.batch_size)
      encode, decode_pre, decode_post = input_ops.parse_example_batch(
        serialized)

      encode_ids = encode.ids
      decode_pre_ids = decode_pre.ids
      decode_post_ids = decode_post.ids

      encode_mask = encode.mask
      decode_pre_mask = decode_pre.mask
      decode_post_mask = decode_post.mask

    self.encode_ids = encode_ids
    self.decode_pre_ids = decode_pre_ids
    self.decode_post_ids = decode_post_ids

    self.encode_mask = encode_mask
    self.decode_pre_mask = decode_pre_mask
    self.decode_post_mask = decode_post_mask

  def build_word_embeddings(self):
    """Builds the word embeddings.

    Inputs:
      self.encode_ids
      self.decode_pre_ids
      self.decode_post_ids

    Outputs:
      self.encode_emb
      self.decode_pre_emb
      self.decode_post_emb
    """
    if self.mode == "encode":
      # Word embeddings are fed from an external vocabulary which has
      # possibly been expanded (see vocabulary_expansion.py).
      encode_emb = tf.placeholder(tf.float32, (
        None, None, self.config.word_embedding_dim), "encode_emb")
      # No sequences to decode.
      decode_pre_emb = None
      decode_post_emb = None
    elif self.mode == "encode-decode":
      # Word embeddings are fed from an external vocabulary which has
      # possibly been expanded (see vocabulary_expansion.py).
      encode_emb = tf.placeholder(tf.float32, (
        None, None, self.config.word_embedding_dim), "encode_emb")
      decode_pre_emb = tf.placeholder(tf.float32, (
        None, None, self.config.word_embedding_dim), "decode_pre_emb")
      decode_post_emb = tf.placeholder(tf.float32, (
        None, None, self.config.word_embedding_dim), "decode_post_emb")

      word_emb = tf.get_variable(
        name="word_embedding",
        shape=[self.config.vocab_size, self.config.word_embedding_dim],
        initializer=self.uniform_initializer)
    else:
      if self.config.pretrained_word_emb_file:
        word_emb_values = np.load(self.config.pretrained_word_emb_file)
        assert self.config.word_embedding_dim == word_emb_values.shape[1]

        word_emb = tf.get_variable(
          name="word_embedding",
          initializer=tf.constant(word_emb_values, dtype=tf.float32),
          trainable=self.config.word_emb_trainable)
      else:
        word_emb = tf.get_variable(
          name="word_embedding",
          shape=[self.config.vocab_size, self.config.word_embedding_dim],
          initializer=self.uniform_initializer)

      encode_emb = tf.nn.embedding_lookup(word_emb, self.encode_ids)
      decode_pre_emb = tf.nn.embedding_lookup(word_emb, self.decode_pre_ids)
      decode_post_emb = tf.nn.embedding_lookup(word_emb, self.decode_post_ids)

    self.encode_emb = encode_emb
    self.decode_pre_emb = decode_pre_emb
    self.decode_post_emb = decode_post_emb

  def _initialize_gru_cell(self, num_units):
    """Initializes a GRU cell.

    The Variables of the GRU cell are initialized in a way that exactly matches
    the skip-thoughts paper: recurrent weights are initialized from random
    orthonormal matrices and non-recurrent weights are initialized from random
    uniform matrices.

    Args:
      num_units: Number of output units.

    Returns:
      cell: An instance of RNNCell with variable initializers that match the
        skip-thoughts paper.
    """
    return gru_cell.LayerNormGRUCell(
      num_units,
      w_initializer=self.uniform_initializer,
      u_initializer=random_orthonormal_initializer,
      b_initializer=tf.constant_initializer(0.0))

  def build_encoder(self):
    """Builds the sentence encoder.

    Inputs:
      self.encode_emb
      self.encode_mask

    Outputs:
      self.thought_vectors

    Raises:
      ValueError: if config.bidirectional_encoder is True and config.encoder_dim
        is odd.
    """
    with tf.variable_scope("encoder") as scope:
      if self.config.skipgram_encoder:
        w_dim = self.config.word_embedding_dim
        e_dim = self.config.encoder_dim

        # For FastSent sentence emb dim is the same word emb dim
        assert w_dim == e_dim

        encode_emb = tf.reshape(self.encode_emb, [-1, w_dim])
        weights = tf.to_float(tf.reshape(self.encode_mask, [-1, 1]))
        encode_emb = encode_emb * weights
        seq_len = tf.shape(self.encode_mask)[-1]
        encode_emb = tf.reshape(encode_emb, tf.stack([-1, seq_len, w_dim]))

        self.thought_vectors = tf.reduce_sum(encode_emb,
                                             axis=1,
                                             name="thought_vectors")
        return

      length = tf.to_int32(tf.reduce_sum(self.encode_mask, 1), name="length")

      if self.config.bidirectional_encoder:
        if self.config.encoder_dim % 2:
          raise ValueError(
            "encoder_dim must be even when using a bidirectional encoder.")
        num_units = self.config.encoder_dim // 2
        cell_fw = self._initialize_gru_cell(num_units)  # Forward encoder
        cell_bw = self._initialize_gru_cell(num_units)  # Backward encoder
        _, states = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=self.encode_emb,
          sequence_length=length,
          dtype=tf.float32,
          scope=scope)
        thought_vectors = tf.concat(states, 1, name="thought_vectors")
      else:
        cell = self._initialize_gru_cell(self.config.encoder_dim)
        _, state = tf.nn.dynamic_rnn(
          cell=cell,
          inputs=self.encode_emb,
          sequence_length=length,
          dtype=tf.float32,
          scope=scope)
        # Use an identity operation to name the Tensor in the Graph.
        thought_vectors = tf.identity(state, name="thought_vectors")

    self.thought_vectors = thought_vectors

  def _build_sequence_decoder(self, name, embeddings, targets, mask,
                              initial_state, reuse_logits):
    """Builds a sentence decoder.

    Args:
      name: Decoder name.
      embeddings: Batch of sentences to decode; a float32 Tensor with shape
        [batch_size, padded_length, emb_dim].
      targets: Batch of target word ids; an int64 Tensor with shape
        [batch_size, padded_length].
      mask: A 0/1 Tensor with shape [batch_size, padded_length].
      initial_state: Initial state of the GRU. A float32 Tensor with shape
        [batch_size, num_gru_cells].
      reuse_logits: Whether to reuse the logits weights.
    """
    # Decoder RNN.
    cell = self._initialize_gru_cell(self.config.encoder_dim)
    with tf.variable_scope(name) as scope:
      # Add a padding word at the start of each sentence (to correspond to the
      # prediction of the first word) and remove the last word.
      decoder_input = tf.pad(
        embeddings[:, :-1, :], [[0, 0], [1, 0], [0, 0]], name="input")
      length = tf.reduce_sum(mask, 1, name="length")
      decoder_output, decoder_state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=decoder_input,
        sequence_length=length,
        initial_state=initial_state,
        scope=scope)

      decoder_output = tf.identity(decoder_output, name='decoder_output')
      decoder_state = tf.identity(decoder_state, name='decoder_state')

    # Stack batch vertically.
    decoder_output = tf.reshape(decoder_output, [-1, self.config.encoder_dim])

    # Logits.
    with tf.variable_scope("logits", reuse=reuse_logits) as scope:
      logits = tf.contrib.layers.fully_connected(
        inputs=decoder_output,
        num_outputs=self.config.vocab_size,
        activation_fn=None,
        weights_initializer=self.uniform_initializer,
        scope=scope)

      logits = tf.identity(logits, name='logits/' + name)

    # If we just want the encode-decode, stop here
    if self.mode == "encode-decode":
      return None

    targets = tf.reshape(targets, [-1])
    weights = tf.to_float(tf.reshape(mask, [-1]))

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=targets, logits=logits)
    batch_loss = tf.reduce_sum(losses * weights)

    if self.config.normalise_decoder_losses:
      batch_loss = batch_loss / self.config.num_sequence_decoders

    batch_loss = batch_loss * self.config.sequence_prefactor

    tf.losses.add_loss(batch_loss)

    tf.summary.scalar("losses/" + name, batch_loss)

    self.target_cross_entropy_losses.append(losses)
    self.target_cross_entropy_loss_weights.append(weights)

  def _build_skipgram_decoder(self, name, targets, mask, reuse_logits):

    """Builds a skipgram decoder.

    Args:
      name: Decoder name.
      targets: Batch of target word ids; an int64 Tensor with shape
        [batch_size, padded_length].
      mask: A 0/1 Tensor with shape [batch_size, padded_length].
      reuse_logits: Whether to reuse the logits weights.
    """

    with tf.variable_scope("skipgram_logits", reuse=reuse_logits) as scope:
      skipgram_logits = tf.contrib.layers.fully_connected(
        inputs=self.thought_vectors,
        num_outputs=self.config.vocab_size,
        activation_fn=None,
        weights_initializer=self.uniform_initializer,
        scope=scope)

    if self.mode == "encode-decode":
      return None

    multiples = tf.stack([1, tf.shape(targets)[-1], 1])
    skipgram_logits = tf.expand_dims(skipgram_logits, 1)
    skipgram_logits = tf.tile(skipgram_logits, multiples)
    skipgram_logits = tf.reshape(skipgram_logits, [-1, self.config.vocab_size])

    targets = tf.reshape(targets, [-1])
    weights = tf.to_float(tf.reshape(mask, [-1]))

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=targets, logits=skipgram_logits)

    batch_loss = tf.reduce_sum(losses * weights)

    if self.config.normalise_decoder_losses:
      batch_loss = batch_loss / self.config.num_skipgram_decoders

    batch_loss = batch_loss * self.config.skipgram_prefactor

    tf.losses.add_loss(batch_loss)

    tf.summary.scalar("losses/" + name, batch_loss)

    self.target_cross_entropy_losses.append(losses)
    self.target_cross_entropy_loss_weights.append(weights)

  def build_sequence_decoders(self):
    """Builds the sentence decoders.

    Inputs:
      self.decode_pre_emb
      self.decode_post_emb
      self.decode_pre_ids
      self.decode_post_ids
      self.decode_pre_mask
      self.decode_post_mask
      self.thought_vectors

    Outputs:
      self.target_cross_entropy_losses
      self.target_cross_entropy_loss_weights
    """
    reuse_logits = False
    if self.config.sequence_decoder_pre:
      # Pre-sentence decoder.
      self._build_sequence_decoder("decoder_pre", self.decode_pre_emb,
                                   self.decode_pre_ids, self.decode_pre_mask,
                                   self.thought_vectors, reuse_logits)
      reuse_logits = True

    if self.config.sequence_decoder_post:
      # Post-sentence decoder. Logits weights are reused.
      self._build_sequence_decoder("decoder_post", self.decode_post_emb,
                                   self.decode_post_ids, self.decode_post_mask,
                                   self.thought_vectors, reuse_logits)
      reuse_logits = True

    if self.config.sequence_decoder_cur:
      # Cur-sentence decoder. Logits weights are reused.
      self._build_sequence_decoder("decoder_cur", self.encode_emb,
                                   self.encode_ids, self.encode_mask,
                                   self.thought_vectors, reuse_logits)

  def build_skipgram_decoders(self):
    """Builds the sentence decoders.

    Inputs:
      self.decode_pre_ids
      self.decode_post_ids
      self.decode_pre_mask
      self.decode_post_mask
      self.thought_vectors

    Outputs:
      self.target_cross_entropy_losses
      self.target_cross_entropy_loss_weights
    """
    reuse_logits = False
    if self.config.skipgram_decoder_pre:
      # Pre-sentence decoder.
      self._build_skipgram_decoder("skipgram_pre",
                                   self.decode_pre_ids, self.decode_pre_mask,
                                   reuse_logits)
      reuse_logits = True

    if self.config.skipgram_decoder_post:
      # Post-sentence decoder. Logits weights are reused.
      self._build_skipgram_decoder("skipgram_post",
                                   self.decode_post_ids, self.decode_post_mask,
                                   reuse_logits)
      reuse_logits = True

    if self.config.skipgram_decoder_cur:
      # Cur-sentence decoder. Logits weights are reused.
      self._build_skipgram_decoder("skipgram_cur",
                                   self.encode_ids, self.encode_mask,
                                   reuse_logits)

  def build_decoders(self):
    if self.mode != "encode":
      self.build_sequence_decoders()
      self.build_skipgram_decoders()

  def build_loss(self):
    """Builds the loss Tensor.

    Outputs:
      self.total_loss
    """
    if self.mode not in ["encode", 'encode-decode']:
      total_loss = tf.losses.get_total_loss()
      tf.summary.scalar("losses/total", total_loss)

      self.total_loss = total_loss

  def build_global_step(self):
    """Builds the global step Tensor.

    Outputs:
      self.global_step
    """
    self.global_step = tf.contrib.framework.create_global_step()

  def build(self):
    """Creates all ops for training, evaluation or encoding."""
    self.build_inputs()
    self.build_word_embeddings()
    self.build_encoder()
    self.build_decoders()
    self.build_loss()
    self.build_global_step()
