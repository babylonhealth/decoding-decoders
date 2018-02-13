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

import numpy as np
import tensorflow as tf

from skip_thoughts import skip_thoughts_encoder


class Decoder:
    def __init__(
            self,
            g,
            tensor_names_decoder,
            tensor_names_global):
        self.g = g
        self.tensors_decoder = _names_to_tensors(
            g=self.g, x=tensor_names_decoder)
        self.tensors_global = _names_to_tensors(
            g=self.g, x=tensor_names_global)

        self.state = self.tensors_decoder['decoder_state']

        self.embedding_dim = tf.shape(
            self.tensors_global['word_embedding'])[-1]

        self.batch_size, self.decoder_seq_len = _get_batch_seq_len(
            self.tensors_decoder['decoder_output'])

        self.softmax_flat = tf.nn.softmax(
            logits=self.tensors_decoder['logits'])

        self.softmax_w_flat = tf.matmul(
            self.softmax_flat, self.tensors_global['word_embedding'])

        self.softmax_w = tf.reshape(
            self.softmax_w_flat,
            (self.batch_size, self.decoder_seq_len, self.embedding_dim))


def _is_tensor_name(x):
    if ":" in x:
        return True
    return False


def _get_tensor_or_op(g, x):
    if _is_tensor_name(x):
        return g.get_tensor_by_name(x)

    return g.get_operation_by_name(x)


def _get_batch_seq_len(x):
    x_sh = tf.shape(x)
    return x_sh[0], x_sh[1]


def _names_to_tensors(g, x):
    return {k: _get_tensor_or_op(g, v) for k, v in x.iteritems()}


def unroll_decoder(
        sess,
        encoder_embeddings,
        encoder_mask,
        decoder_name,
        decoder_softmax_w_embs,
        decoder_state,
        steps=5):
    n_input_sequences = encoder_embeddings.shape[0]
    sequence_dim = encoder_embeddings.shape[2]
    start_tokens = np.zeros(shape=(n_input_sequences, 1, sequence_dim))
    decoder_input = start_tokens
    feed_dict = {
        "encode_emb:0": encoder_embeddings,
        "encode_mask:0": encoder_mask}
    dec_emb_feed = "{decoder_name}_emb:0".format(decoder_name=decoder_name)
    dec_mask_feed = "{decoder_name}_mask:0".format(decoder_name=decoder_name)
    all_states = None

    for _ in range(steps):
        len_seq = decoder_input.shape[1]
        decode_mask = np.ones((n_input_sequences, len_seq))
        feed_dict.update({dec_emb_feed: decoder_input,
                          dec_mask_feed: decode_mask})
        softmax_w_embs, states = sess.run(
            (decoder_softmax_w_embs, decoder_state),
            feed_dict=feed_dict)
        states_expanded = np.expand_dims(states, axis=1)
        if all_states is None:
            all_states = states_expanded
        else:
            all_states = np.concatenate((all_states, states_expanded), axis=1)
        decoder_input = np.concatenate((start_tokens, softmax_w_embs), axis=1)
    return all_states


def decode(sess,
           data,
           encoder,
           decoder_pre,
           decoder_post,
           use_norm=True,
           verbose=True,
           batch_size=128,
           use_eos=False,
           steps=5):
    data = encoder.encoders[0]._preprocess(data=data, use_eos=use_eos)

    pre_states = []
    post_states = []

    batch_indices = np.arange(0, len(data), batch_size)
    for batch, start_index in enumerate(batch_indices):
        if verbose:
            tf.logging.info("Batch %d / %d.", batch, len(batch_indices))

        (encoder_embeddings,
         encoder_mask) = skip_thoughts_encoder._batch_and_pad(
            data[start_index:start_index + batch_size])

        pre_states.extend(unroll_decoder(
            sess=sess,
            encoder_embeddings=encoder_embeddings,
            encoder_mask=encoder_mask,
            decoder_name='decode_pre',
            decoder_softmax_w_embs=decoder_pre.softmax_w,
            decoder_state=decoder_pre.state,
            steps=steps))

        post_states.extend(unroll_decoder(
            sess=sess,
            encoder_embeddings=encoder_embeddings,
            encoder_mask=encoder_mask,
            decoder_name='decode_post',
            decoder_softmax_w_embs=decoder_post.softmax_w,
            decoder_state=decoder_post.state,
            steps=steps))

    if use_norm:
        pre_states = [v / np.linalg.norm(v) for v in pre_states]
        post_states = [v / np.linalg.norm(v) for v in post_states]

    return pre_states, post_states
