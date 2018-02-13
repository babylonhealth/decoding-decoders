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

import tensorflow as tf


def variable_summaries(var, summary_prefix):
    """Attach a lot of summaries to a Tensor
    (for TensorBoard visualization)."""
    mean = tf.reduce_mean(var)
    tf.summary.scalar('{sp} mean'.format(
        sp=summary_prefix), mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('{sp} stddev'.format(
        sp=summary_prefix), stddev)
    tf.summary.scalar('{sp} max'.format(
        sp=summary_prefix), tf.reduce_max(var))
    tf.summary.scalar('{sp} min'.format(
        sp=summary_prefix), tf.reduce_min(var))
    tf.summary.histogram('{sp} histogram'.format(
        sp=summary_prefix), var)
