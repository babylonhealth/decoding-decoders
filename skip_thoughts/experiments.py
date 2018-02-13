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
"""Easy classes for decoder configurations."""


def get_decoder_config(flags):
    n_seq_decoders, n_skipgram_decoders = (
        int(flags.decoder[3]), int(flags.decoder[-1]))

    assert n_seq_decoders in [0, 1, 2, 3]
    assert n_skipgram_decoders in [0, 1, 2, 3]

    decoder_config = DecoderConfig()

    if n_seq_decoders in [1, 3]:
        decoder_config.sequence_decoder_cur = True

    if n_seq_decoders in [2, 3]:
        decoder_config.sequence_decoder_pre = True
        decoder_config.sequence_decoder_post = True

    if n_skipgram_decoders in [1, 3]:
        decoder_config.skipgram_decoder_cur = True

    if n_skipgram_decoders in [2, 3]:
        decoder_config.skipgram_decoder_pre = True
        decoder_config.skipgram_decoder_post = True

    return decoder_config


class DecoderConfig:
    def __init__(self):
        self.sequence_decoder_pre = False
        self.sequence_decoder_cur = False
        self.sequence_decoder_post = False
        self.skipgram_decoder_pre = False
        self.skipgram_decoder_cur = False
        self.skipgram_decoder_post = False
