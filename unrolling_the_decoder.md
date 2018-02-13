# Unrolling the Decoder

## Introduction

As discussed in the decoding decoders paper, the optimal space for an RNN plus softmax projection is
obtained by unrolling the decocer and using the concatentation of its hidden states as the 
representation of the input (see figure below).
![Unrolling an RNN Decoder](/images/unroll.png)

## Code

In order use the decoder at inference time, in contrast to the original [TensorFlow SkipThoughts implementation](https://github.com/tensorflow/models/tree/master/research/skip_thoughts), we need to load the entire graph. 

First, we load the encoder.

```{python}
flags.data_dir = ...
flags.uni_vocab_file = ...
flags.uni_embeddings_file = ...
flags.uni_checkpoint_path = ...
flags.decoder = ...

decoder_config = experiments.get_decoder_config(flags=flags)

uni_config = configuration.model_config(
    sequence_decoder_pre=decoder_config.sequence_decoder_pre,
    sequence_decoder_cur=decoder_config.sequence_decoder_cur,
    sequence_decoder_post=decoder_config.sequence_decoder_post,
    skipgram_decoder_pre=decoder_config.skipgram_decoder_pre,
    skipgram_decoder_cur=decoder_config.skipgram_decoder_cur,
    skipgram_decoder_post=decoder_config.skipgram_decoder_post)

encoder = encoder_manager.EncoderManager()

encoder.load_model(uni_config, 
                   flags.uni_vocab_file,
                   flags.uni_embeddings_file, 
                   flags.uni_checkpoint_path,
                   mode='encode-decode')
```
We then pull the graph and session object from the encoder
```{python}
g, sess = encoder.graph, encoder.sessions[0]
```
In order to perform the unrolling, we need the names of tensors involed in the unrolling process. 
To help you, if tracking the names of tensors is diffcult, you can always modify the architecture post-training in such
a way that data-flow tensors aquire a speific name using `tf.identity` for example:
```{python}
tensor_to_name = tf.identity(tensor_to_name, name='name_of_tensor_to_name')
```
Now this tensor can be accessed using the `get_tensor_by_name(...)` method of the `tf.Graph`. 
In this case, the tensor would acquire the name `name_of_tensor_to_name:0` since it is the zero-th tensor produced
by the `tf.identity` op named `name_of_tensor_to_name`.

For each decoder we need the following tensors:
+ The logits (for example `decoder_pre_logits:0`)
+ The decoder output (for example `decoder_pre_output:0`)
+ The decoder state (for example `decoder_pre_state:0`)
as well as the word embedding matrix, for example `word_embedding:0`.

Using these, we can define dictionaries of the tensors necessary for the unrolling - one global dictionary, and one specific to each decoder
```
tensor_names_global = {
    'word_embedding': 'word_embedding:0'}

tensor_names_pre = {
    'logits': 'decoder_pre_logits:0',
    'decoder_output': 'decoder_pre_output:0',
    'decoder_state': 'decoder_pre_state:0'}

tensor_names_post = {
    'logits': 'decoder_post_logits:0',
    'decoder_output': 'decoder_post_output:0',
    'decoder_state': 'decoder_post_state:0'}
```
To build the decoders, we use these dictionaries to creeate instances of the [Decoder class](/skip_thoughts/decode.py)
```
decoder_pre = decode.Decoder(
    g=g,
    tensor_names_decoder=tensor_names_pre,
    tensor_names_global=tensor_names_global)

decoder_post = decode.Decoder(
    g=g,
    tensor_names_decoder=tensor_names_post,
    tensor_names_global=tensor_names_global)
```
With this setup, we can now do some unrolling. 
```
batch_size = 2
unroll_steps = 5

data = [
    "and wow",
    "hey !",
    "what's this thing suddenly coming towards me very fast ?",
    "very very fast" 
    "so big and flat and round , it needs a big wide sounding name like ow ound round ground !"
    "that's it !" 
    "that's a good name – ground !"
    "i wonder if it will be friends with me ?"]

decode_pre_rep, decode_post_rep = decode.decode(
    sess=sess, data=data,
    encoder=encoder, decoder_pre=decoder_pre, decoder_post=decoder_post,
    batch_size=batch_size, use_norm=True, steps=unroll_steps)
```
The vector representations `decode_pre_rep` and `decode_post_rep` are the unrolled representations for the prev and post decoders respectively. They are aligned in sentences, and can be concatenated to produce a single representation
```
decode_rep_concat = np.concatenate(
    (np.array(decode_pre_rep), np.array(decode_post_rep)), axis=1)
```
which can then be used for downstream tasks, such as similarity and transfer.


