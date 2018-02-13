# Decoding Decoders: Finding Optimal Representation Spaces for Unsupervised Similarity Tasks

TensorFlow implementation of the models described in the
[Decoding Decoders](https://openreview.net/forum?id=SJOOAEJwf) paper.

This codebase builds on top of [Tensorflow Skip-Thought](https://github.com/tensorflow/models/tree/master/research/skip_thoughts) implementation by Chris Shallue
and uses [SentEval](https://github.com/facebookresearch/SentEval) from Facebook for evaluations on transfer tasks.

The aim is to study how different choices of decoders affect the performance on unsupervised similarity tasks, such as STS.


## Contents
* [Requirements](#requirements)
* [Data Preprocessing](#data-preprocessing)
* [Training](#training)
* [Vocabulary Expansion](#vocabulary-expansion)
* [Evaluation](#evaluation)

## Requirements

This code uses Python 2.7. Please install the requirements in `requirements.txt`.


## Data Preprocessing

### Preparation

You will need to obtain the BookCorpus dataset from [this website](http://yknzhu.wixsite.com/mbweb).

### Quick run
```shell

# Comma-separated list of globs matching the input files. The format of
# the input files is assumed to be a list of newline-separated sentences, where
# each sentence is already tokenized.
INPUT_FILES=<raw input files>

# Location to save the preprocessed training and validation data.
DATA_DIR=<data directory>

# Run the preprocessing script.
python -m skip_thoughts.data.preprocess_dataset \
    --input_files=${INPUT_FILES} \
    --output_dir=${DATA_DIR}
```


## Training

### Training params
We added a couple of new parameters in the `train.py` script.
The most important ones are described here, please see the code to see additional functionality we have added.

**`--decoder=SEQxSKGy`** where `x`, `y` can be `0`, `1`, `2`, and `3`.

SEQ stands for sequence (recurrent) decoder and SKG stands for bag-of-words (BOW) decoder.
* `0` - no decoder of this type is present
* `1` - decoder for the current sentence (Autoencoder)
* `2` - decoders for the previous and next sentences (Skip-Though/FastSent style)
* `3` - decoders for previous, current, and next sentences (Skip-Thought + Autoencoder)
Note that it is possible to combine SEQ and SKG

**`--skipgram_encoder=True|False`**

* `True` The architecture has a bag-of-words (BOW) encoder.
* `False` The architecture has a sequence (RNN) encoder.

Defaults to `False`.

### Quick run
```shell
# Directory containing the preprocessed data.
DATA_DIR=<data directory>

# Directory to save the model. Note: A new folder will be created in here called run_{unixtimestamp}. Into this folder, the model checkpoints will be saved. Also, the FLAGS wile, as well as its dict and json representations will be stored as `flags.pkl`, `config.pkl` and `config.json` respectively. 
RUN_DIR=<run directory>

# Model decoder configuration (choose one of SEQ0SKG2 SEQ0SKG3 SEQ2SKG2 or SEQ3SKG3)
DECODER="SEQ0SKG2"

# Whether to use skipgram (BOW) encoder (choose True or False). Defaults to False.
SKIPGRAM_ENCODER=False

# Run the training script.

python -m skip_thoughts.train \
    --input_file_pattern="${DATA_DIR}/train-?????-of-00100" \
    --run_dir="${RUN_DIR}" \
    --decoder="${DECODER}" \
    --skipgram_encoder="${SKIPGRAM_ENCODER}"
```
This will train a model with an RNN encoder and 2 BOW decoders.

## Vocabulary Expansion

### Preparation

You will need to download the pretrained Google News word2vec vectors, found [here](https://code.google.com/archive/p/word2vec/).
Please see the SkipThought readme for more details on vocab expansion.

### Quick run
```shell
MODEL_DIR=<path to model>
SKIP_THOUGHTS_VOCAB=<path to skipthoughts vocab>
W2VMODEL=<path to W2V model>
LOG_FILE=<path to log file>

python -m skip_thoughts.vocabulary_expansion \
     --skip_thoughts_model="${MODEL_DIR}" \
     --skip_thoughts_vocab="${SKIP_THOUGHTS_VOCAB}" \
     --word2vec_model="${W2VMODEL}" \
     --output_dir="${MODEL_DIR}" \
     > "${LOG_FILE}" 2>&1
```


## Evaluation

### Preparation

You will need to clone the [SentEval repo](https://github.com/facebookresearch/SentEval) and download the data as instructed there.
Then copy our scripts from `sent_eval/evaluation` to the `examples` directory to run.

### The scripts

The [`SentEval` evaluation scripts](/sent_eval/evaluation) either use the encoder output (which we confusingly call `context` here), or the unrolled decoder (which we less confusingly call `unroll` for the similarity and transfer tasks. 

The similarity scripts
[similarity_context.py](/sent_eval/evaluation/similarity_context.py) and 
[similarity_unroll.py](/sent_eval/evaluation/similarity_context.py)
run the `STS*` tasks (`STS12`, `STS13`, `STS14`, `STS15` and `STS16`) of `SentEval`. 

The transfer scripts
[transfer_context.py](/sent_eval/evaluation/transfer_context.py) and 
[transfer_unroll.py](/sent_eval/evaluation/transfer_context.py)
run the transfer tasks (`CR`, `MR`, `MPQA`, `SUBJ`, `SST`, `TREC`, `MRPC`,
                  `SICKRelatedness`, `SICKEntailment` and `STSBenchmark`) of `SentEval`.
                  
Each script runs with 10-fold cross validation, and saves the dictionary of all results as a pickle to the desired location. This can then be used for easy generation of plots and other analysis.

#### Context

The context scripts 
[similarity_context.py](/sent_eval/evaluation/similarity_context.py) and 
[transfer_context.py](/sent_eval/evaluation/transfer_context.py) work for all decoder types. 

The parameters of the context scripts are:

+ `--model_dir` The path to the saved model you want to evaluate. Specifically, this should include this should be a folder containined checkpoint and decoder configuration information produced by [`train.py`](/skip_thoughts/train.py).
+ `--output_results_path` The full path to save the pickle file containing all of the results from this evaluation.

#### Unroll

The unroll scripts 
[similarity_unroll.py](/sent_eval/evaluation/similarity_context.py) and 
[transfer_unroll.py](/sent_eval/evaluation/transfer_context.py) only work for RNN decoder types, and use the [decoder unrolling mechanism](/unrolling_the_decoder.md) discussed in the [Decoding Decoders](https://openreview.net/forum?id=SJOOAEJwf) paper. 

In addition to the parameters of the context scripts (above), the unroll scripts require the following parameters:

+ `--unroll_length` This should be a positive integer, and corresponds to how many time steps each decoder will "unroll" to produce the sentence representation.
+ `--decoder_type` This should be either `'mean'` or `'concat'` and corresponds to either taking the sentence representation as the mean or concatentation over the unrolled hidden states respectively.

### Quick run
The example below is for running `similarity_context.py`, the exact same process will work for the other evaluation scripts.
```shell
# Directory to load the model from
MODEL_DIR=<model directory>

# Which GPU(s) to use (choose from e.g. one of [0 1 0,1])
GPU_IDS=0

# Log file
LOG_FILE=<log file>

# Pickle save path
PICKLE_PATH=<pickle path>

# Run the evaluation script.
CUDA_VISIBLE_DEVICES=$GPU_IDS \
  python -m sent_eval.evaluation.similarity_context \
    --model_dir="${MODEL_DIR}" \
    --output_results_path="${PICKLE_PATH}" \
    > "${LOG_FILE}" 2>&1
```

## Contact

Vitalii Zhelezniak <vitali.zhelezniak@babylonhealth.com>
