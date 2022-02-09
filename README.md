# ANLP Project#18 - Benchmarking of CDCR models

This repository is for my project in the course `Applied Natural Language Processing`


## Important files
The ``CONSTANTS.py`` file holds 7 different dictionaries.\
The following will mark the most important attributes of these dictionaries in order to run the script.\
((C) will mark all attributes, which will be changed by the code, all other attributes will be fix for the whole run)
#### configs
+ The `CONFIG`, which holds all path and general information
    + `seed` - the seed for reproducibility
      + This seed is used globally for all randomness - torch, numpy & random
    + `test` / `train` - a boolean, if you want to test or train a model (C)
    + `dataset_name` - which dataset you will be using  (C)
    + `use_singletons` - a boolean, if you want to use singletons or not (C)
    + `use_dep` - boolean, if you want to use dependency parsing for SRL
    + `use_srl` - boolean, if you want to use SRL with AllenNLP and Bert
    + `wiggle` - boolean, if you want to use SRL via the nearest mentions
    + `bert_file` - a path to the pretrained BERT SRL model
      + download here: [bert model](https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz)
    + `elmo_options_file` - a path to the ELMO option file
      + download here: [options file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json)
    + `elmo_weight_file` - a path to the ELMO weight file
      + download here: [weights file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5)

    + `wd_entity_coref_file_path` - a path to the within document CR file.
        + This path will be created by the code itself.
    + `wd_entity_coref_file` - the WDCR file itself.
        + The file will be saved in the `wd_entity_coref_file_path`
    + `gold_files_dir` - the path to the gold files
        + This path will also be created by the code



+ The `EECDCR_CONFIG_DICT` holds all information regarding the `test` part
  + `cd_event_model_path` - a path to the model you want to test on event prediction
  + `cd_entity_model_path` - a path to the model you want to test on entity prediction
  + `wd_entity_coref_file_path` - same as in `CONFIG`
  + `wd_entity_coref_file` - same as in `CONFIG`
  + `event_gold_file_path` - same as `gold_files_dir` in `CONFIG`
  + `event_gold_file` - the file itself (named `CD_test_event_mention_based.key_conll`)
    + If you want to adapt the name, please adapt it in `make_gold_files.py`
  + `entity_gold_file_path` - same as `gold_files_dir` in `CONFIG`
  + `entity_gold_file` - the file itself (named `CD_test_entity_mention_based.key_conll`)
    + If you want to adapt the name, please adapt it in `make_gold_files.py`



+ The `EECDCR_TRAIN_CONFIG_DICT` holds all information regarding the `train` part
  + `char_vocab_path` - a patch to the glove embedding file for char vocab
  + `char_pretrained_path` - a patch to the glove embedding file for the .npy file
  + `glove_path` - the path to the glove embeddings file
    + download here: [Glove Embeddings](https://nlp.stanford.edu/projects/glove/) (glove.6B.300d was used.)
  + `wd_entity_coref_file` - the same as for `CONFIG`


#### data
+ The `dataset_path` holds the path for both, the ECB+ and MEANTime, conll files
+ The `train_test_path` holds the path for the `train_test_split.json`
  + The `train_test_split.json` can be adapted to change the train / test / dev split
+ The `mentions_path` holds the path for the `mentions.json` files for both datasets


## Setup
All path mentioned in the Setup section are the default paths, predefined in the `CONSTANTS.py` dictionaries.\
If you do not want to use these paths, please adapt them accordingly.
### Requirements for testing and training

Regardless of you want to test or train the model, the following steps need to be performed:
+ Download ELMO files
  + [weights file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5)
  + [options file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json)
  + Place both files into ```./resources/word_vector_models/ELMO_Original_55B```
    + Alternatively adapt ``elmo_weight_file`` and ``elmo_options_file`` in the `CONFIG`
+ Download BERT SRL
  + [bert model](https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz)
  + Place the model into ```./resources/word_vector_models/BERT_SRL```
    + Alternatively adapt `bert_file` in the `CONFIG`


### Additional requirements for training
+ Glove Embeddings
    + Download [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) (we used glove.6B.300d).
    + Place in ``./resources/Glove_Embedding_Files/``
      + Alternatively adapt `char_pretrained_path`, `glove_path`, `char_vocab_path` in the `EECDCR_TRAIN_CONFIG_DICT`

These Glove Embeddings are only used for training and 

### Additional requirements for testing
You can also skip the training and use the pretrained models, prepared by Barhom et al.\
These pretrained models were trained on ECB+ and can be downloaded.
+ ECB+ Model from Barhom et al.
  + Download [Event and Entity model](https://drive.google.com/open?id=197jYq5lioefABWP11cr4hy4Ohh1HMPGK)
  + Place both in `./resources/eecdcr_models/from_barhom/`
     + Alternatively adapt `cd_event_model_path`/`cd_entity_model_path` in the `EECDCR_CONFIG_DICT`


## Start the program
So far, you can start the whole program via `python predict_model.py`.\
You will need to enter the following information:
+ Which dataset, do you want to use - ECB+ or MEANTime?
+ Do you want to train/test the model?
+ Do you want to use singletons?
+ Currently, only the [EeCDCR](https://github.com/shanybar/event_entity_coref_ecb_plus) from Barhom et al. can be trained and / or tested.
  + This should be further extended to train / test different models like the CDLM.

All these inputs can also be skipped by removing the input code & adapting the ``CONFIG`` according to your wishes.\
You would need to adapt:
+ ``dataset_name``
  + Can be either `ECB+` or `MEANTime`
+ ``use_singletons``
  + `True`, if you want to use singletons. `False` otherwise.
+ ``train``
  + `True`, if you want to train a new model. `False` otherwise.
+ ``test``
  + `True`, if you want to test a model. `False` otherwise.
+ If both `test` and `train` are set to true, the newly created model will be tested directly after the training.

The results of training and testing will be saved in ``data/output/{date}/{dataset_name}/{starting_time}``.\
(E.g. you start the script on the 01.01.2022, 15:16 with the ECB+ dataset the results will be saved in:\
``data/output/2022-01-01/ECB+/15-16/``)
