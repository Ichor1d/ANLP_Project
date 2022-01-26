# ANLP Project#18 - Benchmarking of CDCR models

This repository is for my project in the course `Applied Natural Language Processing`

## Setup
In order to start everything you need:\
Download the following [archive](https://drive.google.com/file/d/197jYq5lioefABWP11cr4hy4Ohh1HMPGK/view), exract the files,
and place the models cd_entity_best_model and cd_event_best_model into ```./resources/eecdcr_models```.

The other model files will be downloaded directly from the code or can be downloaded manually:
* ELMO
1) [weights](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5)
2) [options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json)
3) Place both files into ```./resources/word_vector_models/ELMO_Original_55B```
* BERT SRL
1) [model](https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz)
2) Place the model into ```./resources/word_vector_models/BERT_SRL```

Alternatively you can place them wherever and adapt the path in `CONSTANTS.py`.

## Start the program
So far, you can start the whole program via `python predict_model.py`.\
You will need to enter the following information:
+ Which dataset, do you want to use - ECB+ or MEANTime?
+ Do you want to train/test the model?
+ Do you want to use singletons?
+ Currently, only the [EeCDCR](https://github.com/shanybar/event_entity_coref_ecb_plus) from Barhom et al. can be trained and / or tested.
  + This should be further extended to train / test different models like the CDLM.

The results of training and testing will be saved in ``data/output/{date}/{datasetName}/{time}``.\
If you want to use the self-trained models, you will need to move the models from the above path into the `./resources/eecdcr_models` path.
