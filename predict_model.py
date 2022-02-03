import pickle
from datetime import datetime
import os

import torch

from EECDCR.all_models.train_model import train_model
from conll_reader import read_CoNLL
from create_wd_document_from_corpus import create_complete_wd_document
from features.build_features import match_allen_srl_structures, load_elmo_embeddings
from features.create_elmo_embeddings import ElmoEmbedding
from make_gold_files import create_gold_files_for_corpus
from mentionsfromjson import loadMentionsFromJson
from run_eecdcr import test_model, run_conll_scorer
from shared.CONSTANTS import CONFIG, EECDCR_TRAIN_CONFIG_DICT, EECDCR_CONFIG_DICT
from srl_things import get_srl_data, find_args_by_dependency_parsing, find_left_and_right_mentions
import logging
import random
import numpy as np

random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])


def shut_up_logger():
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
    logging.getLogger('urllib3.connectionpool').disabled = True
    logging.getLogger('matplotlib').disabled = True
    logging.getLogger('pytorch_transformers').disabled = True
    logging.getLogger('pytorch_pretrained_bert.modeling').disabled = True
    logging.getLogger('allennlp.common.registrable').disabled = True
    logging.getLogger('allennlp.common.from_params').disabled = True
    logging.getLogger('allennlp.data.vocabulary').disabled = True
    logging.getLogger('h5py._conv').disabled = True


def _create_corpus(split: str):
    corpus = read_CoNLL(split=split)
    corpus = loadMentionsFromJson(corpus)

    if CONFIG['use_srl']:
        srl_data = get_srl_data(corpus)
        match_allen_srl_structures(corpus, srl_data, True)
    if CONFIG['use_dep']:
        find_args_by_dependency_parsing(corpus, is_gold=True)
    if CONFIG['wiggle']:
        find_left_and_right_mentions(corpus, is_gold=True)

    elmo_embedder = ElmoEmbedding(CONFIG['elmo_options_file'], CONFIG['elmo_weight_file'])
    load_elmo_embeddings(corpus, elmo_embedder, set_pred_mentions=True)

    return corpus


def _train():
    _train_start = datetime.now()
    # print("Preparing train corpus")
    # train_corpus = _create_corpus('train')
    #
    ############# The creation of the dev corpus is old code & should just be `_create_corpus('dev')`, but will be kept
    # print(f"Preparing dev corpus after {str(datetime.now() - start)}")
    # dev_corpus = read_CoNLL(split="dev")
    # dev_corpus = loadMentionsFromJson(dev_corpus)
    #
    # srl_data = get_srl_data(dev_corpus)
    # match_allen_srl_structures(dev_corpus, srl_data, True)
    #
    # elmo_embedder = ElmoEmbedding(CONFIG['elmo_options_file'], CONFIG['elmo_weight_file'])
    # load_elmo_embeddings(dev_corpus, elmo_embedder, set_pred_mentions=False)
    #
    # with open(f"pickle_data/{CONFIG['dataset_name']}/train_corpus.p", "wb") as f:
    #     pickle.dump(train_corpus, f)
    #
    # with open(f"pickle_data/{CONFIG['dataset_name']}/dev_corpus.p", "wb") as f:
    #     pickle.dump(dev_corpus, f)

    _train_out_dir = f"resources/eecdcr_models/self_trained/{CONFIG['dataset_name']}/"
    if not os.path.exists(_train_out_dir):
        os.makedirs(_train_out_dir)

    with open(f"pickle_data/{CONFIG['dataset_name']}/train_corpus.p", "rb") as f:
        train_corpus = pickle.load(f)

    with open(f"pickle_data/{CONFIG['dataset_name']}/dev_corpus.p", "rb") as f:
        dev_corpus = pickle.load(f)

    train_start = datetime.now()
    print(f"Start model training after {str(datetime.now() - _train_start)}")
    train_model(train_corpus, dev_corpus, _train_out_dir, EECDCR_TRAIN_CONFIG_DICT)
    print(f"Model training finished, after: {str(datetime.now() - train_start)}")

    return _train_out_dir


def _test():
    out_dir = f"data/output/{datetime.today().strftime('%Y-%m-%d')}/{CONFIG['dataset_name']}/{datetime.now().strftime('%H-%M')}/"
    print('Creating test corpus..')
    test_corpus = _create_corpus('test')

    with open(f"pickle_data/{CONFIG['dataset_name']}/test_corpus", "wb") as f:
        pickle.dump(test_corpus, f)

    with open(f"pickle_data/{CONFIG['dataset_name']}/test_corpus", "rb") as f:
        test_corpus = pickle.load(f)

    print('Creating gold files..')
    create_gold_files_for_corpus(test_corpus)

    print('Testing the model..')
    _, _ = test_model(test_corpus, out_dir)
    run_conll_scorer(out_dir)


def _chooseDataset():
    print("Please choose a dataset")
    print("0: ECB+")
    print("1: MEANTime")

    try:
        _inp = int(input(">>> "))
    except ValueError:
        _inp = -1
    return _inp


if __name__ == '__main__':
    # shut_up_logger()

    print("Do you wanna train or test a model?\n0: train\n1: test\n2: train & test")
    test_or_train = int(input(">>> "))

    if test_or_train == 0:
        CONFIG['train'] = True
        CONFIG['test'] = False
    elif test_or_train == 1:
        CONFIG['train'] = False
        CONFIG['test'] = True
    elif test_or_train == 2:
        CONFIG['train'] = True
        CONFIG['test'] = True

    inp = _chooseDataset()

    while inp not in [0, 1, 2]:
        print("Not a valid input.")
        inp = _chooseDataset()

    CONFIG['dataset_name'] = 'ECB+' if inp == 0 else 'MEANTime'
    inp = int(input("Do you want to use singletons?\n0: Yes\n1: No\n>>> "))
    CONFIG['use_singletons'] = inp == 0

    path_exists = os.path.exists(CONFIG['wd_entity_coref_file'].format(CONFIG['dataset_name']))
    if not path_exists:
        print("Creation of the within-document-coref file started.")
        create_complete_wd_document()

    start = datetime.now()

    if CONFIG['train']:
        print("Start training")
        train_out_dir = _train()

        """
            If you (directly) want to test your trained model you need to adapt the model path to the
            same path where the train_function saves the models.
            The used path between "only train" and "only test" differ, to not overwrite the previous best model,
            just because you trained a new one.
        """
        if CONFIG['test'] and CONFIG['train']:
            EECDCR_CONFIG_DICT['cd_event_model_path'] = os.path.join(train_out_dir, 'cd_event_best_model')
            EECDCR_CONFIG_DICT['cd_entity_model_path'] = os.path.join(train_out_dir, 'cd_entity_best_model')

    if CONFIG['test']:
        _test()

    print(f"Complete.\nDuration: {str(datetime.now() - start)}")
