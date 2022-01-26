from datetime import datetime, timedelta
import json
import os

from EECDCR.all_models.train_model import train_model
from conll_reader import read_CoNLL
from features.build_features import match_allen_srl_structures, load_elmo_embeddings
from features.create_elmo_embeddings import ElmoEmbedding
from mentionsfromjson import loadMentionsFromJson
from run_eecdcr import test_model, run_conll_scorer
from shared.CONSTANTS import CONFIG, EECDCR_TRAIN_CONFIG_DICT, train_test_path, meantimeNameConverter, \
    EECDCR_CONFIG_DICT
from srl_things import get_srl_data
import time
from tqdm import tqdm


def _chooseDataset():
    print("Please choose a dataset")
    print("0: ECB+")
    print("1: MEANTime")

    try:
        inp = int(input(">>> "))
    except ValueError:
        inp = -1
    return inp


if __name__ == '__main__':
    out_dir = f"data/output/{datetime.today().strftime('%Y-%m-%d')}/{CONFIG['dataset_name']}/{datetime.now().strftime('%H-%M')}/"

    if os.path.exists(EECDCR_CONFIG_DICT["wd_entity_coref_file"]):
        os.remove(EECDCR_CONFIG_DICT["wd_entity_coref_file"])

    print("Do you wanna train or test a model?\n0: train\n1: test\n2: train & test")
    CONFIG['test'] = int(input(">>> ")) == 1

    start = time.time()
    inp = _chooseDataset()

    while inp not in [0, 1, 2]:
        print("Not a valid input.")
        inp = _chooseDataset()

    CONFIG['dataset_name'] = 'ECB+' if inp == 0 else 'MEANTime'
    inp = int(input("Do you want to use singletons?\n0: Yes\n1: No\n>>> "))
    CONFIG['use_singletons'] = inp == 0

    train_test_dict = json.load(open(train_test_path[CONFIG['dataset_name']], 'r'))
    mode = 'test' if CONFIG['test'] else 'train'
    neccessary_topics = train_test_dict[mode]

    if CONFIG['test']:
        for topic in tqdm(neccessary_topics, desc='Important topics'):
            if CONFIG['dataset_name'] == 'MEANTime':
                topic = meantimeNameConverter[topic]
            corpus = read_CoNLL(True, topic)
            corpus = loadMentionsFromJson(corpus)
            srl_data = get_srl_data(corpus)
            match_allen_srl_structures(corpus, srl_data, True)

            elmo_embedder = ElmoEmbedding(CONFIG['elmo_options_file'], CONFIG['elmo_weight_file'])
            load_elmo_embeddings(corpus, elmo_embedder, set_pred_mentions=True)

            all_entity_clusters, all_event_clusters = test_model(corpus, out_dir)

        print(f"Begin run_conll_scorer after {str(timedelta(seconds=(time.time() - start)))}")

        run_conll_scorer(out_dir)

        end = (time.time() - start)
        print(f"Done - Duration: {str(timedelta(seconds=end))}")
        print("")
    elif not CONFIG['test']:
        print("Preparing train corpus")
        train_corpus = read_CoNLL(split="train")
        train_corpus = loadMentionsFromJson(train_corpus)
        srl_data = get_srl_data(train_corpus)
        match_allen_srl_structures(train_corpus, srl_data, True)
        elmo_embedder = ElmoEmbedding(CONFIG['elmo_options_file'], CONFIG['elmo_weight_file'])
        load_elmo_embeddings(train_corpus, elmo_embedder, set_pred_mentions=True)

        print("Preparing dev corpus")
        dev_corpus = read_CoNLL(split="dev")
        dev_corpus = loadMentionsFromJson(dev_corpus)
        srl_data = get_srl_data(dev_corpus)
        match_allen_srl_structures(dev_corpus, srl_data, True)
        elmo_embedder = ElmoEmbedding(CONFIG['elmo_options_file'], CONFIG['elmo_weight_file'])
        load_elmo_embeddings(dev_corpus, elmo_embedder, set_pred_mentions=True)

        train_out_dir = f"resources/eecdcr_models/{CONFIG['dataset_name']}/"
        if not os.path.exists(train_out_dir):
            os.makedirs(train_out_dir)

        train_model(train_corpus, dev_corpus, train_out_dir, EECDCR_TRAIN_CONFIG_DICT)
