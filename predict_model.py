import datetime

from conll_reader import read_CoNLL
from features.build_features import match_allen_srl_structures, load_elmo_embeddings
from features.create_elmo_embeddings import ElmoEmbedding
from mentionsfromjson import loadMentionsFromJson
from run_eecdcr import test_model
from shared.CONSTANTS import CONFIG
from srl_things import get_srl_data
import time


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

    print("Do you wanna train or test a model?\n0: train\n1: test")
    CONFIG['test'] = int(input(">>> ")) == 1

    # Delete after some time.
    print("For evaluation you can only test models.")
    CONFIG['test'] = True

    start = time.time()
    inp = _chooseDataset()

    while inp not in [0, 1, 2]:
        print("Not a valid input.")
        inp = _chooseDataset()

    CONFIG['dataset_name'] = 'ECB+' if inp == 0 else 'MEANTime'
    corpus = read_CoNLL()

    inp = 0
    print("Do you want to use only a specific topic?\n0: No (default)\n1: Yes")
    try:
        inp = int(input(">>> "))
    except ValueError:
        print("Invalid input, proceed with default.")
    topic_level = inp == 1

    if topic_level:
        inp = -1
        while inp < 0:
            print("Choose the topic:")
            for i, bs in enumerate(corpus.topics):
                print(f"{i}: {bs}")
            try:
                inp = int(input(">>> "))
            except ValueError:
                print(f"Only numbers between 0 and {len(corpus.topics)} are allowed.\n")
        topic_name = list(corpus.topics)[inp]
        corpus = read_CoNLL(True, topic_name)

    inp = int(input("Do you want to use singletons?\n0: Yes\n1: No\n>>> "))
    CONFIG['use_singletons'] = inp == 0

    print(f"Loading mentions & getting SRL after {time.time() - start}")
    corpus = loadMentionsFromJson(corpus)
    srl_data = get_srl_data(corpus)
    match_allen_srl_structures(corpus, srl_data, True)

    elmo_embedder = ElmoEmbedding(CONFIG['elmo_options_file'], CONFIG['elmo_weight_file'])
    load_elmo_embeddings(corpus, elmo_embedder, set_pred_mentions=True)

    print(f"Begin test after {time.time() - start}")
    test_model(corpus)

    end = (time.time() - start)
    print(f"Done - Duration: {str(datetime.timedelta(seconds=end))}")
    print("")
