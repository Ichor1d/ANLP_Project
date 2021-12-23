from conll_reader import readConll
from features.build_features import match_allen_srl_structures
from mentionsfromjson import loadMentionsFromJson
from shared.CONSTANTS import CONFIG
from srl_things import get_srl_data


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
    inp = _chooseDataset()

    while inp not in [0, 1, 2]:
        print("Not a valid input.")
        inp = _chooseDataset()

    CONFIG['datasetName'] = 'ECB+' if inp == 0 else 'MEANTime'
    corpus = readConll()

    topicLevel = False
    print("Do you want a specific topic?\n0: yes \n1: no (default)")
    topicLevel = True if input(">>> ") == "0" else False

    if topicLevel:
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
        corpus = readConll(topicLevel, topic_name)

    inp = int(input("Do you want to use singletons?\n0: Yes\n1: No\n>>> "))
    CONFIG['use_singletons'] = inp == 0

    corpus = loadMentionsFromJson(corpus)
    srl_data = get_srl_data(corpus)
    match_allen_srl_structures(corpus, srl_data, True)

    print("Fertig.")
