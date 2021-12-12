from conll_reader import readConll


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

    datasetName = 'ECB+' if inp == 0 else 'MEANTime'
    corpus = readConll(datasetName)

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
        corpus = readConll(datasetName, topicLevel, topic_name)

    print("Fertig.")
