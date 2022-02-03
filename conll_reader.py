import json

from shared.CONSTANTS import dataset_path, meantimeNameConverter, CONFIG, train_test_path
from shared.classes import Token, Sentence, Document, Topic, Corpus
from typing import Union

"""
    To no real surprise the topic_id/document_id is not consistent between meantime & ecb+
    or between the conll format and the .json files to be more precise.
    So there was a need to change the topic / documents id in both cases differently to match
    the format of the .json file with all the mentions and I had to differentiate between the
    two datasets. If more data sets will be added in the future, this might look even worse.
"""


def read_CoNLL(topic_level=False, specific_document="", split: Union[None, str] = None) -> Corpus:
    number_of_seen_topics = 1
    dataset_name = CONFIG['dataset_name']

    # train_test_dict = json.load(open(train_test_path[dataset_name], 'r'))
    # mode = 'test' if CONFIG['test'] else 'train'
    # necessary_topics = train_test_dict[mode]

    with open(dataset_path[dataset_name], "r", encoding="utf8") as f:
        data = f.read()

    data = data.split("\n")

    if topic_level and specific_document != "" and split is None:
        data = [x for x in data if x.split("\t")[0].startswith(specific_document + "/")]

    if split in ['test', 'train', 'dev']:
        train_test_dict = json.load(open(train_test_path[dataset_name], 'r'))
        necessary_topics = train_test_dict[split]
        data = [datum for datum in data if datum.split("\t")[0].split("/")[0] in necessary_topics]

    prev_sentence_id = "0"
    sentence = Sentence(0)

    prev_document_name = ""
    document_name = "ijkasdjibnasdkjbn"
    document = Document("Bla, Bla, Bla, Mr. Freeman")

    prev_topic_id = ""
    topic_name = ""
    topic = Topic("Nanananananana, BATMAN!")

    corpus = Corpus()
    for bs in data:
        # jump over empty entries
        if not bs:
            continue

        # if the first entry is a # we know its the beginning or the end line
        if bs.startswith("#begin"):
            # if its the first entry just ignore it
            continue
        if bs.startswith("#end"):
            # if its the last entry just ignore it
            continue

        split_line = bs.split("\t")
        topic_and_document = split_line[0]
        topic_name = topic_and_document.split("/")[0]
        document_name = topic_and_document.split("/")[1]
        if dataset_name == "ECB+":
            document_name = topic_name.replace("ecb", f"_{document_name}ecb")
        if dataset_name == "MEANTime":
            topic_name = meantimeNameConverter[topic_name]

        sentence_id = split_line[1]
        # if we start a new sentence add the old sentence to the document
        if sentence_id != prev_sentence_id:
            if prev_sentence_id != "":
                document.add_sentence(prev_sentence_id, sentence)
            sentence = Sentence(sentence_id)
            prev_sentence_id = sentence_id

        if split_line[3] != "":
            token = Token(split_line[2], split_line[3])
            sentence.add_token(token)
        else:
            # for some reason some tokens are empty. For analysis reasons, these are printed.
            print(f"Skipped the token {bs}, since it was empty.")

        # if we start a new document (name of the document changes)
        if document_name != prev_document_name:
            if prev_document_name != "":
                topic.add_doc(prev_document_name, document)
            document = Document(document_name)
            prev_document_name = document_name

        # if a new topic starts (name of the topic changes)
        if topic_name != prev_topic_id:
            if prev_topic_id != "":
                # if dataset_name == "MEANTime":
                #     # prev_topic_id = f"{number_of_seen_topics}MEANTIMEcross"
                #     prev_topic_id = meantimeNameConverter[topic_name]
                #     number_of_seen_topics += 1
                corpus.add_topic(prev_topic_id, topic)
            topic = Topic(topic_name)
            prev_topic_id = topic_name

    # after we run through all the data we just save the last topic.
    if prev_topic_id not in corpus.topics:
        # if dataset_name == "MEANTime":
        #     topic_name = meantimeNameConverter[topic_name]  # f"{number_of_seen_topics}MEANTIMEcross"
        topic.add_doc(document_name, document)
        corpus.add_topic(topic_name, topic)

    return corpus
