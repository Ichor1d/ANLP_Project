from shared.CONSTANTS import datasetDict, meantimeNameConverter
from shared.classes import Token, Sentence, Document, Topic, Corpus

"""
    To no real surprise the topic_id/document_id is not consistent between meantime & ecb+
    or between the conll format and the .json files to be more precise.
    So there was a need to change the topic / documents id in both cases differently to match
    the format of the .json file with all the mentions and I had to differentiate between the
    two datasets. If more data sets will be added in the future, this might look even worse.
"""


def readConll(datasetName: str, topicLevel=False, specificDocument="") -> Corpus:
    numberOfSeenTopics = 1
    with open(datasetDict[datasetName], "r", encoding="utf8") as f:
        data = f.read()

    data = data.split("\n")

    if topicLevel:
        if datasetName == "MEANTime":
            specificDocument = meantimeNameConverter[specificDocument]
        data = [x for x in data if x.split("\t")[0].startswith(specificDocument + "/")]
        data.append("# end document")

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

        # if the first entry is a # we now its the beginning or the end line
        if bs[0] == "#":
            if bs.split(" ")[1] == "begin":
                # if its the first entry just ignore it
                continue
            # if its the last entry save everything and quit
            if bs.split(" ")[1] == "end":
                if datasetName == "MEANTime":
                    topic_name = f"{numberOfSeenTopics}MEANTIMEcross"
                    numberOfSeenTopics += 1
                topic.add_doc(document_name, document)
                corpus.add_topic(topic_name, topic)
                break
        split_line = bs.split("\t")
        topic_and_document = split_line[0]
        topic_name = topic_and_document.split("/")[0]
        document_name = topic_and_document.split("/")[1]
        if datasetName == "ECB+":
            if 'ecbplus' in topic_name:
                document_name = topic_name.replace("ecbplus", f"_{document_name}ecbplus")
            elif 'ecb' in topic_name:
                document_name = topic_name.replace("ecb", f"_{document_name}ecb")

        # if we start a new document (name of the document changes)
        if document_name != prev_document_name:
            if prev_document_name != "":
                topic.add_doc(prev_document_name, document)
            document = Document(document_name)
            prev_document_name = document_name

        # if a new topic starts (name of the topic changes)
        if topic_name != prev_topic_id:
            if prev_topic_id != "":
                if datasetName == "MEANTime":
                    prev_topic_id = f"{numberOfSeenTopics}MEANTIMEcross"
                    numberOfSeenTopics += 1
                corpus.add_topic(prev_topic_id, topic)
            topic = Topic(topic_name)
            prev_topic_id = topic_name

        sentence_id = split_line[1]
        # if we start a new sentence add the old sentence to the document
        if sentence_id != prev_sentence_id:
            if prev_sentence_id != "":
                document.add_sentence(sentence_id, sentence)
            sentence = Sentence(sentence_id)
            prev_sentence_id = sentence_id

        token = Token(split_line[2], split_line[3])
        sentence.add_token(token)

    return corpus
