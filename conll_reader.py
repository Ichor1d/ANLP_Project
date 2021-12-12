from shared.CONSTANTS import datasetDict
from shared.classes import Token, Sentence, Document, Topic, Corpus


def readConll(datasetName: str, topicLevel=False, specificDocument="") -> Corpus:
    with open(datasetDict[datasetName], "r", encoding="utf8") as f:
        data = f.read()

    data = data.split("\n")

    if topicLevel:
        data = [x for x in data if x.split("\t")[0].startswith(specificDocument + "/")]

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
                topic.add_doc(document_name, document)
                corpus.add_topic(topic_name, topic)
                break
        split_line = bs.split("\t")
        topic_and_document = split_line[0]
        topic_name = topic_and_document.split("/")[0]
        document_name = topic_and_document.split("/")[1]

        if topic_name != prev_topic_id:
            if prev_topic_id != "":
                corpus.add_topic(prev_topic_id, topic)
            topic = Topic(topic_name)
            prev_topic_id = topic_name

        # if we start a new document (name of the document changes)
        if document_name != prev_document_name:
            if prev_document_name != "":
                topic.add_doc(prev_document_name, document)
            document = Document(document_name)
            prev_document_name = document_name

        sentence_id = split_line[1]
        # if we start a new sentence add the old sentence to the document
        if sentence_id != prev_sentence_id:
            document.add_sentence(sentence_id, sentence)
            sentence = Sentence(sentence_id)
            prev_sentence_id = sentence_id

        token = Token(split_line[2], split_line[3])
        sentence.add_token(token)

    return corpus
