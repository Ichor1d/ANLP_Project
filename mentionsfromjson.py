import time

from conll_reader import readConll
from shared.classes import Corpus, EntityMention, EventMention
from shared.CONSTANTS import mentionsPath
import json
import spacy
from typing import *


def _attachMentionToCorpus(mention, corpus: Corpus, event):
    doc_id = mention.doc_id
    sent_id = f"{mention.sent_id}"

    print(sent_id)

    if 'ecbplus' in doc_id:
        doc_id = doc_id.replace("ecbplus", "").split("_")[0] + "ecbplus"
    elif 'ecb' in doc_id:
        doc_id = doc_id.replace("ecb", "").split("_")[0] + "ecb"
    else:
        doc_id = doc_id.split("_")[0]

    for topic in corpus.topics:
        try:
            sentence = corpus.topics[topic].docs[doc_id].sentences[sent_id]
            sentence.add_gold_mention(mention, event)
            break
        except BaseException:
            continue

    return corpus


def _createMention(entry, nlp, eventOrEntity):
    doc_id = entry['doc_id']
    sent_id = entry['sent_id']
    tokens_number = entry['tokens_number']
    tokens = []
    mention_str = entry['tokens_str']
    is_continuous = entry['is_continuous']
    is_singleton = entry['is_singleton']
    mention_type = entry['mention_type']
    coref_chain = entry['coref_chain']

    head_text = ""
    head_lemma = ""
    doc = nlp(mention_str)
    for token in doc:
        if token.dep_ == 'ROOT':
            head_text = token.text
            head_lemma = token.lemma_
            break

    if eventOrEntity == 'entity':
        mention = EntityMention(doc_id, sent_id, tokens_number, tokens, mention_str,
                      head_text, head_lemma, is_singleton, is_continuous, coref_chain, mention_type)
        return mention
    elif eventOrEntity == 'event':
        mention = EventMention(doc_id, sent_id, tokens_number, tokens, mention_str,
                               head_text, head_lemma, is_singleton, is_continuous, coref_chain)
        return mention


def loadMentionsFromJson(datasetName: str, corpus: Corpus = None):
    start_time = time.time()
    all_mentions = []
    nlp = spacy.load("en_core_web_sm")
    entity_path = mentionsPath[datasetName] + "/entity_mentions_.json"
    event_path = mentionsPath[datasetName] + "/event_mentions_.json"
    with open(entity_path, "r", encoding="utf8") as f:
        entity_mentions = json.load(f)

    with open(event_path, "r", encoding="utf8") as f:
        event_mentions = json.load(f)

    for entity_mention in entity_mentions:
        mention = _createMention(entity_mention, nlp, 'entity')
        corpus = _attachMentionToCorpus(mention, corpus, False)
        all_mentions.append(mention)

    for event_mention in event_mentions:
        mention = _createMention(event_mention, nlp, 'event')
        corpus = _attachMentionToCorpus(mention, corpus, True)
        all_mentions.append(mention)

    end_time = time.time()
    print(f"Dauer: {end_time - start_time}")
    print("")


if __name__ == '__main__':
    datasetName = 'MEANTime'
    corpus = readConll(datasetName)
    loadMentionsFromJson(datasetName, corpus)
