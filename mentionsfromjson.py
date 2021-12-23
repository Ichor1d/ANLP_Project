import time

from shared.classes import Corpus, EntityMention, EventMention
from shared.CONSTANTS import mentions_path, CONFIG
import json
import spacy


def _attachMentionToCorpus(mention, corpus: Corpus, is_event):
    doc_id = mention.doc_id
    sent_id = f"{mention.sent_id}"

    if 'ecbplus' in doc_id:
        doc_id = doc_id.replace("ecbplus", "").split("_")[0] + "ecbplus"
    elif 'ecb' in doc_id:
        doc_id = doc_id.replace("ecb", "").split("_")[0] + "ecb"
    else:
        doc_id = doc_id.split("_")[0]

    for topic in corpus.topics:
        try:
            sentence = corpus.topics[topic].docs[doc_id].sentences[sent_id]
            sentence.add_gold_mention(mention, is_event)
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


def loadMentionsFromJson(corpus: Corpus = None):
    dataset_name = CONFIG['dataset_name']
    start_time = time.time()
    all_mentions = []
    nlp = spacy.load("en_core_web_sm")
    entity_path = mentions_path[dataset_name] + "/entity_mentions_.json"
    event_path = mentions_path[dataset_name] + "/event_mentions_.json"
    with open(entity_path, "r", encoding="utf8") as f:
        entity_mentions = json.load(f)

    with open(event_path, "r", encoding="utf8") as f:
        event_mentions = json.load(f)

    if not CONFIG['use_singletons']:
        entity_mentions = [x for x in entity_mentions if not x['is_singleton']]
        event_mentions = [x for x in event_mentions if not x['is_singleton']]

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

    return corpus
