import time

from shared.classes import Corpus, EntityMention, EventMention
from shared.CONSTANTS import mentions_path, CONFIG
import json
import spacy
from tqdm import tqdm


def _attachMentionToCorpus(mention, corpus: Corpus, is_event):
    doc_id = mention.doc_id
    sent_id = f"{mention.sent_id}"

    if CONFIG['dataset_name'] == "MEANTime":
        doc_id = doc_id.split("_")[0]

    for topic in corpus.topics:
        try:
            sentence = corpus.topics[topic].docs[doc_id].sentences[sent_id]

            for token_number in mention.tokens_numbers:
                mention.add_token(sentence.tokens[token_number])

            sentence.add_gold_mention(mention, is_event)
            break
        except:
            continue

    return corpus


def _createMention(corpus, entry, nlp, eventOrEntity):
    sentence_found = False
    doc_id = entry['doc_id']
    sent_id = f"{entry['sent_id']}"
    tokens_number = entry['tokens_number']
    tokens = []
    mention_str = entry['tokens_str']
    is_continuous = entry['is_continuous']
    is_singleton = entry['is_singleton']
    mention_type = entry['mention_type']
    coref_chain = entry['coref_chain']

    if CONFIG['dataset_name'] == "MEANTime":
        doc_id = doc_id.split("_")[0]

    for topic in corpus.topics:
        try:
            sentence = corpus.topics[topic].docs[doc_id].sentences[sent_id]
            sentence_found = True
            for token_number in tokens_number:
                tokens.append(sentence.tokens[token_number])
            break
        except:
            continue

    if sentence_found:
        head_text = ""
        head_lemma = ""
        token_strings = " ".join([token.get_token() for token in tokens])
        doc = nlp(token_strings)
        for token in doc:
            if token.dep_ == 'ROOT':
                head_text = token.text
                head_lemma = token.lemma_
                break

        if eventOrEntity == 'entity':
            mention = EntityMention(doc_id, sent_id, tokens_number, tokens, mention_str,
                          head_text, head_lemma, is_singleton, is_continuous, coref_chain, mention_type)
            sentence.add_gold_mention(mention, False)
        elif eventOrEntity == 'event':
            mention = EventMention(doc_id, sent_id, tokens_number, tokens, mention_str,
                                   head_text, head_lemma, is_singleton, is_continuous, coref_chain)
            sentence.add_gold_mention(mention, True)


def loadMentionsFromJson(corpus: Corpus = None):
    dataset_name = CONFIG['dataset_name']
    nlp = spacy.load("en_core_web_sm")
    entity_path = mentions_path[dataset_name] + "/entity_mentions_.json"
    event_path = mentions_path[dataset_name] + "/event_mentions_.json"

    with open(entity_path, "r", encoding="utf8") as f:
        entity_mentions = json.load(f)

    with open(event_path, "r", encoding="utf8") as f:
        event_mentions = json.load(f)

    """
        If you don't want to use singletons every mention, which is a singleton ('is_singleton': True) is removed
    """
    if not CONFIG['use_singletons']:
        entity_mentions = [x for x in entity_mentions if not x['is_singleton']]
        event_mentions = [x for x in event_mentions if not x['is_singleton']]

    print("Handling all entity mentions")
    for entity_mention in tqdm(entity_mentions, desc="Entity Mentions"):
        _createMention(corpus, entity_mention, nlp, 'entity')
        # corpus = _attachMentionToCorpus(mention, corpus, False)

    print("Handling all event mentions")
    for event_mention in tqdm(event_mentions, desc="Event Mentions"):
        _createMention(corpus, event_mention, nlp, 'event')
        # corpus = _attachMentionToCorpus(mention, corpus, True)

    return corpus
