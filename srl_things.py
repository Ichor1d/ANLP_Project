import os
from collections import defaultdict

from allennlp.predictors import Predictor

from typing import Dict, List

from features.allen_srl_reader import SRLSentence, SRLVerb, SRLArg
from shared.classes import Corpus
from shared.CONSTANTS import CONFIG
from tqdm import tqdm


def get_srl_data(corpus: Corpus) -> Dict[str, Dict[str, SRLSentence]]:
    """
    Extracts labels from semantic role labeling (SRL).

    Args:
        corpus: A EECDCE document collection object.

    Returns:
        A dictionary with EECDCR SRL sentence structures.

    """

    if not os.path.exists(CONFIG['bert_file']):
        raise Exception("Bert Model was not found.")

    predictor = Predictor.from_path(CONFIG['bert_file'])

    srl_data = defaultdict(lambda: defaultdict(SRLSentence))
    for topic in list(corpus.topics.values()):
        for doc_id, doc in tqdm(topic.docs.items(), desc=topic.topic_id, leave=False):

            for sent_id, sent in doc.sentences.items():
                srl_sent = SRLSentence(doc_id, sent_id)
                srl = predictor.predict_tokenized([t.token for t in sent.tokens])

                for verb in srl["verbs"]:
                    srl_verb_obj = SRLVerb()
                    srl_verb_obj.verb = SRLArg(verb["verb"], [srl["words"].index(verb["verb"])])

                    for tag_id, tag in enumerate(verb["tags"]):
                        for tag_type in ["ARG0", "ARG1", "TMP", "LOC", "NEG"]:
                            check_tag(tag, tag_id, srl_verb_obj, tag_type, srl["words"])
                    srl_sent.add_srl_vrb(srl_verb_obj)

                srl_data[doc_id][sent_id] = srl_sent
    return srl_data


def check_tag(tag: str, tag_id: int, srl_verb_obj: SRLVerb, attr: str, words: List[str]):
    """
    Checks tags from SRL and initialize SRL objects from EECDCR.

    Args:
        tag: A SRL tag.
        tag_id: A SRL tag id.
        srl_verb_obj: A SRL verb object from EECDCR.
        attr: An attribute for which we need to check in tags.
        words: A list of words from SRL tagger.

    """
    tag_attr_dict = {"ARG0": "arg0",
                     "ARG1": "arg1",
                     "TMP": "arg_tmp",
                     "LOC": "arg_loc",
                     "NEG": "arg_neg"}
    if attr in tag:
        if tag[0] == "B":
            setattr(srl_verb_obj, tag_attr_dict[attr], SRLArg(words[tag_id], [tag_id]))
        else:
            srl_arg = getattr(srl_verb_obj, tag_attr_dict[attr])
            if srl_arg is None:
                srl_arg = SRLArg("", [])
            srl_arg.text += " " + words[tag_id]
            srl_arg.ecb_tok_ids.append(tag_id)
            setattr(srl_verb_obj, tag_attr_dict[attr], srl_arg)
