import json
import os.path
from datetime import datetime

import torch

from EECDCR.all_models.model_utils import test_models
from shared.CONSTANTS import EECDCR_CONFIG_DICT, CONFIG


def load_entity_wd_clusters(config_dict):
    '''
    Loads from a file the within-document (WD) entity coreference clusters predicted by an external WD entity coreference
    model/tool and ordered those clusters in a dictionary according to their documents.
    :param config_dict: a configuration dictionary that contains a path to a file stores the
    within-document (WD) entity coreference clusters predicted by an external WD entity coreference
    system.
    :return: a dictionary contains a mapping of a documents to their predicted entity clusters
    '''
    doc_to_entity_mentions = {}

    with open(config_dict["wd_entity_coref_file"], 'r') as js_file:
        js_mentions = json.load(js_file)

    # load all entity mentions in the json
    for js_mention in js_mentions:
        doc_id = js_mention["doc_id"].replace('.xml', '')
        if doc_id not in doc_to_entity_mentions:
            doc_to_entity_mentions[doc_id] = {}
        sent_id = js_mention["sent_id"]
        if sent_id not in doc_to_entity_mentions[doc_id]:
            doc_to_entity_mentions[doc_id][sent_id] = []
        tokens_numbers = js_mention["tokens_numbers"]
        mention_str = js_mention["tokens_str"]

        try:
            coref_chain = js_mention["coref_chain"]
        except:
            continue

        doc_to_entity_mentions[doc_id][sent_id].append((doc_id, sent_id, tokens_numbers,
                                                        mention_str, coref_chain))
    return doc_to_entity_mentions


def _load_check_point(fname):
    '''
    Loads Pytorch model from a file
    :param fname: model's filename
    :return:Pytorch model
    '''
    return torch.load(fname, map_location=torch.device('cpu'))


def test_model(corpus):
    output_dir = f"data/output/{datetime.today().strftime('%Y-%m-%d')}/{CONFIG['dataset_name']}"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    device = torch.device("cpu")

    cd_event_model = _load_check_point(EECDCR_CONFIG_DICT["cd_event_model_path"])
    cd_entity_model = _load_check_point(EECDCR_CONFIG_DICT["cd_entity_model_path"])

    cd_event_model.to(device)
    cd_entity_model.to(device)

    doc_to_entity_mentions = load_entity_wd_clusters(EECDCR_CONFIG_DICT)
    _, _ = test_models(corpus, cd_event_model, cd_entity_model, device,
                       EECDCR_CONFIG_DICT, write_clusters=True, out_dir=output_dir,
                       doc_to_entity_mentions=doc_to_entity_mentions, analyze_scores=False)
