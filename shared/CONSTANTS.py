dataset_path = {
    'ECB+': 'data/ecb+/ecbplus.conll',
    'MEANTime': 'data/meantime/meantime.conll'
}

mentions_path = {
    'ECB+': 'data/ecb+/mentions/',
    'MEANTime': 'data/meantime/mentions/'
}

meantimeNameConverter = {
    '1MEANTIMEcross': 'corpus_airbus',
    '2MEANTIMEcross': 'corpus_apple',
    '3MEANTIMEcross': 'corpus_gm',
    '4MEANTIMEcross': 'corpus_stock'
}

CONFIG = {
    'test': True,
    'dataset_name': 'MEANTime',
    'use_singletons': False,
    'bert_file': 'resources/word_vector_models/BERT_SRL/bert-base-srl-2019.06.17.tar.gz',
    'use_dep': True
}

EECDCR_CONFIG_DICT = {
    "test_path": "data/processed/full_swirl_ecb/test_data",

    "cd_event_model_path": "resources/eecdcr_models/cd_event_best_model",
    "cd_entity_model_path": "resources/eecdcr_models/cd_entity_best_model",

    "gpu_num": 0,
    "event_merge_threshold": 0.5,
    "entity_merge_threshold": 0.5,
    "use_elmo": True,
    "use_args_feats": True,
    "use_binary_feats": True,

    "test_use_gold_mentions": True,
    "wd_entity_coref_file": "data/external/stanford_neural_wd_entity_coref_out/ecb_wd_coref.json",
    "merge_iters": 2,

    "load_predicted_topics": True,
    "predicted_topics_path": "data/external/document_clustering/predicted_topics",

    "seed": 1,
    "random_seed": 2048,

    "event_gold_file_path": "data/gold/cybulska_gold/CD_test_event_mention_based.key_conll",
    "entity_gold_file_path": "data/gold/cybulska_gold/CD_test_entity_mention_based.key_conll"
}
