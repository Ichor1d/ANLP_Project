dataset_path = {
    'ECB+': 'data/ecb+/ecbplus.conll',
    'MEANTime': 'data/meantime/meantime.conll'
}

train_test_path = {
    'ECB+': 'data/ecb+/train_test_split.json',
    'MEANTime': 'data/meantime/train_test_split.json'
}

mentions_path = {
    'ECB+': 'data/ecb+/mentions/',
    'MEANTime': 'data/meantime/mentions/'
}

meantimeNameConverter = {
    '1MEANTIMEcross': 'corpus_airbus',
    'corpus_airbus': '1MEANTIMEcross',
    '2MEANTIMEcross': 'corpus_apple',
    'corpus_apple': '2MEANTIMEcross',
    '3MEANTIMEcross': 'corpus_gm',
    'corpus_gm': '3MEANTIMEcross',
    '4MEANTIMEcross': 'corpus_stock',
    'corpus_stock': '4MEANTIMEcross'
}

CONFIG = {
    'test': True,
    'dataset_name': 'MEANTime',
    'use_singletons': False,
    'bert_file': 'resources/word_vector_models/BERT_SRL/bert-base-srl-2019.06.17.tar.gz',
    'elmo_options_file': 'resources/word_vector_models/ELMO_Original_55B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
    'elmo_weight_file': 'resources/word_vector_models/ELMO_Original_55B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',
    'use_dep': False
}

EECDCR_CONFIG_DICT = {
    "test_path": "data/processed/full_swirl_ecb/test_data",

    "cd_event_model_path": "resources/eecdcr_models/from_barhom/cd_event_best_model",
    "cd_entity_model_path": "resources/eecdcr_models/from_barhom/cd_entity_best_model",

    "gpu_num": 0,
    "event_merge_threshold": 0.5,
    "entity_merge_threshold": 0.5,
    "use_elmo": True,
    "use_args_feats": True,
    "use_binary_feats": True,

    "test_use_gold_mentions": True,
    "wd_entity_coref_file": "data/external/wd_coref.json",
    # "wd_entity_coref_file": "data/ecb+/ecb_wd_coref.json",
    "merge_iters": 2,

    "load_predicted_topics": False,
    "predicted_topics_path": "data/external/document_clustering/predicted_topics",

    "seed": 1,
    "random_seed": 2048,

    "event_gold_file_path": f"data/gold/{CONFIG['dataset_name']}/own_event_mention_based.key_conll",
    "entity_gold_file_path": f"data/gold/{CONFIG['dataset_name']}/own_entity_mention_based.key_conll"
}


EECDCR_TRAIN_CONFIG_DICT = {
    "batch_size": 16,
    "char_pretrained_path": "resources/Glove_Embedding_Files/char_embed/glove.840B.300d-char.npy",
    "char_rep_size": 50,
    "char_vocab_path": "resources/Glove_Embedding_Files/char_embed/glove.840B.300d-char.vocab",
    "dev_path": "data/processed/cybulska_setup/full_swirl_ecb/dev_data",
    "dev_th_range": [0.5, 0.6],

    "entity_merge_threshold": 0.5,
    "epochs": 50,
    "event_merge_threshold": 0.5,
    "feature_size": 50,
    "glove_path": "resources/Glove_Embedding_Files/glove.6B.300d.txt",
    "gpu_num": 0,

    "load_predicted_topics": False,
    "log_interval": 32,
    "loss": "bce",
    "lr": 0.0001,
    "merge_iters": 2,
    "momentum": 0,

    "optimizer": "adam",
    "patient": 7,
    "random_seed": 2048,
    "regressor_epochs": 1,
    "remove_singletons": False,
    "resume_training": False,
    "seed": 1,

    "test_use_gold_mentions": True,
    "train_init_wd_entity_with_gold": True,
    "train_path": "data/processed/cybulska_setup/full_swirl_ecb/training_data",
    "use_args_feats": True,
    "use_binary_feats": True,
    "use_diff": False,
    "use_mult": True,
    "use_pretrained_char": True,

    "wd_entity_coref_file": "data/external/wd_coref.json",
    "weight_decay": 0
}
