# @Time   : 2024/11/15
# @Author : Yong Zheng

general_arguments = [
    'gpu_id', 'use_gpu',
    'seed',
    'reproducibility',
    'state',
    'data_path',
    'checkpoint_dir',
    'show_progress',
    'config_file',
    'save_dataset',
    'dataset_save_path',
    'save_dataloaders',
    'dataloaders_save_path',
    'print_loss','device','time_id',
]

training_arguments = [
    'MODEL_TYPE', 'criteria_model', 'overall_model',
    'epochs', 'train_batch_size',
    'learner', 'learning_rate',
    'neg_sampling',
    'eval_step', 'stopping_step',
    'clip_grad_norm',
    'weight_decay',
    'loss_decimal_place','MODEL_INPUT_TYPE',
    'train_neg_sample_args',
]

evaluation_arguments = [
    'eval_args', 'repeatable', 'eval_type',
    'metrics', 'topk', 'valid_metric', 'valid_metric_bigger',
    'eval_batch_size',
    'metric_decimal_place','eval_neg_sample_args',
]

mcranking_arguments = [
    'sorting_algorithm', 'sub_sort', 'error_margin', 'dominance_k', 'epsilon'
]

dataset_arguments = [
    'field_separator', 'seq_separator',
    'USER_ID_FIELD', 'ITEM_ID_FIELD', 'RATING_FIELD', 'TIME_FIELD', 'MULTI_LABEL_FIELD',
    'RATING_RANGE',
    'LABEL_FIELD', 'threshold',
    'NEG_PREFIX','seq_len',
    'ITEM_LIST_LENGTH_FIELD', 'LIST_SUFFIX', 'MAX_ITEM_LIST_LENGTH', 'POSITION_FIELD',
    'HEAD_ENTITY_ID_FIELD', 'TAIL_ENTITY_ID_FIELD', 'RELATION_ID_FIELD', 'ENTITY_ID_FIELD',
    'load_col', 'unload_col', 'unused_col', 'additional_feat_suffix',
    'rm_dup_inter', 'val_interval', 'filter_inter_by_user_or_item',
    'user_inter_num_interval', 'item_inter_num_interval',
    'alias_of_user_id', 'alias_of_item_id', 'alias_of_entity_id', 'alias_of_relation_id',
    'preload_weight', 'normalize_field', 'normalize_all',
    'benchmark_filename',
]
