# @Time : 2024/10/24
# @Author : David Wang, Yong Zheng, Qin Ruan

import copy
from logging import getLogger

from recbole.data import KnowledgeBasedDataLoader, FullSortEvalDataLoader, NegSampleEvalDataLoader, TrainDataLoader
from recbole.sampler import RepeatableSampler, KGSampler
from recbole.utils import EvaluatorType, set_color

from mcreckit.data.dataloader.general_dataloader import LabeledRankingEvalDataLoader
from mcreckit.sampler import MCSampler
from mcreckit.utils import MCModelType
from recbole.data.utils import create_dataset as recbole_create_dataset, _get_AE_dataloader, save_split_dataloaders


def create_dataset(config):
    """Create dataset from config
    """
    model_type = config['MODEL_TYPE']

    if model_type == MCModelType.MULTICRITERIA:
        from .dataset import MultiCriteriaDataset
        return MultiCriteriaDataset(config)

    return recbole_create_dataset(config)


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    phases = ['train', 'valid', 'test']
    train_neg_sample_args = config['train_neg_sample_args']  # David Wang: for rating data: {'strategy': 'none'}

    # DW: determined by config['eval_args'], if config['eval_args']['mode'] != labeled, it would be set up
    # DW: config['eval_args']['mode'] == labeled: {'strategy': 'none', 'distribution': 'none'}
    eval_neg_sample_args = config['eval_neg_sample_args']

    # David Wang: if 'strategy' == 'none', three samplers are None
    sampler = None
    train_sampler, valid_sampler, test_sampler = None, None, None

    # David Wang: create training sampler
    if train_neg_sample_args['strategy'] != 'none':
        if not config['repeatable']:
            sampler = MCSampler(phases, built_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_sampler = sampler.set_phase('train')

    # David Wang: create evaluation sampler
    if eval_neg_sample_args['strategy'] != 'none':
        if sampler is None:
            if not config['repeatable']:
                sampler = MCSampler(phases, built_datasets, eval_neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        valid_sampler = sampler.set_phase(
            'valid')  # DW: this sampler has used item for each user in validation data and training data
        test_sampler = sampler.set_phase(
            'test')  # DW: this sampler has used item for each user in whole dataset (training, valid, testing)

    # for ranking evaluation
    elif config['eval_type'] == EvaluatorType.RANKING:
        sampler = MCSampler(phases, built_datasets, eval_neg_sample_args['distribution'],
                            neg_sampling=config['neg_sampling'])
        valid_sampler = sampler.set_phase('valid')
        test_sampler = sampler.set_phase('test')

    return train_sampler, valid_sampler, test_sampler


def data_preparation(config, dataset, split_data, use_criteria_model=False, save=False):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        split_data: list of split dataset: training, validation, testing
        use_criteria_model: used for created correct data loader for criteria model training
        save (bool, optional): If ``True``, it will call :func:`save_datasets` to save split dataset.
            Defaults to ``False``.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    model_type = config['MODEL_TYPE']
    config = copy.deepcopy(config)

    built_datasets = copy.deepcopy(split_data)

    logger = getLogger()

    train_dataset, valid_dataset, test_dataset = built_datasets

    # David Wang: if config['train_neg_sample_args'] and config['eval_neg_sample_args'] specify 'strategy' to 'none',
    # there three samplers are empty
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

    if model_type != MCModelType.KNOWLEDGE:
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
    else:
        kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, kg_sampler, shuffle=True)

    valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)

    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    if save:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    return train_data, valid_data, test_data


def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    register_table = {
        "MultiDAE": _get_AE_dataloader,
        "MultiVAE": _get_AE_dataloader,
        'MacridVAE': _get_AE_dataloader,
        'CDAE': _get_AE_dataloader,
        'ENMF': _get_AE_dataloader,
        'RaCT': _get_AE_dataloader,
        'RecVAE': _get_AE_dataloader,
    }

    # David Wang: get data loader based on model name
    if config['model'] in register_table:
        return register_table[config['model']](config, phase)

    model_type = config['MODEL_TYPE']
    if phase == 'train':
        if model_type != MCModelType.KNOWLEDGE:
            return TrainDataLoader
        else:
            return KnowledgeBasedDataLoader
    else:  # David Wang: for evaluation and testing phases based on config['eval_neg_sample_args']['strategy']
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if config['eval_args']['mode'] == 'labeled' and config['eval_type'] == EvaluatorType.RANKING:
            return LabeledRankingEvalDataLoader
        elif eval_strategy in {'none', 'by'} and config['eval_type'] == EvaluatorType.VALUE:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader
