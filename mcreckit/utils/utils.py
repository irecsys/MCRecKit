# @Time : 2024/10/24
# @Author : Yong Zheng


import importlib

import yaml
from recbole.utils.utils import get_model as recbole_get_model
from sklearn.feature_selection import mutual_info_classif
from mcreckit.utils import MCModelType


def get_model(model_name):
    r"""Return appropriate recommendation model

    Args:
        model_name (str): model name

    Returns:
        recommender
    """
    model_file_name = model_name.lower()
    module_paths = [
        '.'.join(['mcreckit.model', 'multi_criteria_recommender', model_file_name])
    ]
    # module_paths = [
    #     '.'.join(['mcreckit.model', 'multi_criteria_recommender', model_file_name]),
    #     '.'.join(['mcreckit.model', 'general_recommender', model_file_name])
    # ]

    for module_path in module_paths:
        try:
            if importlib.util.find_spec(module_path, __name__):
                model_module = importlib.import_module(module_path, __name__)

                if hasattr(model_module, model_name):
                    model_class = getattr(model_module, model_name)
                    return model_class
                else:
                    raise ValueError(f'`model_name` [{model_name}] is not found in {module_path}')
        except ImportError as e:
            print(f"ImportError in {module_path}: {e}. Trying next module path.")

    print(f"Falling back to get_model from RecBole.")

    return recbole_get_model(model_name)


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    try:
        if model_name in ['MONeuMF', 'JointNeuMF']:
            model_name = 'JointRP'
        return getattr(importlib.import_module('mcreckit.trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type == MCModelType.KNOWLEDGE:
            return getattr(importlib.import_module('recbole.trainer'), 'KGTrainer')
        elif model_type == MCModelType.TRADITIONAL:
            return getattr(importlib.import_module('recbole.trainer'), 'TraditionalTrainer')
        elif model_type == MCModelType.MULTICRITERIA:
            # David Wang: add new MultiCriteriaTrainer
            return getattr(importlib.import_module('mcreckit.trainer'), 'MultiCriteriaTrainer')
        else:
            return getattr(importlib.import_module('mcreckit.trainer'), 'Trainer')


def read_yaml(yaml_file):
    """read yaml file into dictionary object,
        The number of weights in 'ranking_weight' should not less than the number of models in 'sub_models' list
    Args:
        yaml_file: hybrid model yaml file name
    Returns:
        root_config: dictionary object of configuration of dataset and evaluation methods
        ranking_weight: a list of weights
        sub_models:  list of model definition, training, evaluation parameters
    """
    # read yaml file into dictionary
    with open(yaml_file, "r") as stream:
        try:
            root_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # check basic settings in yaml file
    if 'ranking_weight' in root_config and 'sub_models' in root_config:
        ranking_weight = root_config.pop('ranking_weight')
        sub_models = root_config.pop('sub_models')
    else:
        raise ValueError(f"Either 'ranking_weight' or 'sub_models' is not in yaml file: {yaml_file}")

    return root_config, ranking_weight, sub_models


def set_information_gain_order(dataset, config):
    """Calculate information gain for each label in the label_list and order the list accordingly
    Args:
        dataset: data set object with data in label_list
        config: list of labels
    Returns:
        list of labels in order of information gain
    """

    label_mutual_score = {}
    if config['LABEL_FIELD']:
        rating_label = config['LABEL_FIELD']
    else:
        rating_label = config['RATING_FIELD']

    feature = dataset[config['MULTI_LABEL_FIELD']]
    target = dataset[rating_label]

    # calculate mutual information
    feature_score = mutual_info_classif(feature, target)

    # assign to dictionary
    for label, score in zip(config['MULTI_LABEL_FIELD'], feature_score):
        label_mutual_score[label] = score

    label_mutual_score = dict(sorted(label_mutual_score.items(), key=lambda item: item[1], reverse=True))

    return list(label_mutual_score.keys())
