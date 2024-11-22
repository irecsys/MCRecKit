# @Time   : 2022/01/19
# @Author : David Wang, Yong Zheng


import importlib
from logging import getLogger

import numpy as np
import torch
from recbole.utils import InputType, set_color
from torch import nn
from torch.nn.init import normal_

from mcreckit.utils.enum_type import CustomColumn, MCModelType


class AbstractRecommender(nn.Module):
    """Base class for all models
    """

    def __init__(self):
        self.logger = getLogger()
        self.history = {}
        super(AbstractRecommender, self).__init__()

    def calculate_loss(self, interaction):
        """Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        """Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        """full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, 'other_parameter_name'):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def setHistory(self, history):
        self.history = history

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + set_color('\nTrainable parameters', 'blue') + f': {params}'


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    type = MCModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.device = config['device']


# added by David Wang
class MultiCriteriaRecommender(GeneralRecommender):
    """This is an abstract class of multi criteria recommender

    """
    type = MCModelType.MULTICRITERIA
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        """
        Args:
            config: a Config object
            dataset: data loader object
        """
        super(MultiCriteriaRecommender, self).__init__(config, dataset)

        self.history = {}

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.eval_mode = config['eval_args']['mode']

        # get multi criteria specific parameters
        self.LABEL = config['LABEL_FIELD']
        self.criteria_label = config['MULTI_LABEL_FIELD']
        self.criteria_vector_name = CustomColumn.CRITERIA_VECTOR.name
        self.predicted_label_name = CustomColumn.PREDICTED_LABEL.name
        self.num_criteria = len(self.criteria_label)
        self.max_rating_value = config['RATING_RANGE'][1]
        self.min_rating_value = config['RATING_RANGE'][0]

        # load parameters info
        self.device = config['device']

        # get sorting algorithm
        if config['sorting_algorithm']:
            self.sorting_algorithm = getattr(importlib.import_module('mcreckit.model.pareto_sort'),
                                             config['sorting_algorithm'])(config) if config[
                'sorting_algorithm'] else None
            self.sorting_weight = min(1, config['sorting_weight'] if config['sorting_weight'] else 0)
            self.sorting_algorithm.setHistoryItems(self.history)
        else:
            self.sorting_algorithm = None
            self.sorting_weight = 0

    def setHistory(self, history):
        self.history = history

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.05)

    def clamp(self, ratings):
        """clamp the value in ratings tensor into range of self.min_rating_value and self.max_rating_value:
            if rating < self.min_rating_value, then rating = self.min_rating_value
            if rating > self.max_rating_value, then rating = self.max_rating_value
        Args:
            a tensor
        Returns:

        """
        return torch.clamp(ratings, min=self.min_rating_value, max=self.max_rating_value)

    def min_max_scale(self, score):
        """scale value in score tensor to value in [lower_bound, upper_bound] with min max method:
            v' = (v - min(score)) / (max(score) - min(score)) * (upper_bound - lower_bound) + lower_bound
        Args:
            score: a tensor object
        Returns:
            a rescaled tensor
        """

        v_min = score.min()
        v_max = score.max()

        if v_max == v_min:
            scaled_score = self.clamp(score)
        else:
            scaled_score = (score - v_min) / (v_max - v_min) * (self.max_rating_value - self.min_rating_value) \
                           + self.min_rating_value

        return scaled_score
