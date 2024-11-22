# @Time : 2021/12/04
# @Author : David Wang

import copy
import torch

from recbole.data.interaction import Interaction
from mcreckit.model.abstract_recommender import MultiCriteriaRecommender
from mcreckit.utils import get_model
from recbole.utils import init_seed


class CombRP(MultiCriteriaRecommender):
    """
        CombRP = Combo Rating Prediction
        This model combine two stages:
        1). predicting multi-criteria ratings
        2). estimating the overall rating from predicted multi-criteria ratings
    """

    def __init__(self, config, dataset):
        super(CombRP, self).__init__(config, dataset)

        # get criteria model for criteria rating prediction
        self.criteria_model_name = config['criteria_model']['model']

        # get config for criteria rating model
        criteria_config = copy.deepcopy(config)
        criteria_config.final_config_dict.update(criteria_config['criteria_model'])

        # get criteria model
        init_seed(config['seed'], True)
        self.criteria_model = get_model(self.criteria_model_name)(criteria_config, dataset)

        # get overall rating model
        overall_config = copy.deepcopy(config)
        overall_config.final_config_dict.update(overall_config['overall_model'])
        self.overall_model_name = config['overall_model']['model']
        init_seed(config['seed'], True)
        self.overall_model = get_model(self.overall_model_name)(overall_config, dataset)

    def predict(self, interaction):

        dic_criteria_ratings = {}

        # get criteria rating prediction
        criteria_ratings = self.criteria_model.predict(interaction)

        # clamp predicted rating value between [min, max] and round to closed integer rating
        criteria_ratings = torch.clamp(criteria_ratings, min=self.min_rating_value, max=self.max_rating_value)

        # sort criteria ratings
        if self.sorting_weight > 0 and self.sorting_algorithm:
            sort_score = self.sorting_algorithm.sort(interaction, criteria_ratings)
            sort_score = self.min_max_scale(sort_score).to(self.device)
        else:
            sort_score = torch.zeros(criteria_ratings.shape[0]).to(self.device)

        # calculate the weighted sum of sorting score and overall rating score as the final sort_score
        if self.sorting_weight < 1:
            # convert to Interaction type
            dic_criteria_ratings[self.criteria_vector_name] = criteria_ratings
            interaction_criteria_ratings = Interaction(dic_criteria_ratings)

            # get overall rating
            overall_rating_score = self.overall_model.predict(interaction_criteria_ratings)

            # calculate weighted score
            sort_score = self.sorting_weight * sort_score + (1 - self.sorting_weight) * overall_rating_score

        return sort_score
