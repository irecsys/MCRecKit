# @Time : 2022/02/21
# @Author : David Wang

import copy
import torch

from mcreckit.model.abstract_recommender import MultiCriteriaRecommender
from mcreckit.utils import get_model
from recbole.utils import init_seed


class CriteriaSort(MultiCriteriaRecommender):
    """
        This model combine two stages:
        1). predicting multi-criteria ratings
        2). ranking items from predicted multi-criteria ratings by using multi-criteria rankings (e.g., Pareto ranking)

        reference (using major sorting only):
        Zheng, Yong, and David Wang. "Multi-criteria ranking: Next generation of multi-criteria recommendation framework." IEEE Access 10 (2022): 90715-90725.

        reference (using relaxed sorting):
        Zheng, Yong, and David Xuejun Wang. "Multi-Criteria Ranking by Using Relaxed Pareto Ranking Methods." Adjunct Proceedings of ACM UMAP. 2023.

        reference (using subsorting)
        Zheng, Yong, and David Xuejun Wang. "Hybrid Multi-Criteria Preference Ranking by Subsorting." arXiv preprint arXiv:2306.11233 (2023).
    """

    def __init__(self, config, dataset):
        super(CriteriaSort, self).__init__(config, dataset)

        # get criteria model for criteria rating prediction
        self.criteria_model_name = config['criteria_model']['model']
        if config['sub_sort'] == 'OverallRatingRanking':
            if config['overall_model']:
                self.overall_model_name = config['overall_model']['model']
            else:
                raise ValueError("'overall_model' is missing from config file")
        else:
            self.overall_model_name = None

        # get config for criteria rating model
        criteria_config = copy.deepcopy(config)
        criteria_config.final_config_dict.update(criteria_config['criteria_model'])

        # get criteria model
        init_seed(config['seed'], True)
        self.criteria_model = get_model(self.criteria_model_name)(criteria_config, dataset)

        # create overall rating model for overall rating prediction if needed
        if self.overall_model_name is not None:
            overall_config = copy.deepcopy(config)
            overall_config.final_config_dict.update(overall_config['overall_model'])
            init_seed(config['seed'], True)
            self.overall_model = get_model(self.overall_model_name)(overall_config, dataset)
        else:
            self.overall_model = None

    def predict(self, interaction):

        # get criteria rating prediction
        criteria_ratings = self.criteria_model.predict(interaction)

        # clamp predicted rating value between [min, max] and round to closed integer rating
        criteria_ratings = torch.clamp(criteria_ratings, min=self.min_rating_value, max=self.max_rating_value)

        # get overall rating is needed
        if self.overall_model is not None:
            # add predicted criteria rating to interaction data for overall rating prediction
            interaction.interaction[self.overall_model.criteria_vector_name] = criteria_ratings
            overall_ratings = self.overall_model.predict(interaction)
            overall_ratings = torch.clamp(overall_ratings, min=self.min_rating_value, max=self.max_rating_value)
        else:
            overall_ratings = None

        # sort criteria ratings
        sort_score = self.sorting_algorithm.sort(interaction, criteria_ratings, overall_ratings)
        sort_score = self.min_max_scale(sort_score).to(self.device)

        return sort_score
