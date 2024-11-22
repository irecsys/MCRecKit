# @Time : 2021/12/04
# @Author : David Wang

import torch
import torch.nn as nn

from mcreckit.model.abstract_recommender import MultiCriteriaRecommender
from mcreckit.utils import get_model
from recbole.utils import init_seed


class CriteriaRP(MultiCriteriaRecommender):
    """
        CriteriaRP = Criteria Rating Prediction
        This model defines the process of predicting multi-criteria ratings
    """

    def __init__(self, config, dataset):
        super(CriteriaRP, self).__init__(config, dataset)

        # get general model
        self.general_model_name = config['GENERAL_MODEL']
        init_seed(config['seed'], True)
        self.general_model = get_model(config['GENERAL_MODEL'])(config, dataset)

        # modify general model set up
        self.general_model.sigmoid = nn.LeakyReLU()
        self.general_model.loss = nn.MSELoss()

        self.trained_weights = {}

    def predict(self, interaction):

        predicted_results = ()
        for label in self.criteria_label:
            self.general_model.load_state_dict(self.trained_weights[label])
            predicted_rating = self.general_model.predict(interaction)

            # clamp prediction if evaluation is based on label value
            if self.eval_mode == 'labeled':
                predicted_rating = self.clamp(predicted_rating)

            predicted_results = predicted_results + (predicted_rating,)

        return torch.stack(predicted_results, dim=1)
