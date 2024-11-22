import copy
import torch

from mcreckit.model.abstract_recommender import MultiCriteriaRecommender
from mcreckit.utils import get_model
from recbole.utils import init_seed


class HybridRP(MultiCriteriaRecommender):
    """This model combines multiple rating prediction models through a weighted linear aggregation
    """

    def __init__(self, config, dataset):
        super(HybridRP, self).__init__(config, dataset)

        # check basic settings in config file
        if 'ranking_weight' in config and 'sub_models' in config:
            # check size

            self.ranking_weight = config['ranking_weight']
            sub_model_config = config['sub_models']
        else:
            raise ValueError(f"Either 'ranking_weight' or 'sub_models' is not in config file")

        # normalize ranking weight
        self.ranking_weight = [weight / sum(self.ranking_weight) for weight in self.ranking_weight]

        # get list of sub model names and ranking weight
        self.sub_model_name = [sub_model['model'] for sub_model in sub_model_config]

        # create sub model object
        self.sub_model = []
        for model in sub_model_config:
            # get config for each sub model
            model_config = copy.deepcopy(config)
            model_config.final_config_dict.update(model)

            # get sub model object
            init_seed(config['seed'], True)
            self.sub_model.append(get_model(model['model'])(model_config, dataset))

    def predict(self, interaction):

        # calculated weighted sum of score from each sub model prediction
        final_score = torch.zeros(interaction.interaction['user_id'].shape[0])
        for i, model in enumerate(self.sub_model):
            if self.ranking_weight[i] == 0:
                continue

            model_output = model.predict(interaction)
            # if there are multiple outputs, get the overall rating score
            if len(model_output.shape) > 1 and model_output.shape[1] > 1:
                label_score = model_output[:, model.criteria_label.index(model.LABEL)]
            else:
                label_score = model_output

            # normalize the score
            label_score = self.min_max_scale(label_score).to(self.device)
            final_score = final_score.to(self.device) + self.ranking_weight[i] * label_score

        return final_score
