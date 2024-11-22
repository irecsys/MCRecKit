# @Time : 2021/12/04
# @Author : David Wang

import torch
import torch.nn as nn
from recbole.model.layers import MLPLayers
from mcreckit.model.abstract_recommender import MultiCriteriaRecommender


class OverallRP(MultiCriteriaRecommender):
    """
        OverallRP = Overall Rating Prediction
        This model defined the process of estimating the overall rating via MLP
    """

    def __init__(self, config, dataset):
        super(OverallRP, self).__init__(config, dataset)

        # special data properties
        self.rating_range = config['RATING_RANGE'][1] - config['RATING_RANGE'][0] + 2

        # get network parameters
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.use_pretrain = config['use_pretrain']

        # define embedding and network layers
        if self.mlp_embedding_size is not None and self.mlp_embedding_size > 0:
            self.criteria_mlp_embedding = {}

            # set up embedding layer for each criteria
            for label in self.criteria_label:
                self.criteria_mlp_embedding[label] = nn.Embedding(self.rating_range, self.mlp_embedding_size).to(
                    self.device)

            # set up MLP layer
            self.mlp_layers = MLPLayers([self.num_criteria * self.mlp_embedding_size] + self.mlp_hidden_size,
                                        self.dropout_prob, activation='leakyrelu').to(self.device)
        else:
            self.mlp_layers = MLPLayers([self.num_criteria] + self.mlp_hidden_size, self.dropout_prob,
                                        activation='leakyrelu').to(self.device)

        self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1).to(self.device)
        self.sigmoid = nn.LeakyReLU()
        self.loss = nn.MSELoss()

        # initialize weights
        self.apply(self._init_weights)

    def forward(self, criteria_ratings):
        """Calculate network output given a set of criteria rating tensor
        Args:
            criteria_ratings: criteria rating in tensor format
        Returns:
            rating prediction
        """
        if self.mlp_embedding_size > 0:
            # change criteria rating to integer type, used as embedding index
            criteria_ratings = criteria_ratings.int()
            criteria_embedding = ()
            for i, label in enumerate(self.criteria_mlp_embedding):
                criteria_embedding = criteria_embedding + (self.criteria_mlp_embedding[label](criteria_ratings[:, i]),)
            embedding_output = torch.cat(criteria_embedding, -1)
            mlp_output = self.mlp_layers(embedding_output)
        else:
            mlp_output = self.mlp_layers(criteria_ratings)

        output = self.sigmoid(self.predict_layer(mlp_output))
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        criteria_ratings = interaction[self.criteria_vector_name]
        label = interaction[self.LABEL]

        output = self.forward(criteria_ratings)
        return self.loss(output, label)

    def predict(self, interaction):
        criteria_ratings = interaction[self.criteria_vector_name]

        predicted_rating = self.forward(criteria_ratings)

        # clamp prediction if evaluation is based on label value
        if self.eval_mode == 'labeled':
            predicted_rating = self.clamp(predicted_rating)

        return predicted_rating
