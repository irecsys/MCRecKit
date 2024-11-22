# @Time   : 2021/12/27
# @Author : David Wang

import torch
from mcreckit.model.abstract_recommender import MultiCriteriaRecommender


class JointRP(MultiCriteriaRecommender):
    """
        JointRP = Joint Rating Prediction
        This model defines the process of joint optimization
    """

    def __init__(self, config, dataset):
        super(JointRP, self).__init__(config, dataset)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        # get predicted rating for each user and item
        outputs = self.forward(user, item)

        # add loss for each criteria rating
        joint_loss = 0
        for i, label in enumerate(self.criteria_label):
            joint_loss += self.loss(outputs[:, i], interaction[label]) * self.criteria_weights[i]

        # normalize by total weight
        joint_loss = joint_loss / sum(self.criteria_weights)

        return joint_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        predicted_rating = self.forward(user, item)

        # clamp prediction if evaluation is based on label value
        if self.eval_mode == 'labeled':
            predicted_rating = self.clamp(predicted_rating)

        # apply soring algorithm
        if self.sorting_weight > 0:

            # check if overall rating is in the criteria labels
            try:
                # get index if overall rating.
                # This is needed since overall rating may be at any position in the config file
                overall_label_idx = self.criteria_label.index(self.LABEL)
            except ValueError:
                raise ValueError(f"'{self.LABEL}' not in 'MULTI_LABEL_FIELD' setting: {self.criteria_label}")

            # get predicted overall rating
            predicted_overall_rating = predicted_rating[:, overall_label_idx].unsqueeze(1)

            # get predicted criteria rating
            predicted_criteria_rating = predicted_rating[:, [i for i in range(len(self.criteria_label))
                                                             if i != overall_label_idx]]

            # calculate sorting score
            sort_score = self.sorting_algorithm.sort(interaction, predicted_criteria_rating)

            print('start Pareto ranking...')
            # map sort score to value in [min, max] rating
            sort_score = self.min_max_scale(sort_score).unsqueeze(1).to(self.device)

            # calculate weighted sum
            predicted_overall_rating = self.sorting_weight * sort_score \
                                       + (1 - self.sorting_weight) * predicted_overall_rating

            # combine weighted overall rating and criteria rating together
            if overall_label_idx == 0:
                predicted_rating = torch.cat((predicted_overall_rating, predicted_criteria_rating), -1)
            elif overall_label_idx == len(self.criteria_label) - 1:
                predicted_rating = torch.cat((predicted_criteria_rating, predicted_overall_rating), -1)
            else:
                predicted_rating = \
                    torch.cat((predicted_criteria_rating[:, range(overall_label_idx)],
                               predicted_overall_rating,
                               predicted_criteria_rating[:,
                               range(overall_label_idx, predicted_criteria_rating.shape[1])]
                               ), -1)

        return predicted_rating
