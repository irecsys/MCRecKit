# -*- coding: utf-8 -*-
# @Time : 2026/03
# @Author : Yong Zheng

import torch
import torch.nn as nn
from mcreckit.model.abstract_recommender import MultiCriteriaRecommender
from mcreckit.utils import MCModelType


class MCTF(MultiCriteriaRecommender):
    """
    MCTF (Multi-Criteria Tensor Factorization) model based on HOSVD.
    Reference: Hong, M., & Jung, J. J. (2021). "Multi-criteria tensor model for tourism recommender systems". Expert Systems with Applications, 170, 114537.
    """
    type = MCModelType.MULTICRITERIA

    def __init__(self, config, dataset):
        if 'eval_args' in config and 'split' in config['eval_args']:
            if len(config['eval_args']['split']) > 1:
                config['eval_args']['split'] = {'CV': config['eval_args']['split']['CV']}

        super(MCTF, self).__init__(config, dataset)

        latent_dims = config.get('latent_dims') or config.get('mf_embedding_size') or 64
        latent_dims = int(latent_dims)

        self.d_u = latent_dims
        self.d_i = latent_dims
        self.d_r = latent_dims

        # Core tensor Z
        self.core_tensor = nn.Parameter(torch.randn(self.d_u, self.d_i, self.d_r))

        # Factor matrices U, I, R
        self.U = nn.Embedding(self.n_users, self.d_u)
        self.I = nn.Embedding(self.n_items, self.d_i)
        self.R = nn.Embedding(len(self.criteria_label), self.d_r)

        self.loss = nn.MSELoss()
        self._init_weights()

    def _init_weights(self):
        """ Initialize all tensor weights to small values close to zero to prevent numerical explosion from multiplicative accumulation """
        nn.init.normal_(self.core_tensor, mean=0.0, std=0.01)
        nn.init.normal_(self.U.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.I.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.R.weight, mean=0.0, std=0.01)

    def forward(self, user, item):
        u_f = self.U(user)
        i_f = self.I(item)
        preds = torch.einsum('xyz,bx,by,mz->bm', self.core_tensor, u_f, i_f, self.R.weight)
        return preds

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        if self.predicted_label_name in interaction:
            label = interaction[self.predicted_label_name]
        else:
            label_list = [interaction[field] for field in self.criteria_label]
            label = torch.stack(label_list, dim=-1).float()

        preds = self.forward(user, item)
        current_loss = self.loss(preds, label)

        return current_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        preds = self.forward(user, item)
        return preds.mean(dim=-1)