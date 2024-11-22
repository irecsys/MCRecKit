# @Time   : 2021/12/27
# @Author : Yong Zheng and David Wang

import torch
import torch.nn as nn
from recbole.model.layers import MLPLayers
from mcreckit.model.multi_criteria_recommender.jointrp import JointRP


class MONeuMF(JointRP):
    """
        MONeuMF = Multi-Output NeuMF Model
        This model can predict multi-criteria ratings by using a process of joint optimization.
        Reference:
            N. Nassar, A. Jafar, and Y. Rahhal, "Multi-criteria collaborative filtering
            recommender by fusing deep neural network and matrix factorization",
            J. Big Data, vol. 7, no. 1, pp. 1–12, Dec. 2020.
    """

    def __init__(self, config, dataset):
        super(MONeuMF, self).__init__(config, dataset)

        # load parameters info
        self.mf_embedding_size = config['mf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.mf_train = config['mf_train']
        self.mlp_train = config['mlp_train']
        self.use_pretrain = config['use_pretrain']
        self.mf_pretrain_path = config['mf_pretrain_path']
        self.mlp_pretrain_path = config['mlp_pretrain_path']
        self.criteria_weights = config['criteria_weights']

        # check criteria weights to match number of rating inputs
        if len(self.criteria_weights) != len(config['MULTI_LABEL_FIELD']):
            raise ValueError(f"'criteria_weights' and 'MULTI_LABEL_FIELD' must have same number element: \n"
                             f"criteria_weights: {self.criteria_weights} \n"
                             f"MULTI_LABEL_FIELD: {config['MULTI_LABEL_FIELD']}")

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size).to(self.device)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size).to(self.device)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size).to(self.device)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size).to(self.device)

        self.mlp_layers = MLPLayers([2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob).to(
            self.device)

        self.mlp_layers.logger = None  # remove logger to use torch.save()

        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_size[-1], self.num_criteria).to(
                self.device)
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, self.num_criteria).to(self.device)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], self.num_criteria).to(self.device)

        self.sigmoid = nn.LeakyReLU()
        self.loss = nn.MSELoss()

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)

    def forward(self, user, item):
        """Calculate network output for a given user and item
        Args:
            user: tensor of user index
            item: tensor of item index
        Returns:
            list of output of each network of criteria rating
        """

        outputs = []
        user_mf_e = self.user_mf_embedding(user).to(self.device)
        item_mf_e = self.item_mf_embedding(item).to(self.device)
        user_mlp_e = self.user_mlp_embedding(user).to(self.device)
        item_mlp_e = self.item_mlp_embedding(item).to(self.device)

        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e), -1))  # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = self.sigmoid(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        elif self.mf_train:
            output = self.sigmoid(self.predict_layer(mf_output))
        elif self.mlp_train:
            output = self.sigmoid(self.predict_layer(mlp_output))
        else:
            raise RuntimeError('mf_train and mlp_train can not be False at the same time')

        for i in range(self.num_criteria):
            outputs.append(output[:, i].squeeze(-1))

        return torch.stack(outputs, dim=1)
