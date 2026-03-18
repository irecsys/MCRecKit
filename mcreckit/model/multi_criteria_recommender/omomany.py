import torch
import torch.nn as nn
import numpy as np
from mcreckit.model.abstract_recommender import MultiCriteriaRecommender
from mcreckit.utils import MCModelType


class OMOMany(MultiCriteriaRecommender):
    '''
    Using OWA/TOPSIS to aggregate multi-criteria ratings into a single score for similarity calculation.
    Reference: Anwar, K., Wasid, M., Zafar, A., & Iqbal, A. (2025). An efficient framework to combat multidimensionality in multi-criteria recommender systems. Cluster Computing, 28(4), 245.
    '''
    type = MCModelType.MULTICRITERIA

    def __init__(self, config, dataset):
        super(OMOMany, self).__init__(config, dataset)

        # 1. get configurations
        self.rating_range = config['RATING_RANGE']
        self.rating_midpoint = (self.rating_range[0] + self.rating_range[1]) / 2.0
        self.LABEL_FIELD = config['LABEL_FIELD']
        self.k_neighbors = config['k_neighbors'] or 40
        self.device = config['device']
        self.dummy_param = nn.Parameter(torch.zeros(1))

        # 2. read hyper-parameters
        self.owa_quantifier = config.get('owa_quantifier', 'many').lower()
        self.similarity_type = config.get('similarity_type', 'OS').upper()

        # 3. calculate OWA weights dynamically
        self.owa_weights = self._get_owa_weights(self.owa_quantifier)

        self.total_similarity = None
        self.user_mean_ratings = None
        self.rating_matrix = None

    def _get_owa_weights(self, quantifier):
        """Calculate OWA weights based on fuzzy linguistic quantifiers"""
        n = len(self.criteria_label)
        # set parameters a and b
        if quantifier == "most":
            a, b = 0.0, 0.5
        elif quantifier == "least":
            a, b = 0.5, 1.0
        else:  # "many"
            a, b = 0.3, 0.8

        def Q(r):
            if r < a: return 0.0
            if r > b: return 1.0
            return (r - a) / (b - a)

        weights = []
        for j in range(1, n + 1):
            weights.append(Q(j / n) - Q((j - 1) / n))
        return torch.tensor(weights, device=self.device).float()

    def _precompute_similarities(self, interaction):
        """Fusing similarities based on the logic in the paper"""
        user_ids = interaction[self.USER_ID]
        item_ids = interaction[self.ITEM_ID]
        ratings = interaction[self.LABEL_FIELD]

        self.rating_matrix = torch.zeros((self.n_users, self.n_items), device=self.device)
        self.rating_matrix[user_ids, item_ids] = ratings.float()
        mask = self.rating_matrix > 0
        self.user_mean_ratings = (self.rating_matrix.sum(dim=1) / (mask.sum(dim=1) + 1e-9))

        # 1. calculate base similarity (CRS or OS)
        if self.similarity_type == "CRS":
            # 计算 CRS 相似度 (JTC * CRWF)
            common_mask = mask.float()
            intersection = torch.mm(common_mask, common_mask.t())
            union = common_mask.sum(dim=1).unsqueeze(1) + common_mask.sum(dim=1).unsqueeze(0) - intersection
            sim_overall = intersection / (union + 1e-9)  # Simplified JTC
        else:
            # Adopting a simplified OS similarity based on common item counts and average rating differences
            N = self.n_items
            common_counts = torch.mm(mask.float(), mask.float().t())
            sim_pncr = torch.exp(-(N - common_counts) / N)  # [cite: 185]

            # Calculate ADF rating differences in batches to avoid memory issues
            abs_diff_matrix = torch.zeros((self.n_users, self.n_users), device=self.device)
            for i in range(0, self.n_users, 128):
                end = min(i + 128, self.n_users)
                diffs = torch.abs(self.rating_matrix[i:end].unsqueeze(1) - self.rating_matrix.unsqueeze(0))
                # 仅计算共同项
                valid_diffs = diffs * (self.rating_matrix[i:end].unsqueeze(1) > 0) * (
                            self.rating_matrix.unsqueeze(0) > 0)
                abs_diff_matrix[i:end] = valid_diffs.sum(dim=-1)

            avg_abs_diff = abs_diff_matrix / (common_counts + 1e-9)
            sim_adf = torch.exp(-avg_abs_diff) * (common_counts / N)
            sim_overall = sim_pncr * sim_adf

        # 2. Fuse multi-criteria ratings using OWA aggregation (normalized and aggregated)
        mc_labels = [interaction[field] for field in self.criteria_label]
        mc_ratings = torch.stack(mc_labels, dim=-1).float()
        mc_norm = (mc_ratings - self.rating_range[0]) / (self.rating_range[1] - self.rating_range[0] + 1e-9)
        sorted_mc, _ = torch.sort(mc_norm, dim=-1, descending=True)
        aggregated_scores = torch.sum(sorted_mc * self.owa_weights, dim=-1)

        # 3. Calculate MED similarity and fuse
        agg_matrix = torch.zeros((self.n_users, self.n_items), device=self.device)
        agg_matrix[user_ids, item_ids] = aggregated_scores
        dist = torch.cdist(agg_matrix, agg_matrix, p=2)
        sim_med = 1.0 / (1.0 + dist)

        # Calculate the final fused similarity
        self.total_similarity = sim_overall * sim_med

    def calculate_loss(self, interaction):
        if self.total_similarity is None:
            self._precompute_similarities(interaction)
        return self.dummy_param * 0

    def predict(self, interaction):
        user_ids, item_ids = interaction[self.USER_ID], interaction[self.ITEM_ID]
        if self.total_similarity is None:
            return torch.full_like(user_ids, self.rating_midpoint).float()

        if not hasattr(self, 'diff_matrix'):
            self.diff_matrix = (self.rating_matrix - self.user_mean_ratings.unsqueeze(1)) * (self.rating_matrix > 0)

        # Acquire top-k similar users for each target user and compute weighted average of their rating differences for the target item
        batch_sims = self.total_similarity[user_ids]
        item_rated_mask = (self.rating_matrix[:, item_ids] > 0).float().t()
        masked_sims = batch_sims * item_rated_mask

        topk_sims, topk_idx = torch.topk(masked_sims, self.k_neighbors, dim=1)
        batch_diffs = self.diff_matrix[topk_idx, item_ids.unsqueeze(1)]

        sum_abs_sim = torch.sum(torch.abs(topk_sims), dim=1) + 1e-9
        weighted_diff = torch.sum(topk_sims * batch_diffs, dim=1) / sum_abs_sim
        return self.user_mean_ratings[user_ids] + weighted_diff