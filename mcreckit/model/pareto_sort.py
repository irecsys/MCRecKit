# @Time   : 2022/01/19
# @Author : David Wang, Yong Zheng

"""
    Implementing all Pareto sorting algorithms here
    Usage: import recbole.model.pareto_sort
"""

import importlib
import itertools
import logging
import math
import numpy as np
import operator
import torch

from multiprocessing.dummy import Pool as ThreadPool
from numba import njit, cuda
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

@njit(fastmath=True)
def greater_equal(lt1, lt2, margin):
    greater = True
    greater_done = False
    smaller = True
    smaller_done = False
    conf = False
    eq = True

    for a, b in zip(lt1, lt2):
        if a > b - margin:
            if greater_done is False:
                smaller = False
                smaller_done = True
                eq = False
            else:
                conf = True
                break
        elif a < b + margin:
            if smaller_done is False:
                greater = False
                greater_done = True
                eq = False
            else:
                conf = True
                break
    if eq:
        conf = True
    return greater, smaller, conf


class FavourRelationGraph:
    def __init__(self, criteria_scores, favor='bigger'):
        """
        Args:
            criteria_scores: an array of dimension [n, k], n is number of scores, k is number of criteria
            favor: 'bigger' or 'smaller'
        """
        self.criteria_scores = criteria_scores
        self.num_criteria = criteria_scores.shape[1]
        self.num_scores = criteria_scores.shape[0]

        if favor == 'smaller':
            self.favour_op = operator.lt
        elif favor == 'bigger':
            self.favour_op = operator.gt
        else:
            raise ValueError(f'favour operator {favor} not supported')

        # create the graph without edge, each score is a node
        self.graph = {idx: [] for idx in range(self.num_scores)}
        self.scc_list = []
        self.scc_rank = None

    def build_edges(self):
        """Build relation between each score based on Favour Relation
        """
        for i in range(self.num_scores):
            for j in range(i + 1, self.num_scores):
                favour_i = self.favour_op(self.criteria_scores[i], self.criteria_scores[j]).sum()
                favour_j = self.favour_op(self.criteria_scores[j], self.criteria_scores[i]).sum()
                if favour_i > favour_j:
                    # solution i is favour than solution j
                    self.graph[i].append(j)
                if favour_i < favour_j:
                    # solution j is favour than solution i
                    self.graph[j].append(i)

    def dfs(self, start_node, end_node):
        """This is the implementation of Deep First Search (DFS) method to find loop in the graph
        Args:
            start_node: node index to search
            end_node: node index to search
        Returns:
            None
        """
        fringe = [(start_node, set())]
        while fringe:
            node, path = fringe.pop()
            if path and node == end_node:
                return path
            for next_node in self.graph[node]:
                if next_node in path:
                    continue
                fringe.append((next_node, path.union({next_node})))

        return None

    def partition_graph(self):
        """Partition graph nodes into Strongly Connected Components (SCC), based on the loop connection
        """

        # scc = [path.union({node}) for node in self.graph for path in self.dfs(self.graph, node, node)]
        scc = []
        found_nodes = set()
        for node in self.graph:
            if node not in found_nodes:
                path = self.dfs(node, node)
                if path is None:
                    scc.append({node})
                    found_nodes = found_nodes.union({node})
                elif path not in scc:
                    scc.append(path)
                    found_nodes = found_nodes.union(path)

        self.scc_list = scc

    def perform_level_sorting(self):
        """Determine the rank of SCC
        """
        self.scc_rank = [0] * len(self.scc_list)
        for i in range(len(self.scc_list)):
            for j in range(i + 1, len(self.scc_list)):
                scc_i = self.scc_list[i]
                scc_j = self.scc_list[j]
                for a, b in itertools.product(scc_i, scc_j):
                    if b in self.graph[a]:
                        self.scc_rank[i] += 1
                        break

    def calculate_criteria_rank(self):
        """Output find ranking for each criteria rating
        Returns:
            array of ranking
        """
        criteria_rank = [0] * self.num_scores

        for node in self.graph:
            for idx in range(len(self.scc_list)):
                if node in self.scc_list[idx]:
                    criteria_rank[node] = self.scc_rank[idx]
                    break

        return np.array(criteria_rank)

    def get_fr_ranks(self):
        """Thi is the method to execute all the steps to get Favour Relation ranks
        Returns:
            array of FR rank score (the higher, the better)
        """
        self.build_edges()
        self.partition_graph()
        self.perform_level_sorting()

        return self.calculate_criteria_rank()


class MultiCriteriaSort(object):
    """This is the base class for all Pareto Sort algorithms

    """

    def __init__(self, config, bigger_is_better=True):
        """class initialization
        Args:
            config: a Config object
            bigger_is_better: True if bigger rating is better, default to True
        """
        self.config = config
        self.bigger_is_better = bigger_is_better
        self.max_k = max(config['topk'])
        self.device = config['device']
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.criteria_label = config['MULTI_LABEL_FIELD']

        self.sorting_label = config['sorting_label']
        self.sorting_label_index = self.get_sorting_label_index()

        self.history = None
        # set up dominance compare operator
        if self.bigger_is_better:
            # self.dominate_operator = np.greater_equal
            self.dominate_operator = self.greater_equal
        else:
            # self.dominate_operator = np.less_equal
            self.dominate_operator = self.less_equal

        # determine if use GPU or not
        # cannot use self.device here, since it gives numba error when allocating GPU memory for unknown reason
        if self.device.type == 'cuda':
            self.use_gpu = True
        else:
            self.use_gpu = False

    def get_sorting_label_index(self):
        """Get index in the criteria label list
        """
        if self.sorting_label is None:
            return None

        label_index = []
        for label in self.sorting_label:
            label_index.append(self.criteria_label.index(label))

        return label_index

    def setHistoryItems(self, history_items):
        self.history = history_items

    def cpu_sort_algorithm(self, criteria_ratings):
        """Must be implemented in sub class"""
        raise None

    @staticmethod
    def sort_cuda_kernel(ratings, num_items, ranks, dominated_scores):
        """CUDA kernel function to sort. Need to be implemented for each subclass
        Args:
            ratings: two dimension array with ratings for multiple users
            num_items: one dimension array with only one element
            ranks: output variable for rankings
            dominated_scores: used for local sort
        """
        return None

    @staticmethod
    def greater_equal(lt1, lt2):
        greater_eq = True
        greater_eq_done = False
        eq = True
        eq_done = False

        for a, b in zip(lt1, lt2):
            if greater_eq_done is False and a < b:
                greater_eq = False
                greater_eq_done = True
            if eq_done is False and a != b:
                eq = False
                eq_done = True
            if greater_eq_done and eq_done:
                break
        return greater_eq, eq

    @staticmethod
    def less_equal(lt1, lt2):
        less_eq = True
        less_eq_done = False
        eq = True
        eq_done = False

        for a, b in zip(lt1, lt2):
            if less_eq_done is False and a > b:
                less_eq = False
                less_eq_done = True
            if eq_done is False and a != b:
                eq = False
                eq_done = True
            if less_eq_done and eq_done:
                break
        return less_eq, eq

    def sort_parallel(self, interaction, criteria_ratings):
        """The method to perform sorting (self.cpu_sort_algorithm()) for a given user in CPU parallel mode
        Args:
            interaction: a tensor of user item pair
            criteria_ratings: tensor of criteria ratings for each (user, item) pair
        Return:
            a tensor object with ranking score for each (user, item) pair
        """
        if len(criteria_ratings) != len(interaction):
            raise ValueError(f'Number of criteria ratings ({len(criteria_ratings)}) is not the same as number of user '
                             f'item pairs ({len(interaction)})')

        num_of_ratings = len(criteria_ratings)

        # non dominated list for each criteria rating
        rank_scores = np.zeros(num_of_ratings)

        last_uid = None
        idx = 0
        uid_idx_list = []
        uid_criteria_rating_list = []
        process_criteria_rating_list = []
        process_uid_idx_list = []

        for uid in interaction.interaction[self.USER_ID]:
            # encounter a new user id
            if last_uid and last_uid != uid:
                # save data for parallel processing
                process_criteria_rating_list.append(torch.stack(uid_criteria_rating_list, dim=0).cpu().numpy())
                # process_criteria_rating_list.append(torch.stack(uid_criteria_rating_list, dim=0))
                process_uid_idx_list.append(uid_idx_list)

                # reset idx list
                uid_idx_list = []
                uid_criteria_rating_list = []

            uid_criteria_rating_list.append(criteria_ratings[idx])
            uid_idx_list.append(idx)
            last_uid = uid
            idx += 1

        # save data for last uid
        process_criteria_rating_list.append(torch.stack(uid_criteria_rating_list, dim=0).cpu().numpy())
        process_uid_idx_list.append(uid_idx_list)

        # build multiprocessing pool and start parallel run
        pool = ThreadPool()
        result_pool = pool.map(self.cpu_sort_algorithm, process_criteria_rating_list)
        pool.close()
        pool.join()

        # collect data from result pool
        for uid_idx_list, sorted_score in zip(process_uid_idx_list, result_pool):
            for i, uid_idx in enumerate(uid_idx_list):
                rank_scores[uid_idx] = sorted_score[i]

        return torch.tensor(rank_scores)

    def sort_gpu(self, ratings, num_items, dominated_scores=None):
        """Perform non dominated sort in GPU CUDA platform. If the parameter is different, the subclass must implement
        its own method
        Args:
            ratings: two-dimensional array
            num_items: number of items per user id
            dominated_scores: item dominated scores, used to sub sorting
        Return:
             one dimension array
        """
        if num_items is None:
            raise ValueError('Number of items is not assigned')

        ranks = np.zeros(ratings.shape[0])

        # load array to GPU memory
        ratings_gpu = cuda.to_device(ratings)
        num_items_gpu = cuda.to_device([num_items])
        ranks_gpu = cuda.to_device(ranks)
        if dominated_scores is not None:
            dominated_scores_gpu = cuda.to_device(dominated_scores)
        else:
            dominated_scores_gpu = cuda.to_device(np.zeros(ratings.shape[0]))

        # call CUDA kernel
        self.sort_cuda_kernel[2000, 256](ratings_gpu, num_items_gpu, ranks_gpu, dominated_scores_gpu)

        # return the results from GPU memory
        return ranks_gpu.copy_to_host()

    def sort(self, interaction, criteria_ratings, overall_ratings, dominated_scores=None):
        """A standard interface to perform the sort.
        Args:
            interaction: a tensor of user item pair
            criteria_ratings: tensor of criteria ratings for each (user, item) pair
            overall_ratings: tensor of overall rating prediction. default to None
            dominated_scores: item dominated scores, used to sub sorting
        Return:
            a tensor object with ranking score for each (user, item) pair
        """
        if len(criteria_ratings) != len(interaction):
            raise ValueError(f'Number of criteria ratings ({len(criteria_ratings)}) is not the same as number of user '
                             f'item pairs ({len(interaction)})')

        if self.sorting_label:
            sorting_ratings = criteria_ratings[:, self.sorting_label_index]
        else:
            sorting_ratings = criteria_ratings

        if self.use_gpu:
            num_of_items = interaction.interaction[self.ITEM_ID].unique().shape[0]
            num_of_users_items = interaction.interaction[self.USER_ID].shape[0]

            if num_of_users_items % num_of_items != 0:
                raise ValueError(f'User item rating is not complete, cannot use this batch sort on GPU ')

            # sort ratings with GPU
            dominated_score = self.sort_gpu(sorting_ratings.cpu().numpy(), num_of_items, dominated_scores)
        else:
            dominated_score = self.sort_parallel(interaction, sorting_ratings)

        return torch.tensor(dominated_score)

    @staticmethod
    def lp_norm(rating, ideal, p, weight):
        """Calculate L^p norm
        reference: Yuan, Xu, Wang, Evolutionary Many-Objective Optimization Using Ensemble Fitness Ranking,
        GECCO'14, July 12-16, 2014, Vancouver, BC, Canada
        Args:
            rating: two dimension array
            ideal: one dimension array, dim = rating.shape[1]
            p: real number >= 1
            weight: one dimension array, dim = rating.shape[1], with all positive element, and sum = 1
        return:
            one dimension array, dim = rating.shape[0]
        """
        return pow(np.dot(pow(abs(ideal - rating), p), weight), 1 / p)

    @staticmethod
    def tchebycheff_norm(rating, ideal, weight):
        """Calculate L^p norm
        reference: Yuan, Xu, Wang, Evolutionary Many-Objective Optimization Using Ensemble Fitness Ranking,
        GECCO'14, July 12-16, 2014, Vancouver, BC, Canada
        Args:
            rating: two dimension array
            ideal: one dimension array, dim = rating.shape[1]
            weight: one dimension array, dim = rating.shape[1], with all positive element, and sum = 1
        return:
            one dimension array, dim = rating.shape[0]
        """
        return (abs(ideal - rating) / weight).max(axis=1)

    @staticmethod
    def pbi_norm(rating, ideal, weight, theta):
        """Calculate L^p norm
        reference: Yuan, Xu, Wang, Evolutionary Many-Objective Optimization Using Ensemble Fitness Ranking,
        GECCO'14, July 12-16, 2014, Vancouver, BC, Canada
        Args:
            rating: two dimension array
            ideal: one dimension array, dim = rating.shape[1]
            weight: one dimension array, dim = rating.shape[1], with all positive element, and sum = 1
            theta: a real number > 0
        return:
            one dimension array, dim = rating.shape[0]
        """
        d_1 = abs(np.dot((ideal - rating), weight)) / np.linalg.norm(weight)
        d_2 = np.linalg.norm(rating - ideal - np.multiply(d_1[:, None], weight / np.linalg.norm(weight)), axis=1)
        return d_1 + theta * d_2

    @staticmethod
    def min_max_scale(score, upper_bound, lower_bound):
        """scale value in score tensor to value in [lower_bound, upper_bound] with min max method:
            v' = (v - min(score)) / (max(score) - min(score)) * (upper_bound - lower_bound) + lower_bound
        Args:
            score: a tensor object
            upper_bound: upper bound of the scaled score
            lower_bound: lower bound of the scaled score
        Returns:
            a rescaled tensor
        """

        v_min = score.min()
        v_max = score.max()

        if v_max == v_min:
            scaled_score = torch.clamp(score, min=lower_bound, max=upper_bound)
        else:
            scaled_score = (score - v_min) / (v_max - v_min) * (upper_bound - lower_bound) + lower_bound

        return scaled_score


class FastNonDominatedSort(MultiCriteriaSort):
    """A fast soring algorithm based on 'fast non-dominated sorting' proposed by:

    Deb, Kalyanmoy, et al. "A fast elitist non-dominated sorting genetic algorithm for multi-objective optimization:
        NSGA-II." Parallel problem solving from nature PPSN VI. Springer Berlin Heidelberg, 2000.

    """

    def __init__(self, config):
        super().__init__(config)

        if self.use_gpu:
            self.sort_algorithm = self.non_dominated_sort_gpu
        else:
            self.sort_algorithm = self.sort_parallel

        self.error_margin = config['error_margin']
        sub_sort = config['sub_sort']
        if sub_sort:
            self.sub_sort_algorithm = getattr(importlib.import_module('mcreckit.model.pareto_sort'), sub_sort)(config)
            self.local_sort = config['local_sort']
        else:
            self.sub_sort_algorithm = None

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def cpu_sort_algorithm(criteria_ratings, error_margin):
        """perform non dominated sorting and return the ranking for each rating
        Args:
            criteria_ratings: list of multi criteria rating for each item
                'sub_sorting': can be value of
                    None: do not perform sub sorting
                    'average': take average of criteria rating as ranking score
                    'crowding': use crowding measurement as ranking score
            error_margin: used to compare criteria value
        Returns:
            list of rank score
        """

        num_of_items = len(criteria_ratings)

        # non dominated list for each criteria rating
        num_of_dominated = np.zeros(num_of_items)

        for i in range(num_of_items):
            for j in range(i):

                greater, smaller, conf = greater_equal(criteria_ratings[i], criteria_ratings[j], error_margin)
                if conf is False:
                    if greater:
                        num_of_dominated[i] += 1
                    if smaller:
                        num_of_dominated[j] += 1

        return num_of_dominated / num_of_items

    @staticmethod
    @cuda.jit
    def non_dominated_sort_cuda_kernel(ratings, num_items, ranks):
        """CUDA kernel function to dominated sort
        Args:
            ratings: two dimension array with ratings for multiple users
            num_items: one dimension array with only one element
            ranks: output variable for rankings
        """
        error_margin = num_items[1]
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        for idx in range(start, ratings.shape[0], stride):
            dominated = 0
            # calculate the rating item range for each user
            start_id = math.floor(idx / num_items[0]) * num_items[0]
            end_id = start_id + num_items[0]
            for k in range(start_id, end_id):
                if k == idx:
                    continue

                # decide if current rating is better than others, the bigger, the better
                is_greater = True
                is_equal = True
                for j in range(ratings.shape[1]):
                    if ratings[idx][j] < ratings[k][j] + error_margin:
                        is_greater = False
                        is_equal = False
                        break
                    elif ratings[idx][j] > ratings[k][j] - error_margin:
                        is_equal = False

                if is_greater and not is_equal:
                    dominated += 1

            ranks[idx] = dominated

    def non_dominated_sort_gpu(self, interaction, criteria_ratings):
        """Perform non dominated sort in GPU CUDA platform
        Args:
            interaction: user item interaction data
            criteria_ratings: two-dimensional tensor
        Return:
             one dimension array
        """
        num_of_items = interaction.interaction[self.ITEM_ID].unique().shape[0]
        num_of_users_items = interaction.interaction[self.USER_ID].shape[0]

        if num_of_users_items % num_of_items != 0:
            raise ValueError(f'User item rating is not complete, cannot use this batch sort on GPU ')

        # convert tensor to numpy array
        ratings = criteria_ratings.cpu().numpy()

        # create array to store the rank score
        ranks = np.zeros(ratings.shape[0])

        # load array to GPU memory
        ratings_gpu = cuda.to_device(ratings)
        num_items_gpu = cuda.to_device([num_of_items, self.error_margin])
        ranks_gpu = cuda.to_device(ranks)

        # call CUDA kernel
        self.non_dominated_sort_cuda_kernel[2000, 256](ratings_gpu, num_items_gpu, ranks_gpu)

        # return the results from GPU memory
        return ranks_gpu.copy_to_host()

    def sort(self, interaction, criteria_ratings, overall_ratings=None):
        """Customized sort interface
        Args:
            interaction: a tensor of user item pair
            criteria_ratings: tensor of criteria ratings for each (user, item) pair
            overall_ratings: overall rating
        Return:
            a tensor object with ranking score for each (user, item) pair
        """
        if len(criteria_ratings) != len(interaction):
            raise ValueError(f'Number of criteria ratings ({len(criteria_ratings)}) is not the same as number of user '
                             f'item pairs ({len(interaction)})')

        if self.sorting_label:
            sorting_ratings = criteria_ratings[:, self.sorting_label_index]
        else:
            sorting_ratings = criteria_ratings

        # get non dominated rank
        dominated_score = self.sort_algorithm(interaction, sorting_ratings)

        # convert to tensor if needed
        if torch.is_tensor(dominated_score):
            dominated_score.to(self.device)
        else:
            dominated_score = torch.tensor(dominated_score).to(self.device)

        # apply sub sort
        if self.sub_sort_algorithm:
            if not self.local_sort:
                sub_rank_score = self.sub_sort_algorithm.sort(interaction, criteria_ratings, overall_ratings)
            else:
                sub_rank_score = \
                    self.sub_sort_algorithm.sort(interaction, criteria_ratings, overall_ratings, dominated_score)

            # make sure the sub rank score is always in [0, 1)
            sub_rank_score = self.min_max_scale(sub_rank_score, 0.0, 0.9999)

            # add the sub rank score to non-dominated score, assume that the dominated score is always integer
            dominated_score = dominated_score + sub_rank_score.to(self.device)

        return dominated_score / len(criteria_ratings)


class OverallRatingRanking(MultiCriteriaSort):
    """Use the overall rating as ranking score
    """

    def __init__(self, config):
        super().__init__(config)

    def sort(self, interaction, criteria_ratings, overall_ratings):
        overall_ratings_score = self.min_max_scale(overall_ratings, 0.0, 0.9999)

        return overall_ratings_score


class WeightedSumRanking(MultiCriteriaSort):
    """Simply takes weighted sum of all criteria score
    """

    def __init__(self, config):
        super().__init__(config)

        criteria_weight = np.array(config['criteria_weight'])
        if self.sorting_label:
            self.weights = criteria_weight[self.sorting_label_index]

            # check the weights is consistent with number of sorting labels
            if len(self.sorting_label) != len(self.weights):
                raise ValueError(f"The length of 'criteria_weight' is less than length of 'sorting_label'")

        else:
            self.weights = criteria_weight

            # check the weights is consistent with number of criteria labels
            if len(self.criteria_label) != len(self.weights):
                raise ValueError(f"'criteria_weight' setting does not have the same length as 'MULTI_LABEL_FIELD'")

        # normalize if needed
        if self.weights.sum() != 1.0:
            self.weights = self.weights / self.weights.sum()

    def cpu_sort_algorithm(self, criteria_ratings):
        return torch.sum(torch.mul(criteria_ratings, torch.tensor(self.weights).to(self.device)), dim=1)

    def sort(self, interaction, criteria_ratings, overall_ratings):

        if self.sorting_label:
            sorting_ratings = criteria_ratings[:, self.sorting_label_index]
        else:
            sorting_ratings = criteria_ratings

        return self.cpu_sort_algorithm(sorting_ratings)


class AverageRanking(MultiCriteriaSort):
    """A ranking method provided by 'Bentley, P.J., Wakefield, J.P.: Finding Acceptable Solutions in the Pareto-Optimal
    Range using Multiobjective Genetic Algorithms. In: Chawdhry, P.K., Roy, R., Pant, R.K. (eds.) Soft Computing in
    Engineering Design and Manufacturing. Part 5, June 1997, pp. 231–240. Springer, London (1997) '

    For each solution, get the ranking score of each criteria and take the average or sum of ranking across all criteria
    """

    def __init__(self, config):
        super().__init__(config)

    def sort(self, interaction, criteria_ratings, overall_ratings):

        if self.sorting_label:
            sorting_ratings = criteria_ratings[:, self.sorting_label_index]
        else:
            sorting_ratings = criteria_ratings

        return torch.mean(sorting_ratings, dim=1)


class CrowdingDistanceRanking(MultiCriteriaSort):
    """Rank the multi criteria solution based on crowding distance implemented in pymoo
        reference: https://pymoo.org/algorithms/moo/nsga2.html
    """

    def __init__(self, config):
        super().__init__(config)

    def sort(self, interaction, criteria_ratings, overall_ratings):

        if self.sorting_label:
            sorting_ratings = criteria_ratings[:, self.sorting_label_index]
        else:
            sorting_ratings = criteria_ratings

        # get crowding distance
        crowding_distance = calc_crowding_distance(sorting_ratings.cpu().numpy())

        # rescale all distance within (0, 1) and set inf to 1
        real_max = crowding_distance[crowding_distance < np.inf].max()
        if real_max <= 1:
            crowding_distance[crowding_distance == np.inf] = 1.0
        else:
            crowding_distance = crowding_distance / real_max
            crowding_distance[crowding_distance == np.inf] = 1.0

        return torch.tensor(crowding_distance)


class MaximumRanking(MultiCriteriaSort):
    """A ranking method provided by ''Bentley, P.J., Wakefield, J.P.: Finding Acceptable Solutions in the Pareto-Optimal
    Range using Multiobjective Genetic Algorithms. In: Chawdhry, P.K., Roy, R., Pant, R.K. (eds.) Soft Computing in
    Engineering Design and Manufacturing. Part 5, June 1997, pp. 231–240. Springer, London (1997) '

    For each solution, get the ranking score for each criterion and takes max or min among all criteria
    """

    def __init__(self, config):
        super().__init__(config)

    def sort(self, interaction, criteria_ratings, overall_ratings):

        if self.sorting_label:
            sorting_ratings = criteria_ratings[:, self.sorting_label_index]
        else:
            sorting_ratings = criteria_ratings

        return torch.max(sorting_ratings, dim=1).values


class FavourRelationRanking(MultiCriteriaSort):
    """Provided by 'Drechsler, N., Drechsler, R., Becker, B.: Multi-objective Optimisation Based on Relation favour.
    In: Zitzler, E., Deb, K., Thiele, L., Coello Coello, C.A., Corne, D. (eds.) EMO 2001. LNCS, vol. 1993, pp. 154–166.
    Springer, Heidelberg (2001)'

    For each pair of solutions, company number of objectives (criteria) that are better, the solution with the higher
    number is favorable than the other solution with the smaller number. A graph is constructed with 'favour' relation.
    all nodes (solutions) will be partitioned into 'Satisfiability Classes' based on the 'cycle'. Then solutions in the
    same SC has the same rank, the rank of the solution will be determined by the rank of SCs, which do have 'cycle'
    relations.
    """

    def __init__(self, config):
        super().__init__(config)

    def cpu_sort_algorithm(self, criteria_ratings):
        """calculate Favour Relation Ranking
        Args:
            criteria_ratings: list of criteria rankings
        Returns:
            list of Favour Relation Rankings
        """
        fr_graph = FavourRelationGraph(criteria_ratings)
        fr_ranks = fr_graph.get_fr_ranks()

        return torch.tensor(fr_ranks)

    def sort(self, interaction, criteria_ratings, overall_ratings):

        if self.sorting_label:
            sorting_ratings = criteria_ratings[:, self.sorting_label_index]
        else:
            sorting_ratings = criteria_ratings

        return self.sort_parallel(interaction, sorting_ratings)


class GlobalDetriment(MultiCriteriaSort):
    """Proposed by 'Mario Garza-Fabre1, Gregorio Toscano Pulido1, and Carlos A. Coello Coello2,
        Ranking Methods for Many-Objective Optimization,
        A. Hernandez Aguirre et al. (Eds.): MICAI 2009, LNAI 5845, pp. 633–645, 2009. ,
        Springer-Verlag Berlin Heidelberg 2009'
        It calculates the degree that each solution is better than others by aggregating the difference of
        each objective.
    """

    def __init__(self, config):
        super().__init__(config)

    def cpu_sort_algorithm(self, criteria_ratings):
        """Calculate Global Detriment rank for each ratings
        Args:
            criteria_ratings: an array of criteria ratings
        Returns:
            an array of Global Detriment rank score
        """

        num_ratings = criteria_ratings.shape[0]

        # calculate difference matrix
        diff_matrix = np.zeros([num_ratings, num_ratings])
        for i, j in itertools.product(range(num_ratings), range(num_ratings)):
            # find difference for each criterion
            diff = criteria_ratings[i] - criteria_ratings[j]

            # set zero if negative
            diff[diff < 0] = 0
            diff_matrix[i, j] = diff.sum()

        # calculate rank
        gd_ranks = diff_matrix.sum(axis=1)

        return torch.tensor(gd_ranks)

    @staticmethod
    @cuda.jit
    def sort_cuda_kernel(ratings, num_items, ranks, dominated_scores):
        """CUDA kernel function to dominated sort
        Args:
            ratings: two dimension array with ratings for multiple users
            num_items: one dimension array with only one element
            ranks: output variable for rankings
            dominated_scores: used for local sort
        """
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        num_criteria = ratings.shape[1]

        for idx in range(start, ratings.shape[0], stride):
            # get dominated score at idx
            dominated_score_idx = dominated_scores[idx]

            # calculate the rating item range for each user
            start_id = math.floor(idx / num_items[0]) * num_items[0]
            end_id = start_id + num_items[0]

            total_diff = 0
            for k in range(start_id, end_id):
                # only count the item that the dominated score is the same as current idx
                if dominated_score_idx != dominated_scores[k]:
                    continue

                diff = 0
                for j in range(num_criteria):
                    diff += max(ratings[idx, j] - ratings[k, j], 0)

                total_diff += diff

            ranks[idx] = total_diff


class ProfitGain(MultiCriteriaSort):
    """Proposed by 'Mario Garza-Fabre1, Gregorio Toscano Pulido1, and Carlos A. Coello Coello2,
        Ranking Methods for Many-Objective Optimization,
        A. Hernandez Aguirre et al. (Eds.): MICAI 2009, LNAI 5845, pp. 633–645, 2009. ,
        Springer-Verlag Berlin Heidelberg 2009'
        It calculates the max gain of this solution comparing with all other solutions.
    """

    def __init__(self, config):
        super().__init__(config)

    def cpu_sort_algorithm(self, criteria_ratings):
        """Calculate Global Detriment rank for each ratings
        Args:
            criteria_ratings: an array of criteria ratings
        Returns:
            an array of Global Detriment rank score
        """

        num_ratings = criteria_ratings.shape[0]

        # calculate difference matrix
        diff_matrix = np.zeros([num_ratings, num_ratings])
        for i, j in itertools.product(range(num_ratings), range(num_ratings)):
            # find difference for each criterion
            diff = criteria_ratings[i] - criteria_ratings[j]

            # set zero if negative
            diff[diff < 0] = 0
            diff_matrix[i, j] = diff.sum()

        # calculate rank
        gd_ranks = diff_matrix.max(axis=1) - diff_matrix.max(axis=0)

        return torch.tensor(gd_ranks)

    @staticmethod
    @cuda.jit
    def sort_cuda_kernel(ratings, num_items, ranks, dominated_scores):
        """CUDA kernel function to dominated sort
        Args:
            ratings: two dimension array with ratings for multiple users
            num_items: one dimension array with only one element
            ranks: output variable for rankings
            dominated_scores: used for local sort
        """
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        num_criteria = ratings.shape[1]

        for idx in range(start, ratings.shape[0], stride):
            dominated = 0
            # calculate the rating item range for each user
            start_id = math.floor(idx / num_items[0]) * num_items[0]
            end_id = start_id + num_items[0]

            # get dominated score at idx
            dominated_score_idx = dominated_scores[idx]

            max_gain_1 = 0
            max_gain_2 = 0
            for k in range(start_id, end_id):

                # only count the item that the dominated score is the same as current idx
                if dominated_score_idx != dominated_scores[k]:
                    continue

                diff_1 = 0
                diff_2 = 0
                for j in range(num_criteria):
                    diff_1 += max(ratings[idx, j] - ratings[k, j], 0)
                    diff_2 += max(ratings[k, j] - ratings[idx, j], 0)

                max_gain_1 = max(max_gain_1, diff_1)
                max_gain_2 = max(max_gain_2, diff_2)

            ranks[idx] = max_gain_1 - max_gain_2


class DistanceBestSolution(MultiCriteriaSort):
    """Proposed by 'Mario Garza-Fabre1, Gregorio Toscano Pulido1, and Carlos A. Coello Coello2,
        Ranking Methods for Many-Objective Optimization,
        A. Hernandez Aguirre et al. (Eds.): MICAI 2009, LNAI 5845, pp. 633–645, 2009. ,
        Springer-Verlag Berlin Heidelberg 2009'
        It uses the distance of this solution to the best solution, which is the max of each objective
    """

    def __init__(self, config):
        super().__init__(config)

    def cpu_sort_algorithm(self, criteria_ratings):
        """Calculate Global Detriment rank for each rating
        Args:
            criteria_ratings: an array of criteria ratings
        Returns:
            an array of Global Detriment rank score
        """

        # get the best solution
        best_solution = criteria_ratings.max(axis=0)

        # calculate distance to best solution
        distance = np.linalg.norm(criteria_ratings - best_solution, axis=1)

        return torch.tensor(distance)

    @staticmethod
    @cuda.jit
    def sort_cuda_kernel(ratings, num_items, ranks, best_solution, dominated_scores):
        """CUDA kernel function to dominated sort
        Args:
            ratings: two dimension array with ratings for multiple users
            num_items: one dimension array with only one element
            ranks: output variable for rankings
            best_solution: variable for best solution
            dominated_scores: used for local sort
        """
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        num_criteria = ratings.shape[1]

        for idx in range(start, ratings.shape[0], stride):
            # calculate the rating item range for each user
            start_id = math.floor(idx / num_items[0]) * num_items[0]
            end_id = start_id + num_items[0]
            sqr_sum = 0

            # get dominated score at idx
            dominated_score_idx = dominated_scores[idx]

            # get best solution
            for j in range(num_criteria):
                for i in range(start_id, end_id + 1):

                    # only count the item that the dominated score is the same as current idx
                    if dominated_score_idx != dominated_scores[i]:
                        continue

                    if best_solution[j] < ratings[i, j]:
                        best_solution[j] = ratings[i, j]

            # calculate distance
            for j in range(num_criteria):
                sqr_sum += (ratings[idx][j] - best_solution[j]) * (ratings[idx][j] - best_solution[j])

            ranks[idx] = math.sqrt(sqr_sum)

    def sort_gpu(self, ratings, num_items, dominated_scores=None):
        """Perform non dominated sort in GPU CUDA platform
        Args:
            ratings: two-dimensional array
            num_items: number of items per user id
            dominated_scores: used for sub sort with local_sort = True
        Return:
             one dimension array
        """
        if num_items is None:
            raise ValueError('Number of items is not assigned')

        ranks = np.zeros(ratings.shape[0])
        best_solution = np.zeros(ratings.shape[1])

        # load array to GPU memory
        ratings_gpu = cuda.to_device(ratings)
        num_items_gpu = cuda.to_device([num_items])
        ranks_gpu = cuda.to_device(ranks)
        best_solution_gpu = cuda.to_device(best_solution)

        if dominated_scores is not None:
            dominated_scores_gpu = cuda.to_device(dominated_scores)
        else:
            dominated_scores_gpu = cuda.to_device(np.zeros(ratings.shape[0]))

        # call CUDA kernel
        self.sort_cuda_kernel[2000, 256](ratings_gpu, num_items_gpu, ranks_gpu, best_solution_gpu, dominated_scores_gpu)

        # return the results from GPU memory
        return ranks_gpu.copy_to_host()


class LayeringRanking(MultiCriteriaSort):
    """Proposed by 'Yuan, Xu, Wang, Evolutionary Many-Objective Optimization Using Ensemble Fitness Ranking,
        GECCO'14, July 12-16, 2014, Vancouver, BC, Canada'
        It uses multiple fitness value function to sort through the whole solution set.
    """

    def __init__(self, config):
        super().__init__(config)

        self.fitness_functions_config = config['fitness_functions']

        if self.fitness_functions_config is None:
            raise ValueError('fitness_functions is not configured')

    def calculate_fitness_values(self, criteria_ratings, best_solution):
        """Calculate fitness values for each solution for all fitness functions
        Args:
            criteria_ratings: an array of criteria ratings
            best_solution: a one dimension array
        Return:
            an array of fitness values, dim = [criteria_ratings.shape[0], len(self.fitness_functions_config)]
        """
        fitness_value_all = np.zeros([criteria_ratings.shape[0], len(self.fitness_functions_config)])
        for k, func_config in enumerate(self.fitness_functions_config):
            func_name = func_config['func']
            fitness_func = getattr(self, func_name)
            params = func_config['params']
            fitness_value = fitness_func(criteria_ratings, best_solution, **params)
            fitness_value_all[:, k] = fitness_value

        return fitness_value_all

    def cpu_sort_algorithm(self, criteria_ratings):
        """Calculate Global Detriment rank for each rating
        Args:
            criteria_ratings: an array of criteria ratings
        Returns:
            an array of Global Detriment rank score
        """
        ranks = np.zeros(criteria_ratings.shape[0])

        # calculate fitness values
        best_solution = criteria_ratings.max(axis=0)
        fitness_value = self.calculate_fitness_values(criteria_ratings=criteria_ratings, best_solution=best_solution)

        fitness_value_tensor = torch.tensor(fitness_value).to(self.device)

        score = 1
        while fitness_value_tensor.min().item() > -1:
            for k in range(3):
                max_idx = torch.argmax(fitness_value_tensor[:, k])
                # assign ranking score
                ranks[max_idx] = score
                fitness_value_tensor[max_idx] = -1

            # update ranking score
            score += 1

        # reverse rank score, bigger is better
        ranking_score = np.max(ranks) - ranks

        return torch.tensor(ranking_score)

    def sort_gpu(self, ratings, num_items, dominated_scores=None):
        """Perform non dominated sort in GPU CUDA platform
        Args:
            ratings: two-dimensional array
            num_items: number of items per user id
            dominated_scores: used for sub sorting
        Return:
             one dimension array
        """
        if num_items is None:
            raise ValueError('Number of items is not assigned')

        ranks = np.zeros(ratings.shape[0])

        # calculate fitness value for each rating and for all fitness functions
        start_idx = 0
        end_idx = num_items - 1
        while end_idx < ratings.shape[0]:
            user_ratings = ratings[start_idx:end_idx + 1]
            best_solution = user_ratings.max(axis=0)
            fitness_value = self.calculate_fitness_values(criteria_ratings=user_ratings, best_solution=best_solution)

            # utilize GPU for tensor operation if possible
            fitness_value_tensor = torch.tensor(fitness_value).to(self.device)

            score = 1
            while fitness_value_tensor.min().item() > -1:
                for k in range(3):
                    max_idx = torch.argmax(fitness_value_tensor[:, k])
                    # assign ranking score
                    ranks[start_idx + max_idx] = score
                    fitness_value_tensor[max_idx] = -1

                # update ranking score
                score += 1

            start_idx = end_idx + 1
            end_idx += num_items

        # reverse rank score, bigger is better
        ranking_score = np.max(ranks) - ranks

        # return the results from GPU memory
        return ranking_score

    def sort(self, interaction, criteria_ratings, overall_ratings, dominated_scores=None):
        """A standard interface to perform the sort.
        Args:
            interaction: a tensor of user item pair
            criteria_ratings: tensor of criteria ratings for each (user, item) pair
            overall_ratings: tensor of overall rating for each (user, item) pair
            dominated_scores: the Pareto dominated score, used only for sub sort
        Return:
            a tensor object with ranking score for each (user, item) pair
        """
        if len(criteria_ratings) != len(interaction):
            raise ValueError(f'Number of criteria ratings ({len(criteria_ratings)}) is not the same as number of user '
                             f'item pairs ({len(interaction)})')

        if self.sorting_label:
            sorting_ratings = criteria_ratings[:, self.sorting_label_index]
        else:
            sorting_ratings = criteria_ratings

        num_of_items = interaction.interaction[self.ITEM_ID].unique().shape[0]
        num_of_users_items = interaction.interaction[self.USER_ID].shape[0]

        if num_of_users_items % num_of_items != 0:
            raise ValueError(f'User item rating is not complete, cannot use this batch sort on GPU ')

        # sort ratings with GPU
        dominated_score = self.sort_gpu(sorting_ratings.cpu().numpy(), num_of_items, dominated_scores)
        # dominated_score = self.sort_parallel(interaction, criteria_ratings)

        return torch.tensor(dominated_score)


class PreferenceOrderRanking(MultiCriteriaSort):
    """Implement the Preference Order Ranking, proposed by:
    di Pierro, Khu, Savic, An Investigation on Preference Order -- Ranking Scheme for Multi Objective Evolutionary
    Optimization, IEEE Transactions on Evolutionary Computation, 11(1), 17-45 (2007)

    """

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def is_efficient(idx, ratings, k):
        """Determine if ratings[idx] is k-efficient
        Args:
            idx: index of rating to be evaluated
            ratings: list of criteria ratings
            k: the order of efficiency, k in [1, num_of_criteria]
        """
        is_efficient = False
        current_rating = ratings[:, 0:k][idx]
        diff = current_rating - ratings[:, 0:k]
        if diff.ge(0).all() and diff.ge(1).any():
            is_efficient = True

        return is_efficient

    @staticmethod
    @cuda.jit
    def sort_cuda_kernel(ratings, num_items, ranks):
        """CUDA kernel function to dominated sort
        Args:
            ratings: two dimension array with ratings for multiple users
            num_items: one dimension array with only one element
            ranks: output variable for rankings
        """
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        num_criteria = ratings.shape[1]

        for idx in range(start, ratings.shape[0], stride):
            # calculate the rating item range for each user
            start_id = math.floor(idx / num_items[0]) * num_items[0]
            end_id = start_id + num_items[0]

            for k in range(num_criteria):
                for i in range(start_id, end_id + 1):
                    break

    def cpu_sort_algorithm(self, criteria_ratings):
        """Calculate preference order ranking score for items of an user
        """

        num_of_criteria = criteria_ratings.shape[1]
        num_of_items = criteria_ratings.shape[0]

        ranks = np.zeros(num_of_items)
        for i in range(num_of_items):
            for k in range(num_of_criteria):
                if self.is_efficient(i, criteria_ratings, k):
                    ranks[i] = num_of_criteria - k
                    break
        return ranks

    def sort_gpu(self, ratings, num_items, dominated_scores=None):
        """Perform non dominated sort in GPU CUDA platform
        Args:
            ratings: two-dimensional array
            num_items: number of items per user id
            dominated_scores: the Pareto dominated score, used only for sub sort
        Return:
             one dimension array
        """
        if num_items is None:
            raise ValueError('Number of items is not assigned')

        ranks = np.zeros(ratings.shape[0])

        # calculate fitness value for each rating and for all fitness functions
        start_idx = 0
        end_idx = num_items - 1
        while end_idx < ratings.shape[0]:
            user_ratings = ratings[start_idx:end_idx + 1]
            user_ranks = self.cpu_sort_algorithm(user_ratings)
            ranks[start_idx:end_idx + 1] = user_ranks

            start_idx = end_idx + 1
            end_idx += num_items

        # return the results from GPU memory
        return ranks

    def sort(self, interaction, criteria_ratings, overall_ratings, dominated_scores=None):
        """A standard interface to perform the sort.
        Args:
            interaction: a tensor of user item pair
            criteria_ratings: tensor of criteria ratings for each (user, item) pair
            overall_ratings: tensor of overall rating for each (user, item) pair
            dominated_scores: the Pareto dominated score, used only for sub sort
        Return:
            a tensor object with ranking score for each (user, item) pair
        """
        if len(criteria_ratings) != len(interaction):
            raise ValueError(f'Number of criteria ratings ({len(criteria_ratings)}) is not the same as number of user '
                             f'item pairs ({len(interaction)})')

        if self.sorting_label:
            sorting_ratings = criteria_ratings[:, self.sorting_label_index]
        else:
            sorting_ratings = criteria_ratings

        num_of_items = interaction.interaction[self.ITEM_ID].unique().shape[0]
        num_of_users_items = interaction.interaction[self.USER_ID].shape[0]

        if num_of_users_items % num_of_items != 0:
            raise ValueError(f'User item rating is not complete, cannot use this batch sort on GPU ')

        # sort ratings with GPU
        dominated_score = self.sort_gpu(sorting_ratings, num_of_items)

        return torch.tensor(dominated_score)


class KDominance(MultiCriteriaSort):
    """Reference:
    M. Farina and P. Amato, A Fuzzy Definition of "Optimality" for Many-Criteria Optimization Problems,
    IEEE Transactions on Systems, Man, and Cybernetics, Part A: Systems and Humans, Vol 34, No 3. May 2004
    Main idea: apply dominance relations allowing some inferior objectives
    """

    def __init__(self, config):

        super().__init__(config)

    def cpu_sort_algorithm(self, criteria_ratings):
        """Calculate Global Detriment rank for each ratings
        Args:
            criteria_ratings: an array of criteria ratings
        Returns:
            an array of ranking score based on k-optimality
        """

        num_ratings = criteria_ratings.shape[0]
        num_criteria = criteria_ratings.shape[1]

        # calculate difference matrix
        dominance_matrix = np.zeros([num_ratings, num_ratings])
        for i, j in itertools.product(range(num_ratings), range(num_ratings)):
            # find difference for each criterion
            diff = criteria_ratings[i] - criteria_ratings[j]

            # find number of criteria for better, equal, and worse
            n_better = len(diff[diff > 0])
            n_equal = len(diff[diff == 0])

            if n_equal < num_criteria and n_better >= (num_criteria - n_equal) / (self.config['dominance_k'] + 1):
                dominance_matrix[i, j] = 1

        # calculate rank
        k_ranks = dominance_matrix.sum(axis=1)

        return torch.tensor(k_ranks)

    @staticmethod
    @cuda.jit
    def sort_cuda_kernel(ratings, parameters, ranks, dominated_scores):
        """CUDA kernel function to dominated sort
        Args:
            ratings: two dimension array with ratings for multiple users
            parameters: one dimension array with parameter values
            ranks: output variable for rankings
            dominated_scores: item dominated scores, used to sub sorting
        """
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        num_criteria = ratings.shape[1]

        num_items = parameters[0]
        dominance_k = parameters[1]

        for idx in range(start, ratings.shape[0], stride):
            # calculate the rating item range for each user
            start_id = math.floor(idx / num_items) * num_items
            end_id = start_id + num_items

            # get dominated score at idx
            dominated_score_idx = dominated_scores[idx]

            k_dominance_count = 0
            for k in range(start_id, end_id):

                # only count the item that the dominated score is the same as current idx
                if dominated_score_idx != dominated_scores[k]:
                    continue

                n_better = 0
                n_equal = 0
                for j in range(num_criteria):
                    n_better += 1 if (ratings[idx, j] - ratings[k, j]) > 0 else 0
                    n_equal += 1 if (ratings[idx, j] - ratings[k, j]) == 0 else 0

                if n_equal < num_criteria and n_better >= (num_criteria - n_equal) / (dominance_k + 1):
                    k_dominance_count += 1

            ranks[idx] = k_dominance_count

    def sort_gpu(self, ratings, num_items, dominated_scores=None):
        """Perform non dominated sort in GPU CUDA platform. If the parameter is different, the subclass must implement
        its own method
        Args:
            ratings: two-dimensional array
            num_items: number of items per user id
            dominated_scores: item dominated scores, used to sub sorting
        Return:
             one dimension array
        """
        if num_items is None:
            raise ValueError('Number of items is not assigned')

        ranks = np.zeros(ratings.shape[0])

        # load array to GPU memory
        ratings_gpu = cuda.to_device(ratings)
        parameters_gpu = cuda.to_device([num_items, self.config['dominance_k']])
        ranks_gpu = cuda.to_device(ranks)
        if dominated_scores is not None:
            dominated_scores_gpu = cuda.to_device(dominated_scores)
        else:
            dominated_scores_gpu = cuda.to_device(np.zeros(ratings.shape[0]))

        # call CUDA kernel
        self.sort_cuda_kernel[2000, 256](ratings_gpu, parameters_gpu, ranks_gpu, dominated_scores_gpu)

        # return the results from GPU memory
        return ranks_gpu.copy_to_host()


class EpsilonDominance(MultiCriteriaSort):
    """Reference:
    M. Laumanns, L. Thiele, K. Deb, E. Zitzler, Combining Convergence and Diversity in Evolutionary Multi-Objective
    Optimization, Evolutionary Computation 10(3), 2002 by MIT
    Main idea: apply (epsilon + 1) f_i >= f_j
    """

    def __init__(self, config):

        super().__init__(config)

    def cpu_sort_algorithm(self, criteria_ratings):
        """Calculate Global Detriment rank for each ratings
        Args:
            criteria_ratings: an array of criteria ratings
        Returns:
            an array of ranking score based on epsilon dominance
        """

        num_ratings = criteria_ratings.shape[0]
        num_criteria = criteria_ratings.shape[1]

        # calculate difference matrix
        dominance_matrix = np.zeros([num_ratings, num_ratings])
        for i, j in itertools.product(range(num_ratings), range(num_ratings)):
            # find difference for each criterion
            diff = (self.config['epsilon'] + 1) * criteria_ratings[i] - criteria_ratings[j]

            # check if meet the epsilon dominance relation
            if diff.min() >= 0:
                dominance_matrix[i, j] = 1

        # calculate rank
        k_ranks = dominance_matrix.sum(axis=1)

        return torch.tensor(k_ranks)

    @staticmethod
    @cuda.jit
    def sort_cuda_kernel(ratings, parameters, ranks, dominated_scores):
        """CUDA kernel function to dominated sort
        Args:
            ratings: two dimension array with ratings for multiple users
            parameters: one dimension array with parameter values
            ranks: output variable for rankings
            dominated_scores: the Pareto dominated score, used only for sub sort
        """
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        num_criteria = ratings.shape[1]

        num_items = parameters[0]
        epsilon = parameters[1]

        for idx in range(start, ratings.shape[0], stride):
            # calculate the rating item range for each user
            start_id = math.floor(idx / num_items) * num_items
            end_id = start_id + num_items

            # get dominated score at idx
            dominated_score_idx = dominated_scores[idx]

            k_dominance_count = 0
            for k in range(start_id, end_id):

                # only count the item that the dominated score is the same as current idx
                if dominated_score_idx != dominated_scores[k]:
                    continue

                n_better = 0
                for j in range(num_criteria):
                    n_better += 1 if ((epsilon + 1) * ratings[idx, j] - ratings[k, j]) >= 0 else 0

                if n_better == num_criteria:
                    k_dominance_count += 1

            ranks[idx] = k_dominance_count

    def sort_gpu(self, ratings, num_items, dominated_scores=None):
        """Perform non dominated sort in GPU CUDA platform. If the parameter is different, the subclass must implement
        its own method
        Args:
            ratings: two-dimensional array
            num_items: number of items per user id
            dominated_scores: the Pareto dominated score, used only for sub sort
        Return:
             one dimension array
        """
        if num_items is None:
            raise ValueError('Number of items is not assigned')

        ranks = np.zeros(ratings.shape[0])

        # load array to GPU memory
        ratings_gpu = cuda.to_device(ratings)
        parameters_gpu = cuda.to_device([num_items, self.config['epsilon']])
        ranks_gpu = cuda.to_device(ranks)
        if dominated_scores is not None:
            dominated_scores_gpu = cuda.to_device(dominated_scores)
        else:
            dominated_scores_gpu = cuda.to_device(np.zeros(ratings.shape[0]))

        # call CUDA kernel
        self.sort_cuda_kernel[2000, 256](ratings_gpu, parameters_gpu, ranks_gpu, dominated_scores_gpu)

        # return the results from GPU memory
        return ranks_gpu.copy_to_host()
