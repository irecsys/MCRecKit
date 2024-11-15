# @Time : 2024/10/24
# @Author : David Wang, Yong Zheng, Qin Ruan


import numpy as np
from recbole.sampler import Sampler


class MCSampler(Sampler):
    """MCSampler with overridden functions
    """

    def __init__(self, phases, datasets, distribution='uniform', neg_sampling=None):
        self.neg_sampling = neg_sampling

        super(MCSampler, self).__init__(phases, datasets, distribution)

    def get_used_ids(self):
        """
        Returns:
            dict: Used item_ids is the same as positive item_ids.
            Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
        """
        # this dictionary has used item id for each user in three dataset: training, evaluation and teting
        used_item_id = dict()

        # for each user, set up an empty set
        # DW: used_item_id['valid'] = used_item_id['train'] and {used id in valid dataset}
        # DW: used_item_id['test'] = used_item_id['valid'] and {used id in testing dataset}
        last = [set() for _ in range(self.user_num)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            # for each user_id, find used item id
            for uid, iid in zip(dataset.inter_feat[self.uid_field].numpy(), dataset.inter_feat[self.iid_field].numpy()):
                cur[uid].add(iid)
            last = used_item_id[phase] = cur

        # check if testing set has all the items for some user
        if self.neg_sampling:
            for used_item_set in used_item_id[self.phases[-1]]:
                if len(used_item_set) + 1 == self.item_num:  # [pad] is a item.
                    raise ValueError(
                        'Some users have interacted with all items, '
                        'which we can not sample negative items for them. '
                        'Please set `user_inter_num_interval` to filter those users.'
                    )

        return used_item_id
