# @Time : 2024/10/24
# @Author : David Wang, Yong Zheng, Qin Ruan

import numpy as np
import torch
from recbole.data import FullSortEvalDataLoader, Interaction


class LabeledRankingEvalDataLoader(FullSortEvalDataLoader):
    """EvalDataLoader for ranking evaluations in MCRS
    """
    def __init__(self, config, dataset, sampler, shuffle=False):

        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.label_field = config['RATING_FIELD']
        self.threshold = config['threshold'][self.label_field]

        # overwrite the assignment for self.uid2positive_item[uid], self.uid2items_num[uid], self.uid2history_item[uid]
        # with predefined threshold of label value as positive or negative label
        if not self.is_sequential:
            user_num = dataset.user_num
            self.uid_list = []
            self.uid2items_num = np.zeros(user_num, dtype=np.int64)  # DW: number of items for each user
            self.uid2positive_item = np.array([None] * user_num)
            self.uid2history_item = np.array([None] * user_num)

            dataset.sort(by=self.uid_field, ascending=True)
            last_uid = None
            positive_item = set()
            negative_item = set()

            # DW: for each user, get used item in sampler: used item in both training and valid dataset
            uid2used_item = sampler.used_ids

            # DW: iterate each pair of user id and item id in the dataset
            for uid, iid, rating in zip(dataset.inter_feat[self.uid_field].numpy(),
                                        dataset.inter_feat[self.iid_field].numpy(),
                                        dataset.inter_feat[self.label_field].numpy()):
                # for each user id, create a related positive item list based on used item for this user
                if uid != last_uid:
                    self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
                    self._remove_negative_item(last_uid, negative_item)
                    last_uid = uid
                    self.uid_list.append(uid)
                    positive_item = set()
                    negative_item = set()
                # only add item id with positive rating
                if rating > self.threshold:
                    positive_item.add(iid)  # add item id from valid/testing data set
                else:
                    negative_item.add(iid)
            # for last user id
            self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
            self._remove_negative_item(last_uid, negative_item)

            # find user id without positive or history items, and remove them from user id list
            uid_remove = []
            for uid in self.uid_list:
                if self.uid2positive_item[uid] is None or self.uid2history_item[uid] is None or \
                        len(self.uid2positive_item[uid]) == 0 or len(self.uid2history_item[uid]) == 0:
                    uid_remove.append(uid)
            self.uid_list = list(set(self.uid_list) - set(uid_remove))

            self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
            self.user_df = dataset.join(Interaction({self.uid_field: self.uid_list}))

    def _remove_negative_item(self, uid, negative_item):
        """Remove negative items (rating < threshold) for uid in evaluation set from  self.uid2history_item
        Args:
            uid: user id
            negative_item: item id that uid rating < threshold
        Returns:
            None
        """
        if not self.uid2history_item[uid] is None and len(negative_item) > 0:
            self.uid2history_item[uid] = torch.tensor(list(set(self.uid2history_item[uid].numpy()) - negative_item))
