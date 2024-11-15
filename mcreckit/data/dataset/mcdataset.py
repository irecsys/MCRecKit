# @Time : 2024/10/24
# @Author : David Wang, Qin Ruan


import numpy as np
from sklearn.model_selection import KFold
from recbole.data.dataset import Dataset
from recbole.utils import FeatureSource, FeatureType

class MCDataset(Dataset):
    def __init__(self, config):
        super(MCDataset, self).__init__(config)

    def _get_innderid_from_rawid(self, field, rawid):
        innerid = -1
        dict = self.field2token_id[field]
        if rawid in dict:
            innerid = dict[rawid]
        return innerid

    def _get_rawid_from_innerid(self, field, innerid):
        rawid = -1
        dict = self.field2token_id[field]
        if innerid in dict.values():
            rawid = list(dict.keys())[list(dict.values()).index(innerid)]
        return rawid

    def _set_label_by_threshold(self):
        """Generate 0/1 labels according to value of features.

        According to ``config['threshold']``, those rows with value lower than threshold will
        be given negative label, while the other will be given positive label.
        See :doc:`../user_guide/data/data_args` for detail arg setting.

        Note:
            Key of ``config['threshold']`` if a field name.
            This field will be dropped after label generation.
        """
        threshold = self.config['threshold']
        if threshold is None:
            return

        self.logger.debug(f'Set label by {threshold}.')

        if len(threshold) != 1:
            raise ValueError('Threshold length should be 1.')

        self.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
        for field, value in threshold.items():
            if field in self.inter_feat:  # if LABEL_FIELD is the same as in threshold dic key, do not convert data
                if self.label_field != field and self.label_field is not None:
                    self.inter_feat[self.label_field] = (self.inter_feat[field] >= value).astype(int)
                    # David Wang: if label_field == field in threshold setting , it will delete the label_field from data
                    self._del_col(self.inter_feat, field)
            else:
                raise ValueError(f'Field [{field}] not in inter_feat.')

    def unique(self, field):
        values = np.unique(self.field2id_token[field])
        return values

    def split_by_folds(self, folds, group_by=None):
        """Split interaction records by N fold
        Args:
            folds (int): Number of folds
            target: used by StratifiedKFold for data distribution in each fold
            group_by (str, optional): Field name that interaction records should grouped by before splitting.
                Defaults to ``None``, only support 'user' and None
        Returns:
            list of data set tuple of train data and evaluation data for each fold
        """
        self.logger.debug(f'split by folds [{folds}], group_by=[{group_by}]')

        tot_cnt = self.__len__()
        tot_index = np.arange(0, tot_cnt, 1).reshape(-1)
        target = self.inter_feat[self.label_field]

        # create fold object
        skf = KFold(n_splits=folds, random_state=233, shuffle=True)

        if group_by is None:
            folds_index = skf.split(tot_index, target)
        else:
            folds_index = skf.split(tot_index, target, self.inter_feat[group_by])

        # create fold index list
        folds_index_list = []
        for train_index, test_index in folds_index:
            folds_index_list.append([train_index, test_index])

        self._drop_unused_col()
        ds_split_list = []
        for fold in folds_index_list:
            next_df = [self.inter_feat[fold[0]], self.inter_feat[fold[1]]]
            # need to add empty testing data set (the third in the list) for consistency required by sampler creation
            next_ds = [self.copy(_) for _ in next_df] + [self.copy(self.inter_feat[[]])]
            ds_split_list.append(next_ds)

        return ds_split_list

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        # set feature format
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [self.copy(self.inter_feat[start:end]) for start, end in zip([0] + cumsum[:-1], cumsum)]
            return datasets

        # ordering
        ordering_args = self.config['eval_args']['order']
        if ordering_args == 'RO':  # David Wang: random order
            self.shuffle()
        elif ordering_args == 'TO':  # David Wang: time order
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(f'The ordering_method [{ordering_args}] has not been implemented.')

        # splitting & grouping
        split_args = self.config['eval_args']['split']  # David Wang: split_args = {'RS': [0.8, 0.1, 0.1]}
        if split_args is None:
            raise ValueError('The split_args in eval_args should not be None.')
        if not isinstance(split_args, dict):
            raise ValueError(f'The split_args [{split_args}] should be a dict.')

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) >= 1
        group_by = self.config['eval_args']['group_by']
        if split_mode == 'RS':  # David Wang: random split
            if not isinstance(split_args['RS'], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == 'none':
                datasets = self.split_by_ratio(split_args['RS'], group_by=None)
            elif group_by == 'user':
                datasets = self.split_by_ratio(split_args['RS'], group_by=self.uid_field)
            else:
                raise NotImplementedError(f'The grouping method [{group_by}] has not been implemented.')
        elif split_mode == 'LS':
            datasets = self.leave_one_out(group_by=self.uid_field, leave_one_mode=split_args['LS'])
        elif split_mode == 'CV':
            # return a dictionary. Key is fold number, value is a list of training and validation sets
            if group_by is None or group_by.lower() == 'none':
                datasets = self.split_by_folds(split_args['CV'], group_by=None)
            elif group_by == 'user':
                datasets = self.split_by_folds(split_args['CV'], group_by=self.uid_field)
            else:
                raise NotImplementedError(f'The grouping method [{group_by}] has not been implemented.')
        else:
            raise NotImplementedError(f'The splitting_method [{split_mode}] has not been implemented.')

        return datasets
