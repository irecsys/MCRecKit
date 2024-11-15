# @Time : 2024/10/24
# @Author : David Wang, Yong Zheng, Qin Ruan


from recbole.utils import set_color
from mcreckit.data.dataset.mcdataset import MCDataset

class MultiCriteriaDataset(MCDataset):
    """ data set class for multi criteria user-item interaction data

    """

    def __init__(self, config):
        super(MultiCriteriaDataset, self).__init__(config)

    def _get_field_from_config(self):
        """Initialization common field names.
        """
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.multi_label_field = self.config['MULTI_LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']
        self.label_field = self.config['LABEL_FIELD']

        if (self.uid_field is None) ^ (self.iid_field is None):
            raise ValueError(
                'USER_ID_FIELD and ITEM_ID_FIELD need to be set at the same time or not set at the same time.'
            )

        self.logger.debug(set_color('uid_field', 'blue') + f': {self.uid_field}')
        self.logger.debug(set_color('iid_field', 'blue') + f': {self.iid_field}')
