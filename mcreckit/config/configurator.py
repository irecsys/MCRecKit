# @Time : 2024/10/24
# @Author : Qin Ruan, Yong Zheng

import os

from recbole.config import Config
from recbole.utils import set_color
from mcreckit.utils import (get_model, general_arguments, training_arguments,
                            evaluation_arguments, dataset_arguments, mcranking_arguments)


class MCConfig(Config):
    """MCConfig with overridden functions
    """

    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        super(MCConfig, self).__init__(model, dataset, config_file_list, config_dict)
        if config_file_list:
            self.config_file_name = config_file_list[0]
        # David Wang: add config from general model file
        if 'GENERAL_MODEL' in self.variable_config_dict:
            self._general_model_config_dict(self.variable_config_dict['GENERAL_MODEL'])

    def _init_parameters_category(self):
        """Output arguments to logs by categories
        """
        self.parameters = dict()
        self.parameters['General'] = general_arguments
        self.parameters['Training'] = training_arguments
        self.parameters['Evaluation'] = evaluation_arguments
        self.parameters['Dataset'] = dataset_arguments
        self.parameters['Multi-Criteria Ranking'] = mcranking_arguments

    def _get_model_and_dataset(self, model, dataset):
        """Retrieve model, model class and final data set
        """
        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] '
                )
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[dataset variable, config file, config dict, command line] '
                )
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset

    def _set_train_neg_sample_args(self):
        """Set negative samples for training
        """
        neg_sampling = self.final_config_dict['neg_sampling']
        if neg_sampling is None:  # this is for labeled data (explicit rating)
            self.final_config_dict['train_neg_sample_args'] = {'strategy': 'none'}
        else:
            if not isinstance(neg_sampling, dict):
                raise ValueError(f"neg_sampling:[{neg_sampling}] should be a dict.")
            if len(neg_sampling) > 1:
                raise ValueError(f"the len of neg_sampling [{neg_sampling}] should be 1.")

            distribution = list(neg_sampling.keys())[0]
            sample_num = neg_sampling[distribution]
            if distribution not in ['uniform', 'popularity']:
                raise ValueError(f"The distribution [{distribution}] of neg_sampling "
                                 f"should in ['uniform', 'popularity']")

            self.final_config_dict['train_neg_sample_args'] = {
                'strategy': 'by',
                'by': sample_num,
                'distribution': distribution
            }

    def __str__(self):
        """Output parameters in config to logs
        """
        args_info = '\n'
        args_info += f"model = {self.__getattr__('model')} \n"
        args_info += f"dataset = {self.__getattr__('dataset')} \n\n"

        for category in self.parameters:
            args_info += set_color(category + ' Hyper Parameters:\n', 'pink')
            args_info += '\n'.join([(set_color("{}", 'cyan') + " =" + set_color(" {}", 'yellow')).format(arg, value)
                                    for arg, value in self.final_config_dict.items()
                                    if arg in self.parameters[category]])
            args_info += '\n\n'

        args_info += set_color('Other Hyper Parameters: \n', 'pink')
        args_info += '\n'.join([
            (set_color("{}", 'cyan') + " = " + set_color("{}", 'yellow')).format(arg, value)
            for arg, value in self.final_config_dict.items()
            if arg not in {
                _ for args in self.parameters.values() for _ in args
            }.union({'model', 'dataset', 'config_files'})
        ])
        args_info += '\n\n'
        return args_info

    def _general_model_config_dict(self, general_model_name):
        """Load .yaml file for general model
        Args:
            general_model_name: model name
        Returns:
        """
        current_path = os.path.dirname(os.path.realpath(__file__))
        model_yaml_file = os.path.join(current_path, '../properties/model/' + general_model_name + '.yaml')

        if os.path.isfile(model_yaml_file):
            config_dict = self._update_internal_config_dict(model_yaml_file)
        else:
            config_dict = {}

        return config_dict
