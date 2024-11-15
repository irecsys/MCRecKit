# @Time   : 2022/01/19
# @Author : David Wang


import copy
import os
import re
from collections import defaultdict
from logging import getLogger
from multiprocessing.pool import ThreadPool
from time import time
import numpy as np
import torch
from recbole.utils import init_seed
from recbole.data import FullSortEvalDataLoader, Interaction
from recbole.evaluator import Collector, Evaluator
from recbole.trainer import AbstractTrainer
from recbole.utils import EvaluatorType, set_color, get_gpu_usage, get_tensorboard, ensure_dir, get_local_time, \
    calculate_valid_score, early_stopping, dict2str
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from mcreckit.config.configurator import MCConfig
from mcreckit.data.utils import create_samplers, get_dataloader
from mcreckit.utils import CustomColumn, get_trainer, MCModelType


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
    resume_checkpoint() and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model, built_datasets=None):
        super(Trainer, self).__init__(config, model)
        self.built_datasets = built_datasets

        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(self.config['model'], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config['weight_decay']

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.best_model_state = {}  # added by David Wang
        self.train_loss_dict = dict()
        if not {'sub_models', 'criteria_model'}.intersection(set(config.final_config_dict)):
            self.optimizer = self._build_optimizer(self.model.parameters())
        self.eval_type = config['eval_type']
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_tensor = None
        self.tot_item_num = None

    def _build_optimizer(self, params):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.config['reg_weight'] and self.weight_decay and self.weight_decay * self.config['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=self.learning_rate)
            if self.weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        # David Wang: set the model to training mode: mode=True
        self.model.train()  # David Wang: this train() method is the built in method of torch.nn model
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(  # David Wang: tqdm is a package that show the progress of iteration
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()  # David Wang: set all tensor gradient to zero
            losses = loss_func(interaction)  # David Wang: calculate loss
            if isinstance(losses, tuple):  # David Wang: if there are multiple losses, take a simple sum
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward(retain_graph=True)  # David Wang: save the gradient
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()  # David Wang: update model parameters
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

        return total_loss

    def _valid_epoch(self, valid_data, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result, _ = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'other_parameter': self.model.other_parameter(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file, _use_new_zipfile_serialization=True)

        self.best_model_state = copy.deepcopy(state)

    def resume_checkpoint(self, resume_file):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        self.saved_model_file = resume_file
        checkpoint = torch.load(resume_file)
        self.start_epoch = checkpoint['epoch'] + 1
        self.cur_step = checkpoint['cur_step']
        self.best_valid_score = checkpoint['best_valid_score']

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.load_other_parameter(checkpoint.get('other_parameter'))

        # load optimizer state from checkpoint only when optimizer type is not changed
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses
        return train_loss_output + ']'

    def _add_train_loss_to_tensorboard(self, epoch_idx, losses, tag='Loss/Train'):
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, losses, epoch_idx)

    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'learner': self.config['learner'],
            'learning_rate': self.config['learning_rate'],
            'train_batch_size': self.config['train_batch_size']
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values() for parameter in parameters
        }.union({'model', 'dataset', 'config_files', 'device'})
        # other model-specific hparam
        hparam_dict.update({
            para: val
            for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """

        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1)

        self.eval_collector.data_collect(train_data)

        # print('\nStart epoch training ...')
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)

            if self.config['print_loss']:
                print(f'loss for epoch #{epoch_idx}: {train_loss}')

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            # eval
            # David Wang: if config['eval_step'] is 0 or no validation data, no validation is performed. The model
            # training will go through each epoch
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx)
                    update_output = set_color('Saving current', 'blue') + ': %s' % self.saved_model_file
                    if verbose:
                        self.logger.info(update_output)
                continue

            """ David Wang: the training will be evaluated by valid_score based on config['valid_metric']
                for each evaluation (defined by config['eval_step']):
                    1) if valid_score is getting better, update best_valid_score, and set update_flag=True and save 
                        model parameters to file if save=True
                    2) if valid_score is not getting better, and number of such epoch >= config['stopping_step'], stop
                        training
            """
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)

                if show_progress:
                    print(valid_score, valid_result)

                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )

                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                        if verbose:
                            self.logger.info(update_output)
                    else:
                        self.best_model_state['epoch'] = epoch_idx
                        self.best_model_state['state_dict'] = self.model.state_dict()
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

        # print('Epoch training cycle finished\n')
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_result

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        # store history info in dict, key = userid, value = a list of rated items
        history = defaultdict(list)
        row_idx, col_idx = history_index
        for user, item in zip(row_idx, col_idx):
            history[user].append(item)
        self.model.setHistory(history)

        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        # set score to -np.inf for item index = 0 and user items in history, not in current validation dataset
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            row_idx, col_idx = history_index
            scores[row_idx.long(), col_idx.long()] = -np.inf
        return interaction, scores, positive_u, positive_i

    def _neg_sample_batch_eval(self, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        # interaction: contain both positive and negative data
        # row_idx: user id index for interaction data, same as in positive_u
        # positive_u: positive user id
        # positive_i: positive item id
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = self.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(interaction, batch_size)

        if self.config['eval_type'] == EvaluatorType.VALUE:
            return interaction, origin_scores, positive_u, positive_i
        elif self.config['eval_type'] == EvaluatorType.RANKING:
            # if evaluation type is ranking, positive_u in data loader must be created
            col_idx = interaction[self.config['ITEM_ID_FIELD']]
            batch_user_num = positive_u[-1] + 1  # DW: max possible number of positive users
            scores = torch.full((batch_user_num, self.tot_item_num), -np.inf, device=self.device)
            # scores[row_idx, col_idx] = origin_scores
            scores[row_idx.long(), col_idx.long()] = origin_scores  # DW: assign predicted score to each user item pair
            return interaction, scores, positive_u, positive_i

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file  # David Wang: use saved the model of the trainer object

            check_file = True
            import time
            while check_file:
                try:
                    checkpoint = torch.load(checkpoint_file)  # load trained model
                    check_file = False
                except:
                    import time
                    time.sleep(3)

            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                # get item feature name such as: movie_id
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num  # David Wang: get total item number in evaluation data

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )

        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)

        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()  # DW: recbole.evaluator.collector.DataStruct type
        result = self.evaluator.evaluate(struct)

        if 'epoch' in self.best_model_state:
            best_epoch_id = self.best_model_state['epoch']
        else:
            best_epoch_id = 0

        return result, best_epoch_id

    def _spilt_predict(self, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = self.model.predict(Interaction(current_interaction).to(self.device))
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)


class MultiCriteriaTrainer(Trainer):
    r"""MultiCriteriaTrainer is designed for multi criteria models, which set the epoch to 1 whatever the config.

    """

    def __init__(self, config, model, built_datasets=None, fold_id=None, caller=None):
        super(MultiCriteriaTrainer, self).__init__(config, model, built_datasets)

        self.label_fields = config['MULTI_LABEL_FIELD']
        self.criteria_vector_name = CustomColumn.CRITERIA_VECTOR.name
        self.predicted_label_name = CustomColumn.PREDICTED_LABEL.name
        self.criteria_results = None
        self.fold_id = fold_id
        self.caller = caller

        self.max_rating_value = config['RATING_RANGE'][1]
        self.min_rating_value = config['RATING_RANGE'][0]

    def construct_criteria_tensor(self, data_loader):
        """This is to construct a tenor which contain criteria rating vector
        Returns:
            a tensor object
        """

        # collect criteria rating tensor
        rating_tensor_list = ()
        for label in self.label_fields:
            rating_tensor_list = rating_tensor_list + (data_loader.dataset.inter_feat.interaction[label],)

        return torch.stack(rating_tensor_list, dim=1)

    def construct_saved_model_file(self, target=None, post_fix=None):
        """Update self.saved_model_file name for different model training/evaluation instance.
        the model training/evaluation instance is created by multi fold evaluation, sub model or training with
        different label
        Args:
            target: target model name, used for non ModelType.MULTICRITERIA model
            post_fix: test added to the end of the file name
        Returns:

        """
        # get current model name
        caller_name = self.config['model']
        if self.caller:
            caller_name = self.caller + '-' + caller_name

        if target:
            caller_name = caller_name + '-' + target

        # split default saved_model_file created from parent Trainer class
        split_name = re.split(r'\\|\.|-', self.saved_model_file)

        if self.fold_id is not None:
            saved_model_file = \
                f"{split_name[0]}\\{caller_name}-{'-'.join(split_name[2:7])}-fold_{self.fold_id}.{split_name[-1]}"
        else:
            saved_model_file = f"{split_name[0]}\\{caller_name}-{'-'.join(split_name[2:7])}.{split_name[-1]}"

        if post_fix:
            saved_model_file = saved_model_file + '_' + post_fix

        return saved_model_file


class CriteriaRPTrainer(MultiCriteriaTrainer):
    r"""MultiCriteriaTrainer is designed for multi criteria models, which set the epoch to 1 whatever the config.

    """

    def __init__(self, config, model, built_datasets=None, fold_id=None, caller=None):
        super(CriteriaRPTrainer, self).__init__(config, model, built_datasets, fold_id, caller)

        # self.general_model = model.general_model
        self.model = model
        self.config = config

        # create general model trainer
        self.general_trainer = {}

        self.trained_model_weights = {}
        self.best_model_state['epoch'] = {}

        self.general_model = {}
        for label in self.label_fields:
            self.general_model[label] = copy.deepcopy(model.general_model)
            self.general_model[label].LABLE = label

    @staticmethod
    def process_fit(args):
        """This is a wrapper for fit() method of general model, used in multiprocessing mode
        Args:
            args: a dictionary object with key:
                'train_data', 'valid_data', 'verbose', 'saved', 'show_progress'
        Returns:
            best_valid_result
        """
        general_model = args['general_model']
        general_model_type = args['model_type']
        general_model_name = args['model_name']
        general_config = args['general_config']
        train_data = args['train_data']
        valid_data = args['valid_data']
        saved = args['saved']
        label = args['label']
        saved_model_file = args['saved_model_file']
        # general_trainer = self.general_trainer[args['general_trainer_idx']]

        print(f'\nTraining label: {label}:\n')

        # create trainer object
        general_trainer = get_trainer(general_model_type, general_model_name)(general_config, general_model)
        general_trainer.saved_model_file = saved_model_file

        # general_model.apply(general_model._init_weights)

        best_valid_result = \
            general_trainer.fit(train_data, valid_data, verbose=False, saved=saved, show_progress=False)

        # save best trained weights
        if general_trainer.best_model_state:
            best_model_weights = general_trainer.best_model_state['state_dict']
            epoch_id = general_trainer.best_model_state['epoch']
        else:
            best_model_weights = general_model.state_dict()
            epoch_id = 0

        return label, best_valid_result, best_model_weights, epoch_id, general_trainer

    def fit(self, train_data, valid_data=None, verbose=False, saved=True, show_progress=False, callback_fn=None):
        """call different fit() method based on config['parallel']
        """
        if self.config['parallel']:
            print('Run parallel model training for each criteria ...')
            return self.parallel_fit(train_data=train_data, valid_data=valid_data, verbose=verbose, saved=saved)
        else:
            print('Run serial model training for each criteria ...')
            return self.serial_fit(train_data=train_data, valid_data=valid_data, verbose=verbose, saved=saved,
                                   show_progress=show_progress)

    def parallel_fit(self, train_data, valid_data=None, verbose=False, saved=True):
        r"""Run general model fit() method in parallel for each criterion
        """
        # training for each criteria label
        best_result_criteria = {}
        args = {'model_type': self.model.general_model.type,
                'model_name': self.model.general_model_name}
        args_list = []
        for idx, label in enumerate(self.label_fields):
            # change training label
            train_data.dataset.label_field = label
            train_data.label_field = label
            valid_data.dataset.label_field = label
            valid_data.label_field = label

            # get config with label
            general_config = copy.deepcopy(self.config)
            general_config['LABEL_FIELD'] = label

            # re set model parameter
            # self.general_model.LABEL = label
            # self.general_model.apply(self.general_model._init_weights)

            # construct process function parameters
            args['general_model'] = self.general_model[label]
            args['general_config'] = general_config
            args['train_data'] = copy.deepcopy(train_data)
            args['valid_data'] = copy.deepcopy(valid_data)
            args['saved'] = saved
            args['general_trainer_idx'] = idx
            args['label'] = label
            args['saved_model_file'] = \
                self.construct_saved_model_file(target=self.model.general_model_name, post_fix=label)

            args_list.append(copy.deepcopy(args))

        # build multi processing pool and start parallel run
        pool = ThreadPool()
        result_pool = pool.map(self.process_fit, args_list)
        pool.close()
        pool.join()

        # collect results from the pool
        for result in result_pool:
            label = result[0]
            best_valid_result = result[1]
            best_model_weights = result[2]
            best_epoch_id = result[3]
            general_trainer = result[4]

            best_result_criteria[label] = best_valid_result

            print(f'Best validation score (at epoch #{best_epoch_id}) for {label}:')
            print(best_valid_result)

            # save best trained weights and trainer
            self.trained_model_weights[label] = best_model_weights
            self.best_model_state['epoch'].update({label: best_epoch_id})
            self.general_trainer[label] = general_trainer

        self.model.trained_weights = self.trained_model_weights

        return best_result_criteria

    def serial_fit(self, train_data, valid_data=None, verbose=False, saved=True, show_progress=False, callback_fn=None):
        r"""Run general model fit() method in serial model for each criteria, has more detail and clear output
        """

        # training for each criteria label
        best_result_criteria = {}
        for label in self.label_fields:
            if show_progress:
                print(f'\nTraining label: {label}')

            # change training label
            train_data.dataset.label_field = label
            train_data.label_field = label
            valid_data.dataset.label_field = label
            valid_data.label_field = label

            # get config with label
            general_config = copy.deepcopy(self.config)
            general_config['LABEL_FIELD'] = label

            # re set model parameter
            # self.general_model.LABEL = label
            # self.general_model.apply(self.general_model._init_weights)

            # create trainer object
            general_trainer = get_trainer(self.model.general_model.type, self.model.general_model_name) \
                (general_config, self.general_model[label])

            # update saved model file name
            general_trainer.saved_model_file = self.construct_saved_model_file(target=self.model.general_model_name,
                                                                               post_fix=label)

            # append to general training list
            self.general_trainer[label] = general_trainer

            # train model
            best_valid_result = \
                general_trainer.fit(train_data, valid_data, verbose=verbose, saved=saved, show_progress=show_progress)
            best_result_criteria[label] = best_valid_result

            # save best trained weights
            if general_trainer.best_model_state:
                self.trained_model_weights[label] = general_trainer.best_model_state['state_dict']
                best_epoch_id = general_trainer.best_model_state['epoch']
            else:
                self.trained_model_weights[label] = self.general_model.state_dict()
                best_epoch_id = 0

            # update best
            self.best_model_state['epoch'].update({label: best_epoch_id})

            if show_progress:
                print(f'Best validation score (at epoch #{best_epoch_id}) for {label}:')
                print(best_valid_result)

        self.model.trained_weights = self.trained_model_weights

        return best_result_criteria

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.
        Args:
            eval_data:
            load_best_model:
            model_file:
            show_progress:
        """
        # evaluate for each criteria label
        best_result_criteria = {}
        best_epoch_id = 0
        for label in self.label_fields:
            print(f'Evaluate label {label}')

            # set label
            eval_data.dataset.label_field = label

            # evaluate model using eval_data
            test_result, best_epoch_id = self.general_trainer[label].evaluate(eval_data,
                                                                              load_best_model=load_best_model,
                                                                              show_progress=show_progress)
            best_result_criteria[label] = test_result

        return best_result_criteria, best_epoch_id


class OverallRPTrainer(MultiCriteriaTrainer):
    r"""MultiCriteriaTrainer is designed for multi criteria models, which set the epoch to 1 whatever the config.

    """

    def __init__(self, config, model, built_datasets, fold_id=None, caller=None):
        super(OverallRPTrainer, self).__init__(config, model, built_datasets, fold_id, caller)

        # update saved model file name
        self.saved_model_file = self.construct_saved_model_file()

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data.
        """

        # construct multi criteria rating tensor
        train_data.dataset.inter_feat.interaction[self.criteria_vector_name] = self.construct_criteria_tensor(
            train_data)
        valid_data.dataset.inter_feat.interaction[self.criteria_vector_name] = self.construct_criteria_tensor(
            valid_data)

        # train criteria rating for overall rating model
        best_result = super().fit(train_data, valid_data, verbose=verbose, show_progress=show_progress)

        return best_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.
        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value.
        """

        if not eval_data:
            return

        # construct multi criteria rating tensor if needed
        if self.criteria_vector_name not in eval_data.dataset.inter_feat.interaction:
            eval_data.dataset.inter_feat.interaction[self.criteria_vector_name] = \
                self.construct_criteria_tensor(eval_data)

        # evaluate overall rating model
        best_result, best_epoch_id = super().evaluate(eval_data, load_best_model=load_best_model,
                                                      show_progress=show_progress)

        return best_result, best_epoch_id


class CombRPTrainer(MultiCriteriaTrainer):
    r"""MultiCriteriaTrainer is designed for multi criteria models, which set the epoch to 1 whatever the config.
        'criteria_model' and 'overall_model' must be of  ModelType.MULTICRITERIA
    """

    def __init__(self, config, model, built_datasets=None, fold_id=None, caller=None):
        super(CombRPTrainer, self).__init__(config, model, built_datasets, fold_id, caller)

        # get trainer for criteria rating model and # reset related configuration settings
        criteria_config = copy.deepcopy(config)
        criteria_config.final_config_dict.pop('overall_model')
        criteria_config.final_config_dict.pop('MODEL_TYPE')
        criteria_config.final_config_dict.update(criteria_config.final_config_dict.pop('criteria_model'))
        self.criteria_config = MCConfig(model=criteria_config.final_config_dict['model'],
                                        config_dict=criteria_config.final_config_dict)
        init_seed(config['seed'], True)

        # make sure criteria prediction model does run Pareto sort
        criteria_config['sorting_algorithm'] = None
        criteria_config['sorting_weight'] = 0
        self.model.criteria_model.sorting_algorithm = None
        self.model.criteria_model.sorting_weight = 0

        caller_name = config['model']
        if self.caller:
            caller_name = self.caller + '-' + caller_name

        # get three data samplers: (train_sampler, valid_sampler, test_sampler) for criteria model
        self.criteria_samplers = create_samplers(self.criteria_config, dataset=None, built_datasets=self.built_datasets)

        # get criteria trainer
        self.criteria_trainer = \
            get_trainer(self.criteria_config['MODEL_TYPE'], self.criteria_config['model']) \
                (self.criteria_config, self.model.criteria_model, self.fold_id, caller=caller_name)

        # get overall rating model trainer and # reset related configuration settings
        overall_config = copy.deepcopy(config)
        # remove criteria model config
        overall_config.final_config_dict.pop('criteria_model')
        overall_config.final_config_dict.pop('MODEL_TYPE')
        overall_config.final_config_dict.update(overall_config.final_config_dict.pop('overall_model'))
        self.overall_config = MCConfig(model=overall_config.final_config_dict['model'],
                                       config_dict=overall_config.final_config_dict)
        init_seed(config['seed'], True)

        # get three data samplers: (train_sampler, valid_sampler, test_sampler) for overall model
        self.overall_samplers = create_samplers(self.overall_config, dataset=None, built_datasets=self.built_datasets)

        self.overall_trainer = \
            get_trainer(self.overall_config['MODEL_TYPE'], self.overall_config['model']) \
                (self.overall_config, self.model.overall_model, self.fold_id, caller=caller_name)

        # multi criteria rating sorting score weight
        self.sorting_weight = min(1, config['sorting_weight'] if config['sorting_weight'] else 0)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train underlying model based on the train data.
        """

        if show_progress:
            print('\nStart criteria model training ...')

        # prepare training and validation data for criteria model training
        criteria_train_data = get_dataloader(self.criteria_config, 'train')(self.criteria_config, train_data.dataset,
                                                                            self.criteria_samplers[0], shuffle=True)
        criteria_valid_data = get_dataloader(self.criteria_config, 'valid')(self.criteria_config, valid_data.dataset,
                                                                            self.criteria_samplers[1], shuffle=False)

        # fit criteria model
        criteria_fit_results = self.criteria_trainer.fit(criteria_train_data, criteria_valid_data, verbose=verbose,
                                                         saved=saved, show_progress=show_progress)
        if show_progress:
            print('\nBest criteria evaluation results average:')
            print(criteria_fit_results)

        overall_fit_results = None
        if self.sorting_weight < 1:
            # train overall rating model
            if show_progress:
                print('\nStart overall rating training ...')

            # prepare data for overall model training
            overall_train_data = get_dataloader(self.overall_config, 'train')(self.overall_config,
                                                                              train_data.dataset,
                                                                              self.overall_samplers[0], shuffle=True)
            overall_valid_data = get_dataloader(self.overall_config, 'valid')(self.overall_config,
                                                                              valid_data.dataset,
                                                                              self.overall_samplers[1], shuffle=False)

            overall_fit_results = self.overall_trainer.fit(overall_train_data, overall_valid_data, verbose=verbose,
                                                           saved=saved, show_progress=show_progress)

            if show_progress:
                print('\nBest overall rating evaluation results:')
                print(overall_fit_results)
                print('')
        else:
            self.overall_trainer.best_model_state['epoch'] = 0

        self.best_model_state['epoch'] = {'criteria': self.criteria_trainer.best_model_state['epoch'],
                                          'overall': self.overall_trainer.best_model_state['epoch']}

        return {'criteria_training': criteria_fit_results, 'overall_training': overall_fit_results}

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.
        This method is called to evaluate the trained criteria model and overall rating model, not called in self.fit()
        Args:
            eval_data:
            load_best_model:
            model_file:
            show_progress:
        """
        if not eval_data:
            return

        # set for evaluation
        self.model.criteria_model.eval()
        self.model.overall_model.eval()

        # set up evaluation function
        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                # get item feature name such as: movie_id
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num  # David Wang: get total item number in evaluation data

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )

        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        criteria_best_epoch_id = self.criteria_trainer.best_model_state['epoch'] \
            if self.criteria_trainer.best_model_state else 0
        overall_best_epoch_id = self.overall_trainer.best_model_state['epoch'] \
            if self.overall_trainer.best_model_state else 0

        return result, {'criteria': criteria_best_epoch_id, 'overall': overall_best_epoch_id}


class JointRPTrainer(MultiCriteriaTrainer):
    """A trainer class for JointRP model. Only evaluate() method need to be implemented
    """

    def __init__(self, config, model, built_datasets=None, fold_id=None, caller=None):

        super(JointRPTrainer, self).__init__(config, model, built_datasets, fold_id, caller)
        self.decimal_place = config['metric_decimal_place']
        self.config['LABEL_FIELD'] = self.criteria_vector_name
        self.criteria_results = None
        self.best_criteria_results = None

        # update saved model file name
        self.saved_model_file = self.construct_saved_model_file()

        try:
            # get index if overall rating.
            # This is needed since overall rating may be at any position in the config file
            self.overall_label_idx = self.label_fields.index(config['RATING_FIELD'])
        except ValueError:
            self.overall_label_idx = None

        if self.config['eval_type'] == EvaluatorType.RANKING and self.overall_label_idx is None:
            raise ValueError(f"overall rating label '{config['RATING_FIELD']}' is not in {self.label_fields}")

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.
        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.
        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """

        # construct multi criteria rating tensor for evaluation purpose
        if valid_data:
            valid_data.dataset.inter_feat.interaction[self.criteria_vector_name] = \
                self.construct_criteria_tensor(valid_data)

        # run parent fit() method
        super().fit(train_data, valid_data=valid_data, verbose=verbose, saved=saved, show_progress=show_progress,
                    callback_fn=callback_fn)

        if show_progress and self.config['eval_type'] == EvaluatorType.VALUE:
            print('\nBest criteria evaluation results:')
            if self.best_criteria_results:
                self.print_criteria_results(self.best_criteria_results)
            else:
                self.print_criteria_results(self.criteria_results)

        return self.best_valid_result

    def _save_checkpoint(self, epoch):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'other_parameter': self.model.other_parameter(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file, _use_new_zipfile_serialization=True)

        self.best_model_state = copy.deepcopy(state)
        self.best_criteria_results = self.criteria_results

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        # store history info in dict, key = userid, value = a list of rated items
        history = defaultdict(list)
        row_idx, col_idx = history_index
        for user, item in zip(row_idx, col_idx):
            history[user].append(item)
        self.model.setHistory(history)

        try:
            # Note: interaction without item ids
            scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        # get only overall rating
        scores = scores[:, self.overall_label_idx]

        # convert to user-item matrix
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            row_idx, col_idx = history_index
            scores[row_idx.long(), col_idx.long()] = -np.inf
        return interaction, scores, positive_u, positive_i

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        # construct multi criteria rating tensor if needed
        if self.criteria_vector_name not in eval_data.dataset.inter_feat.interaction:
            eval_data.dataset.inter_feat.interaction[self.criteria_vector_name] = \
                self.construct_criteria_tensor(eval_data)

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file  # David Wang: use saved the model of the trainer object

            while not os.access(checkpoint_file, os.F_OK) or not os.access(checkpoint_file, os.R_OK):
                import time
                time.sleep(3)
            checkpoint = torch.load(checkpoint_file)

            self.model.load_state_dict(checkpoint['state_dict'])
            if self.best_model_state:
                self.model.load_state_dict(self.best_model_state['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                # get item feature name such as: movie_id
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num  # David Wang: get total item number in evaluation data

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )

        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))

            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)

        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()

        # calculate criteria rating metric for value metric type
        if self.config['eval_type'] == EvaluatorType.VALUE:
            average_results, criteria_results = self.criteria_evaluate(struct)
            self.criteria_results = criteria_results
        else:
            average_results = self.evaluator.evaluate(struct)
            self.criteria_results = None

        # print results
        if show_progress and self.criteria_results:
            self.print_criteria_results(criteria_results)

        best_epoch_id = self.best_model_state['epoch'] if self.best_model_state else 0

        return average_results, best_epoch_id

    @staticmethod
    def print_criteria_results(criteria_result):
        for key in criteria_result.keys():
            print(key, criteria_result[key])

    def criteria_evaluate(self, struct):
        """Evaluate rating prediction for each criteria and  take the average
        Args:
            struct: a DataStruct object with
                'data.label' for true criteria rating
                'rec.score' for predicted criteria rating
        Returns:
            a dictionary with key as metric name and value as the metric value
        """
        # evaluate for each criteria

        results = {}
        for i, label in enumerate(self.label_fields):
            criteria_struct = copy.deepcopy(struct)
            criteria_struct['data.label'] = struct['data.label'][:, i]
            criteria_struct['rec.score'] = struct['rec.score'][:, i]
            criteria_result = self.evaluator.evaluate(criteria_struct)
            results[label] = criteria_result

        # convert criteria results to array
        result_array = np.zeros((len(results), len(list(results.values())[0])), float)
        for i, result in enumerate(results):
            for j, key in enumerate(results[result]):
                result_array[i, j] = results[result][key]

        # take mean for each metric
        average_results = {}
        for j, key in enumerate(list(results.values())[0]):
            average_results[key] = round(result_array[:, j].mean(), self.decimal_place)

        return average_results, results


class HybridRPTrainer(MultiCriteriaTrainer):
    r"""HybridRPTrainer is designed for HybridRP model, which takes weighted sum of multiple rating prediction model

    """

    def __init__(self, config, model, built_datasets=None, fold_id=None):
        super(HybridRPTrainer, self).__init__(config, model, built_datasets, fold_id)

        # get sub model object
        self.sub_model = model.sub_model
        self.sub_model_name = model.sub_model_name
        self.sub_model_config = []
        self.ranking_weight = config['ranking_weight']

        self.train_neg_sample_args = config['train_neg_sample_args']

        # get trainer for each sub model
        self.sub_model_trainer = []
        for model, sub_model_config in zip(self.sub_model, config.final_config_dict['sub_models']):
            # set up config for each sub model
            root_config = copy.deepcopy(config)
            root_config.final_config_dict.pop('sub_models')
            root_config.final_config_dict.pop('MODEL_TYPE')
            root_config.final_config_dict.update(sub_model_config)
            # update other related config parameters
            root_config.model = sub_model_config['model']

            root_config = MCConfig(model=root_config['model'], config_dict=root_config.final_config_dict)
            init_seed(config['seed'], True)

            # get trainer for each sub model
            if root_config['MODEL_TYPE'] == MCModelType.MULTICRITERIA:
                sub_trainer = get_trainer(root_config['MODEL_TYPE'], root_config['model']) \
                    (root_config, model, self.built_datasets, self.fold_id, caller=config['model'])
            else:
                sub_trainer = get_trainer(root_config['MODEL_TYPE'], root_config['model']) \
                    (root_config, model, self.built_datasets)

                # update saved model file name
                sub_trainer.saved_model_file = self.construct_saved_model_file(target=root_config['model'])

            self.sub_model_trainer.append(sub_trainer)

            # save sub model config
            self.sub_model_config.append(root_config)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train underlying model based on the train data.
        """

        if show_progress:
            print('\nStart Hybrid model training ...')

        all_fit_results = []
        for trainer, model_name, config, weight in zip(self.sub_model_trainer, self.sub_model_name,
                                                       self.sub_model_config,
                                                       self.ranking_weight):

            if weight == 0.0:
                continue

            # prepare data load for each sub model
            model_samplers = create_samplers(config, dataset=None, built_datasets=self.built_datasets)
            train_data = get_dataloader(config, 'train')(config, self.built_datasets[0], model_samplers[0],
                                                         shuffle=True)
            valid_data = get_dataloader(config, 'evaluation')(config, self.built_datasets[1], model_samplers[1],
                                                              shuffle=False)

            # train sub model
            fit_results = trainer.fit(train_data, valid_data, verbose=verbose, saved=saved, show_progress=show_progress)
            if show_progress:
                print(f'\nBest sub model {model_name} results average:')
                print(fit_results)

            all_fit_results.append(fit_results)

        return all_fit_results

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.
        This method is called to evaluate the trained sub model, not called in self.fit()
        Args:
            eval_data:
            load_best_model:
            model_file:
            show_progress:
        """
        if not eval_data:
            return

        # set for evaluation
        for model in self.sub_model:
            model.eval()

        # set up evaluation function
        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                # get item feature name such as: movie_id
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num  # David Wang: get total item number in evaluation data

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )

        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        # get best result epic
        best_result_epoch = {}
        for trainer, model_name, weight in zip(self.sub_model_trainer, self.sub_model_name, self.ranking_weight):
            if weight == 0.0:
                continue
            best_result_epoch[model_name] = trainer.best_model_state['epoch']

        return result, best_result_epoch


class CriteriaSortTrainer(MultiCriteriaTrainer):
    r"""MultiCriteriaTrainer is designed for multi criteria models, which set the epoch to 1 whatever the config.

    """

    def __init__(self, config, model, built_datasets, fold_id=None, caller=None):
        super(CriteriaSortTrainer, self).__init__(config, model, built_datasets, fold_id, caller)

        # set up log file name
        caller_name = config['model']
        if self.caller:
            caller_name = self.caller + '-' + caller_name

        # get criteria model
        self.criteria_model = model.criteria_model

        # get trainer for criteria rating model and # reset related configuration settings
        criteria_config = copy.deepcopy(config)
        criteria_config.final_config_dict.update(criteria_config.final_config_dict.pop('criteria_model'))
        criteria_config.final_config_dict.pop('MODEL_TYPE')
        criteria_config = MCConfig(model=criteria_config.final_config_dict['model'],
                                   config_dict=criteria_config.final_config_dict)
        init_seed(config['seed'], True)

        self.criteria_config = criteria_config

        # get criteria trainer
        self.criteria_trainer = \
            get_trainer(criteria_config['MODEL_TYPE'], criteria_config['model']) \
                (self.criteria_config, self.criteria_model, self.built_datasets, self.fold_id, caller=caller_name)

        # get general model training if needed
        self.overall_model = model.overall_model
        if config['sub_sort'] == 'OverallRatingRanking' and self.overall_model is not None:
            overall_config = copy.deepcopy(config)
            overall_config.final_config_dict.update(overall_config.final_config_dict.pop('overall_model'))
            overall_config.final_config_dict.pop('MODEL_TYPE')
            overall_config.final_config_dict.pop('criteria_model')
            overall_config = MCConfig(model=overall_config.final_config_dict['model'],
                                      config_dict=overall_config.final_config_dict)
            init_seed(config['seed'], True)

            self.overall_config = overall_config

            # get overall rating trainer
            self.overall_trainer = \
                get_trainer(overall_config['MODEL_TYPE'], overall_config['model']) \
                    (self.overall_config, self.overall_model, self.built_datasets)
        else:
            self.overall_trainer = None

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train underlying model based on the train data.
        """

        if show_progress:
            print('\nStart criteria model training ...')

        # prepare data load for each sub model
        train_sampler, valid_sampler, _ = create_samplers(self.criteria_config, dataset=None,
                                                          built_datasets=self.built_datasets)
        criteria_train_data = get_dataloader(self.criteria_config, 'train') \
            (self.criteria_config, self.built_datasets[0], train_sampler, shuffle=True)
        criteria_valid_data = get_dataloader(self.criteria_config, 'evaluation') \
            (self.criteria_config, self.built_datasets[1], valid_sampler, shuffle=False)

        # run criteria model training
        criteria_fit_results = self.criteria_trainer.fit(criteria_train_data, criteria_valid_data, verbose=verbose,
                                                         saved=saved, show_progress=show_progress)
        if show_progress:
            print('\nBest criteria evaluation results average:')
            print(criteria_fit_results)

        self.best_model_state['epoch'] = {'criteria': self.criteria_trainer.best_model_state['epoch']}

        # fit general model if needed
        if self.overall_trainer is not None:
            # prepare data load for each sub model
            train_sampler, valid_sampler, _ = create_samplers(self.overall_config, dataset=None,
                                                              built_datasets=self.built_datasets)
            overall_train_data = get_dataloader(self.overall_config, 'train') \
                (self.overall_config, self.built_datasets[0], train_sampler, shuffle=True)
            overall_valid_data = get_dataloader(self.overall_config, 'evaluation') \
                (self.overall_config, self.built_datasets[1], valid_sampler, shuffle=False)

            if show_progress:
                print('\nStart overall rating model training ...')

            # run criteria model training
            overall_fit_results = self.overall_trainer.fit(overall_train_data, overall_valid_data, verbose=verbose,
                                                           saved=saved, show_progress=show_progress)
            if show_progress:
                print('\nBest overall rating evaluation results average:')
                print(overall_fit_results)

            self.best_model_state['epoch'] = {'criteria': self.criteria_trainer.best_model_state['epoch']}

        return {'criteria_training': criteria_fit_results}

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.
        This method is called to evaluate the trained criteria model and overall rating model, not called in self.fit()
        Args:
            eval_data:
            load_best_model:
            model_file:
            show_progress:
        """
        if not eval_data:
            return

        # construct multi criteria rating tensor if needed
        if self.overall_model and self.overall_model._get_name() == 'OverallRP' \
                and self.criteria_vector_name not in eval_data.dataset.inter_feat.interaction:
            eval_data.dataset.inter_feat.interaction[self.criteria_vector_name] = \
                self.construct_criteria_tensor(eval_data)

        # set for evaluation
        self.criteria_model.eval()

        # set up evaluation function
        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                # get item feature name such as: movie_id
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num  # David Wang: get total item number in evaluation data

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )

        for batch_idx, batched_data in enumerate(iter_data):
            start_time = time()
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
            print(
                f'Running time for batch {batch_idx}: {time() - start_time} seconds, total: {batch_idx * (time() - start_time)}')
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        criteria_best_epoch_id = self.criteria_trainer.best_model_state['epoch'] \
            if self.criteria_trainer.best_model_state else 0

        return result, {'criteria': criteria_best_epoch_id}


class DeepCriteriaChainTrainer(MultiCriteriaTrainer):
    r"""MultiCriteriaTrainer is designed for multi criteria models, which set the epoch to 1 whatever the config.

    """

    def __init__(self, config, model, built_datasets, fold_id=None, caller=None):
        super(DeepCriteriaChainTrainer, self).__init__(config, model, built_datasets, fold_id, caller)

        # list of criteria model
        self.criteria_model = model.criteria_model
        self.criteria_label = model.criteria_label
        if model.multi_label:
            self.all_labels = self.criteria_label
            self.config['LABEL_FIELD'] = self.criteria_vector_name
        else:
            self.all_labels = self.criteria_label + [config['LABEL_FIELD']]

        self.criteria_model_trainer = []
        self.criteria_config = []
        for model, label in zip(self.criteria_model, self.all_labels):
            # set up config for each criteria model
            criteria_config = copy.deepcopy(config)
            criteria_config.final_config_dict.update(criteria_config.final_config_dict.pop('criteria_model'))
            criteria_config.final_config_dict.pop('MODEL_TYPE')
            criteria_config['LABEL_FIELD'] = label

            criteria_config = MCConfig(model=criteria_config.final_config_dict['model'],
                                       config_dict=criteria_config.final_config_dict)
            init_seed(config['seed'], True)

            # get trainer for each criteria model
            criteria_trainer = get_trainer(criteria_config['MODEL_TYPE'], criteria_config['model'])(criteria_config,
                                                                                                    model)

            # add fold id to the saved model file name for parallel model
            criteria_trainer.saved_model_file = \
                self.construct_saved_model_file(target=criteria_config['model'], post_fix=label)

            # get trainer for each sub model
            self.criteria_model_trainer.append(criteria_trainer)
            self.criteria_config.append(criteria_config)

        self.best_model_state['epoch'] = {}

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):

        # initiate criteria label data vector
        train_data.dataset.inter_feat.interaction[self.predicted_label_name] = \
            torch.zeros_like(train_data.dataset.inter_feat.interaction[self.criteria_label[0]])

        # train model for each criteria
        all_fit_results = {}
        label_idx = 0
        predicted_train_score = None
        predicted_valid_score = None
        for model_trainer, label, config in zip(self.criteria_model_trainer, self.all_labels, self.criteria_config):

            torch.cuda.empty_cache()

            if show_progress:
                print(f'\nStart criteria model training for label: {label} ...')

            model_train_data = train_data.dataset
            model_valid_data = valid_data.dataset

            model_train_data.label_field = label
            model_valid_data.label_field = label

            # prepare data load for each sub model
            train_sampler, valid_sampler, _ = create_samplers(config, dataset=None, built_datasets=self.built_datasets)
            train_data_loader = get_dataloader(config, 'train')(config, self.built_datasets[0], train_sampler,
                                                                shuffle=True)
            valid_data_loader = get_dataloader(config, 'evaluation')(config, self.built_datasets[1], valid_sampler,
                                                                     shuffle=False)

            # construct criteria vector from trained model
            if label_idx > 0:
                train_data_loader, predicted_train_score = \
                    self.update_data_set(train_data_loader, predicted_train_score, label_idx)
                valid_data_loader, predicted_valid_score = \
                    self.update_data_set(valid_data_loader, predicted_valid_score, label_idx)

            fit_results = model_trainer.fit(train_data_loader, valid_data=valid_data_loader, verbose=verbose,
                                            saved=saved, show_progress=show_progress, callback_fn=callback_fn)
            # update label index
            label_idx += 1
            all_fit_results[label] = fit_results

            # save best trained weights
            if 'epoch' in model_trainer.best_model_state:
                # self.trained_model_weights[label] = model_trainer.best_model_state['state_dict']
                best_epoch_id = model_trainer.best_model_state['epoch']
            else:
                # self.trained_model_weights[label] = model_trainer.state_dict()
                best_epoch_id = 0

            # update best
            self.best_model_state['epoch'].update({label: best_epoch_id})

        return all_fit_results

    def update_data_set(self, data_set, predicted_score, label_idx):
        """predict estimation of the next label
        Args:
            data_set: data set including user_id, item_id
            predicted_score: predicted score of previous trained criteria label
            label_idx: index of label to be predicted in the criteria_label list
        Returns:
            data set containing 'predicted_label_name' tensor
        """

        # add the predicted criteria rating to the data set
        if predicted_score is not None:
            data_set.dataset.inter_feat.interaction[self.predicted_label_name] = predicted_score

        # run prediction
        score = self.criteria_model[label_idx - 1].predict(data_set.dataset.inter_feat.to(self.device))
        score = torch.clamp(score, min=self.min_rating_value, max=self.max_rating_value)

        # update data set
        if label_idx == 1:
            predicted_score = score
        elif len(predicted_score.shape) == 1:
            predicted_score = torch.stack((predicted_score, score), dim=-1)
        else:
            predicted_score = torch.hstack((predicted_score, torch.unsqueeze(score, -1)))

        data_set.dataset.inter_feat.interaction[self.predicted_label_name] = predicted_score

        return data_set, predicted_score

    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.
        This method is called to evaluate the trained sub model, not called in self.fit()
        Args:
            eval_data:
            load_best_model:
            model_file:
            show_progress:
        """
        torch.cuda.empty_cache()

        if not eval_data:
            return
        else:
            # construct multi criteria rating tensor for evaluation purpose
            eval_data.dataset.inter_feat.interaction[self.criteria_vector_name] = \
                self.construct_criteria_tensor(eval_data)

        # set for evaluation
        for model in self.criteria_model:
            model.eval()

        # set up evaluation function
        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                # get item feature name such as: movie_id
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval

        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num  # David Wang: get total item number in evaluation data

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )

        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        # get best result epic
        best_result_epoch = {}
        for trainer, model_name in zip(self.criteria_model_trainer, self.all_labels):
            best_result_epoch[model_name] = trainer.best_model_state['epoch']

        return result, best_result_epoch


class CriteriaNNCFTrainer(MultiCriteriaTrainer):
    def __init__(self, config, model, built_datasets=None, fold_id=None, caller=None):
        super(CriteriaNNCFTrainer, self).__init__(config, model, data_sets=built_datasets, fold_id=fold_id,
                                                  caller=caller)
