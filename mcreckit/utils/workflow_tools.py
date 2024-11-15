# @Time   : 2022/01/19
# @Author : David Wang

"""
All useful functions to manage model training and evaluation workflows

Usage: import recbole.utils.workflow_tools as wft

"""

import copy
import shutil
import time
import gc
import glob
import os
from logging import getLogger

import numpy as np
import torch

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

from recbole.utils import EvaluatorType
from recbole.utils import init_seed
from mcreckit.data.dataloader.general_dataloader import LabeledRankingEvalDataLoader
from mcreckit.data.utils import data_preparation
from mcreckit.utils import CustomConstant, MCModelType, get_model, set_information_gain_order, get_trainer
from mcreckit.utils.logger import init_logger


def process_run(args):
    """This is single process run for model training
    Args:
        args: a dictionary object with keys:
            'config', 'train_data_loader', 'valid_data_loader'
    Returns:
        training results
    """
    global criteria_results
    config = args['config']
    train_data_loader = args['train_data_loader']
    valid_data_loader = args['valid_data_loader']
    test_data_loader = args['test_data_loader']
    dataset = args['dataset']
    fold_id = args['fold_id']
    logger = args['logger']
    saved = args['saved']
    load_best_model = args['load_best_model']

    # get model
    init_seed(config['seed'], True)
    model = get_model(config['model'])(config, train_data_loader.dataset).to(config['device'])

    # get trainer
    if config['MODEL_TYPE'] == MCModelType.MULTICRITERIA:
        trainer_obj = get_trainer(config['MODEL_TYPE'], config['model']) \
            (config, model,
             built_datasets=[train_data_loader.dataset, valid_data_loader.dataset, test_data_loader.dataset],
             fold_id=fold_id)
    else:
        trainer_obj = get_trainer(config['MODEL_TYPE'], config['model']) \
            (config, model,
             built_datasets=[train_data_loader.dataset, valid_data_loader.dataset, test_data_loader.dataset])

        # append saved model file name with fold id
        # the default name used HH_MM_SS and cannot differentiate different fold
        split_name = trainer_obj.saved_model_file.split('.')
        trainer_obj.saved_model_file = f"{split_name[0]}-fold_{fold_id}.{split_name[-1]}"

    # run model training
    logger.info(f'Start training process for fold #{fold_id} ...')
    train_valid_data_loader = copy.deepcopy(valid_data_loader)
    training_results = trainer_obj.fit(train_data_loader, train_valid_data_loader, saved=saved, show_progress=False,
                                       verbose=False)

    # run model testing: use test_data_loader is available or no evaluation results from training process
    test_result = None
    best_epoch_id = 0
    if test_data_loader or isinstance(training_results, dict):

        # check if new data loaders are needed
        # For CombRP model where underlying model evaluation metric has different type than overall evaluation metrics
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy == 'none' and config['eval_type'] == EvaluatorType.RANKING and \
                not isinstance(valid_data_loader, LabeledRankingEvalDataLoader):
            # create data loader
            train_data_loader, valid_data_loader, test_data_loader = \
                data_preparation(config, dataset=dataset, split_data=args['split_data_list'],
                                 use_criteria_model=False)

        if test_data_loader:
            test_result, best_epoch_id = trainer_obj.evaluate(test_data_loader, load_best_model=load_best_model,
                                                              show_progress=False)
        elif isinstance(training_results, dict):
            test_result, best_epoch_id = trainer_obj.evaluate(valid_data_loader, load_best_model=load_best_model,
                                                              show_progress=False)

        try:
            criteria_results = trainer_obj.criteria_results
        except AttributeError:
            criteria_results = None

    torch.cuda.empty_cache()
    del model, trainer_obj
    gc.collect()

    return training_results, test_result, best_epoch_id, criteria_results


def run_model_training_parallel(dataset, config, logger, saved=True, load_best_model=True):
    """Run model training with cross validation in parallel mode
    Args:
        dataset:  dataset of all user interaction data
        config: cnf
        logger: logger object
        saved: True or False, True: save best model file in training
        load_best_model: True: load best saved model file in evaluation
    Returns:
        training result list
    """
    built_data_list = dataset.build()
    if not isinstance(built_data_list[0], list) and len(built_data_list) == 3:
        built_data_list = [built_data_list]

    # use eval_data as test_data for hybrid model if there is no test data
    save_training_result = True
    if config['sub_models'] and built_data_list[0][2].inter_num == 0:
        save_training_result = False
        for i in range(len(built_data_list)):
            # assign validation data to testing data
            built_data_list[i][2] = built_data_list[i][1]

    # prepare data
    args = {'config': copy.deepcopy(config)}
    args_list = []
    fold_id = 0
    for train_data, valid_data, test_data in built_data_list:
        # create data loader for training and evaluation
        train_data_loader, valid_data_loader, test_data_loader = \
            data_preparation(config, dataset=dataset, split_data=[train_data, valid_data, test_data],
                             use_criteria_model=True)

        args['train_data_loader'] = train_data_loader
        args['valid_data_loader'] = valid_data_loader
        args['test_data_loader'] = test_data_loader
        args['split_data_list'] = [train_data, valid_data, test_data]
        args['dataset'] = dataset
        args['fold_id'] = fold_id
        args['logger'] = logger
        args['saved'] = saved
        args['load_best_model'] = load_best_model

        args_list.append(copy.deepcopy(args))
        fold_id += 1

    # build multi processing pool and start parallel run
    # pool = ThreadPool()
    # result_pool = pool.map(process_run, args_list)
    # pool.close()
    # pool.join()

    num_processes = config['eval_args']['split']['num_processes']
    with Pool(processes=num_processes) as pool:
        result_pool = pool.map(process_run, args_list)

    # collect training results
    training_result_list = []
    criteria_evaluation_result_list = []
    for i, result in enumerate(result_pool):
        training_results = result[0]
        test_result = result[1]
        best_epoch_id = result[2]
        criteria_evaluation_results = result[3]

        if training_results and not isinstance(training_results, dict) and save_training_result:
            logger.info(f'Evaluation results for fold #{i} at epoch #{best_epoch_id}:')
            logger.info(training_results)
            training_result_list.append(training_results)
        else:
            if isinstance(training_results, dict):
                logger.info(f"Training results for fold #{i} at epoch #{best_epoch_id}")
                logger.info(training_results)
            logger.info(f'Test results for fold #{i} at epoch #{best_epoch_id}:')
            logger.info(test_result)
            training_result_list.append(test_result)

        criteria_evaluation_result_list.append(criteria_evaluation_results)

    return training_result_list, criteria_evaluation_result_list


def run_model_training(dataset, config, show_progress=True, saved=True, load_best_model=False, run_serial=False):
    """This is the process of training/evaluating and testing model given a set of data loader objects:
    Args:
        dataset: a data set object containing all training and evaluation data
        config: configuration object
        show_progress: True or False
        saved: True for save the best model file
        load_best_model: True for load best model file in evaluation, if True, set saved=True
        run_serial: False for parallel run for cross validation, True for serial run, for N-fold cross validation
    Returns:
        None
    """
    # logger initialization
    log_handler, log_path = init_logger(config)
    logger = getLogger()

    # need to add file handler here is not assigned
    if len(logger.handlers) == 1:
        logger.addHandler(log_handler)

    logger.info(config)

    if config['model'] in CustomConstant.CHAIN_MODELS.value:
        config['MULTI_LABEL_FIELD'] = set_information_gain_order(dataset, config)
        print(f"Criteria label order: {config['MULTI_LABEL_FIELD']}")

    if torch.cuda.is_available():
        logger.info('GPU is available ')
        logger.info(f'Number of GPUs: {torch.cuda.device_count()}')
        logger.info(f'GPU device name: {torch.cuda.get_device_name(0)}')
        logger.info(f'Currently selected GPU device index: {torch.cuda.current_device()}')
        if config['use_gpu'] is None:
            logger.info("GPU is not used based on 'use_gpu' setting")
    else:
        logger.info('GPU is not available')

    start_time = time.time()

    if run_serial:
        if 'CV' in config['eval_args']['split']:
            logger.info('Run cross validation model training in serial ...')
        else:
            logger.info('Run model training in serial ...')

        # make sure CriteriaRP will be run in serial
        config['parallel'] = False
        training_result_list, criteria_result_list = \
            run_model_training_serial(dataset, config, logger, show_progress=show_progress, saved=saved,
                                      load_best_model=load_best_model)

    elif 'CV' in config['eval_args']['split']:
        if config['eval_args']['split']['parallel']:
            logger.info('Run cross validation model training in parallel ...')

            training_result_list, criteria_result_list = run_model_training_parallel(dataset, config, logger, saved=saved,
                                                                                 load_best_model=load_best_model)
        else:
            logger.info('Run model training in serial ...')

            training_result_list, criteria_result_list = \
                run_model_training_serial(dataset, config, logger, show_progress=show_progress, saved=saved,
                                          load_best_model=load_best_model)

    # calculate average if N fold
    average_metric = calculate_average_metrics(training_result_list)

    # print out results
    if len(training_result_list) > 1:
        logger.info(f'{len(training_result_list)} fold training and evaluation average:')
    else:
        logger.info('Training and evaluation results:')

    for label in average_metric:
        if config['eval_type'].name == 'RANKING':
            f1_score = calculate_f1_score(average_metric[label])
            average_metric[label].update(f1_score)

        logger.info(f"{label}: {average_metric[label]}")

    # calculate average criteria rating for N fold
    if None not in criteria_result_list:
        average_criteria_metric = {}
        criteria_results_dict = {}
        for label in criteria_result_list[0].keys():
            criteria_results_dict[label] = list()

        # construct list of evaluation results for each criteria label
        for result in criteria_result_list:
            for label in criteria_results_dict.keys():
                criteria_results_dict[label].append(result[label])

        # calculate average for each criteria
        for label in criteria_results_dict:
            average_criteria_metric[label] = calculate_average_metrics(criteria_results_dict[label])['output']

        # output to log file
        logger.info(f"{len(training_result_list)} fold Criteria metric average:")
        for label in average_criteria_metric:
            logger.info(f"{label}: {str(average_criteria_metric[label])}")

    # print out major training/evaluation settings
    setting_keys = ['epochs', 'stopping_step', 'train_batch_size', 'dropout_prob', 'learning_rate',
                    'eval_batch_size']

    logger.info('Key training parameters:')
    logger.info({key: config[key] for key in setting_keys})

    logger.info(f'Total running time: {time.time() - start_time} seconds')

    # update log file name
    log_handler.close()
    logger.removeHandler(log_handler)

    # get best metric

    validation_metric = config['valid_metric'].lower()
    """
    if config['eval_type'].name == 'RANKING':
        top_k = validation_metric.split('@')[1]
        saved_metrics = [f"{metric.lower()}@{top_k}" for metric in config['metrics']]
    else:
        saved_metrics = [metric.lower() for metric in config['metrics']]
    """

    if 'output' in average_metric:
        best_metric_value = average_metric['output'][validation_metric]

        # get results of metrics to be saved
        # save_results = dict([(metric, average_metric['output'][metric]) for metric in saved_metrics])
        save_results = average_metric['output']
    else:
        # multi output
        average_output = np.zeros((len(average_metric), len(list(average_metric.values())[0])))
        for i, label in enumerate(average_metric):
            for j, metric in enumerate(average_metric[label]):
                average_output[i, j] = average_metric[label][metric]

        best_metric_value = np.NaN
        # get average across all labels
        for j, metric in enumerate(list(average_metric.values())[0]):
            if metric == validation_metric:
                best_metric_value = round(average_output[:, j].mean(), 5)
                break
        # get results of metrics to be saved
        save_results = dict([(metric, value) for metric, value in zip(list(list(average_metric.values())[0].keys()),
                                                                      average_output.mean(axis=0))])

    log_file_name = f"{log_path[:-4]}-{validation_metric}={best_metric_value}.log"
    shutil.move(log_path, log_file_name)

    print(f'Running log saved in {log_file_name}')
    update_best_log(config, log_file_name, best_metric_value)

    # calculate f1 score for ranking metrics
    if config['eval_type'].name == 'RANKING':
        save_results.update(calculate_f1_score(save_results))

    return save_results


def run_model_training_serial(dataset, config, logger, show_progress=True, saved=True, load_best_model=False):
    """This is the process of training/evaluating and testing model given a set of data loader objects:
    Args:
        dataset: Dataset object containing all training and evaluation data
        config: configuration object
        logger: logger object
        saved: True or False
        show_progress: True or False
        load_best_model: True or False
    Returns:
        None
    """
    # split data set into training, validation and testing datasets
    built_data_list = dataset.build()
    if not isinstance(built_data_list[0], list) and len(built_data_list) == 3:
        built_data_list = [built_data_list]

    # use eval_data as test_data for hybrid model if there is no test data
    save_training_result = True
    if config['sub_models']:
        save_training_result = False
        if built_data_list[0][2].inter_num == 0:
            for i in range(len(built_data_list)):
                # assign validation data to testing data
                built_data_list[i][2] = built_data_list[i][1]

    # run model for each fold in data set
    training_result_list = []
    criteria_evaluation_result_list = []
    fold_idx = 0
    for train_data, valid_data, test_data in built_data_list:

        fold_config = copy.deepcopy(config)
        # create data loader for training and evaluation
        train_data_loader, valid_data_loader, test_data_loader = \
            data_preparation(fold_config, dataset=dataset, split_data=[train_data, valid_data, test_data],
                             use_criteria_model=True)

        if len(built_data_list) > 1:
            logger.info(f'Training and evaluating model with data fold #{fold_idx} ...')

        # get model
        init_seed(config['seed'], True)
        model = get_model(fold_config['model'])(fold_config, train_data_loader.dataset).to(fold_config['device'])

        # get trainer
        trainer_obj = get_trainer(fold_config['MODEL_TYPE'], fold_config['model']) \
            (fold_config, model, built_datasets=[train_data, valid_data, test_data])

        # run model training
        logger.info(f'Start training process fold #{fold_idx}...')
        train_valid_data_loader = copy.deepcopy(valid_data_loader)
        training_results = trainer_obj.fit(train_data_loader, train_valid_data_loader, saved=saved,
                                           show_progress=show_progress, verbose=False)

        if training_results and not isinstance(training_results, dict) and save_training_result:
            logger.info(f'Best training results fold #{fold_idx}:')
            logger.info(training_results)
            training_result_list.append(training_results)

        # run model testing: use test_data_loader is available
        # For CombRP model where underlying model evaluation metric has different type than overall evaluation metrics
        if test_data_loader or isinstance(training_results, dict):

            logger.info(f'Start testing process fold #{fold_idx} ...')
            if test_data_loader:
                test_result, best_epoch_id = trainer_obj.evaluate(test_data_loader, load_best_model=load_best_model,
                                                                  show_progress=show_progress)
            else:
                # need a separate evaluation with validation data even no test data loader available
                logger.info(f'Training result for fold #{fold_idx}:')
                logger.info(training_results)
                print('')
                test_result, best_epoch_id = trainer_obj.evaluate(valid_data_loader, load_best_model=load_best_model,
                                                                  show_progress=show_progress)

            logger.info(f'Test results fold #{fold_idx} at epoch #{best_epoch_id}:')
            logger.info(test_result)
            training_result_list.append(test_result)

        try:
            criteria_eval_results = trainer_obj.criteria_results
        except AttributeError:
            criteria_eval_results = None

        criteria_evaluation_result_list.append(criteria_eval_results)
        fold_idx += 1

        # release GPU memory
        del model, trainer_obj, training_results, train_data_loader, valid_data_loader, test_data_loader, fold_config, \
            train_valid_data_loader
        gc.collect()
        torch.cuda.empty_cache()

    return training_result_list, criteria_evaluation_result_list


def calculate_average_metrics(train_results, decimal_place=5):
    """Calculate average training metrics across list of training results
        Args:
            train_results:  list dictionary of evaluation results for multiple output model
                            list of metrics for single output model
            decimal_place: number of decimals of average
        Returns:

    """
    # determine if these are metrics from multi output model
    try:
        multiple_outputs = isinstance(list(train_results[0].values())[0], dict)
    except AttributeError:
        multiple_outputs = False

    final_average_results = {}

    results = []
    if not multiple_outputs:
        # rebuild training results to be the same as multi output cases
        for fold in train_results:
            results.append({'output': fold})
    else:
        results = train_results

    # build result matrix for each output
    metric_array = {}
    fold_0 = results[0]
    for label in fold_0:
        metric_array[label] = np.zeros((len(results), len(fold_0[label])), float)

    # save metric value to metric array
    for i, fold in enumerate(results):
        for label in fold:
            for j, metric in enumerate(fold[label]):
                metric_array[label][i, j] = fold[label][metric]

    # calculate average across all folds
    for label in fold_0:
        average_results = {}
        for j, metric in enumerate(fold_0[label]):
            average_results[metric] = round(metric_array[label][:, j].mean(), decimal_place)
        final_average_results[label] = average_results

    return final_average_results


def update_best_log(config, newlog, metric_value):
    dataset = config['dataset']
    metric = config['valid_metric']

    end = newlog.rindex('.')
    s1 = newlog.index('-')
    s2 = newlog.index('-', s1 + 1, end)
    model = newlog[s1 + 1:s2]

    match = [dataset, model, metric]

    folder_best = './log/best/'
    existing_logs = glob.glob(folder_best + '/*.log')

    found = False
    oldlog = None
    for file in existing_logs:
        if all(x in file for x in match):
            oldlog = file
            found = True
            break

    newlog_filename = newlog[newlog.rindex('/') + 1:]

    if not found:
        shutil.copyfile(newlog, folder_best + newlog_filename)
    else:
        # compare which log file is better
        ranking = False
        if config['eval_type'] == EvaluatorType.RANKING:
            ranking = True

        newvalue = metric_value
        oldvalue = float(oldlog[oldlog.rindex('=') + 1: oldlog.rindex('.')])

        if ranking:
            if newvalue > oldvalue:
                shutil.copyfile(newlog, folder_best + newlog_filename)
                os.remove(oldlog)
                impro = (newvalue - oldvalue) / oldvalue
                print('Better results! improvement: {:.2%}'.format(impro) + ', bes log saved in ' + folder_best)
        else:
            if newvalue < oldvalue:
                shutil.copyfile(newlog, folder_best + newlog_filename)
                os.remove(oldlog)
                impro = (oldvalue - newvalue) / oldvalue
                print('Better results! improvement: {:.2%}'.format(impro) + ', best log saved in ' + folder_best)
    return


def calculate_f1_score(metric_values):
    """Calculate F1 score based on formula:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
    Args:
        metric_values: dictionary of precision and recall value
    Return:
        key value of f1 metric
    """

    # get precision and recall for each top k
    precision_recall = {}
    for metric in metric_values:
        top_k = metric.split('@')[1]

        if metric.split('@')[0] in ['precision', 'recall']:
            if top_k in precision_recall:
                precision_recall[top_k].update({metric: metric_values[metric]})
            else:
                precision_recall[top_k] = {metric: metric_values[metric]}

    # calculate f1 for each top k
    f1_list = dict()
    for top_k in precision_recall:
        precision, recall = None, None
        for metric in precision_recall[top_k]:
            if metric.split('@')[0] == 'precision':
                precision = metric_values[metric]
            elif metric.split('@')[0] == 'recall':
                recall = metric_values[metric]

        if precision and recall:
            f1_score = round(2 * (precision * recall) / (precision + recall), 5)
            f1_list.update({f"f1@{top_k}": f1_score})

    return f1_list
