# @Time : 2024/10/24
# @Author : Yong Zheng


import logging
import os
import re

from colorama import init
from colorlog import colorlog
from recbole.utils import ensure_dir, get_local_time
from recbole.utils.logger import log_colors_config, RemoveColorFilter


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(config)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    LOGROOT = './log/'
    dir_name = os.path.dirname(LOGROOT)
    ensure_dir(dir_name)

    if config.config_file_name:
        log_file_name = re.split('/|\.', config.config_file_name)[-2]
        if config['sorting_weight'] is not None and float(config['sorting_weight']) > 0:
            log_file_name = log_file_name + "_" + config['sorting_algorithm'] + "-w=" + str(config['sorting_weight'])
        if config['MULTI_LABEL_FIELD'] and config['LABEL_FIELD'] in config['MULTI_LABEL_FIELD']:
            log_file_name = log_file_name + "_OverallRating"
        logfilename = '{}-{}-({})-{}.log'.format(config['dataset'], log_file_name, config['time_id'], get_local_time())
    else:
        logfilename = '{}-{}-({})-{}.log'.format(config['dataset'], config['model'], config['time_id'],
                                                 get_local_time())

    logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh])

    return fh, logfilepath
