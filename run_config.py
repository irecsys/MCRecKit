import argparse
import copy
import os
import time
import pandas as pd
import mcreckit.utils.workflow_tools as wft
import shutil

from datetime import datetime
from recbole.utils import init_seed
from mcreckit.config.configurator import MCConfig
from mcreckit.data.utils import create_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("config_files", nargs='?', help='config files')
    parser.add_argument("--serial", action='store_true', help='to set parallel run for cross validation')

    args, _ = parser.parse_known_args()
    run_serial = args.serial

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    config = MCConfig(config_file_list=config_file_list)
    init_seed(config['seed'], True)

    print(f"model: {config['model']}")
    print(f"dataset: {config['dataset']}")

    # generate a time id and save it to config object
    time_id = str(time.time())[4:15]
    config['time_id'] = time_id

    # create file to save the running config
    file_name_root = f"{config['dataset']}_{config_file_list[0].split('/')[-1]}_({time_id})_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}"
    result_file_name = f"log/results/{file_name_root}_results.csv"
    config_file_name = f"log/results/{file_name_root}_config.txt"
    os.makedirs("log/results/", exist_ok=True)

    # copy config file
    shutil.copyfile(config_file_list[0], config_file_name)

    df_all_results = pd.DataFrame()

    # get data
    dataset = create_dataset(copy.deepcopy(config))
    print('\nDataset facts:')
    print(dataset)

    # run model training and testing
    results = wft.run_model_training(dataset, config, show_progress=False, saved=False, load_best_model=False,
                                     run_serial=run_serial)

    # convert results to data frame
    df_results = pd.DataFrame(results.items())
    df_results = pd.pivot_table(df_results, values=1, columns=0)

    if df_all_results.shape[0] == 0:
        df_all_results = df_results
    else:
        df_all_results = pd.concat((df_all_results, df_results))

    # save all testing results
    df_all_results.to_csv(result_file_name, index=False)
    print(f"Results are saved in {result_file_name}")
