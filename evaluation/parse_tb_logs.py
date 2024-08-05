import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

tensor_eval_attrs_ = ['evaluation_loss_vs_iterations',
                      'evaluation_binary_accuracy_vs_iterations',
                      'evaluation_auc_1_vs_iterations',
                      'evaluation_precision_1_vs_iterations',
                      'evaluation_sensitivity_vs_iterations',
                      'evaluation_specificity_vs_iterations',
                      'evaluation_dice_coefficient_vs_iterations',
                      'evaluation_binary_io_u_1_vs_iterations']


def test_tb_logs(log_path):
    # Get a list of all event files in the logs directory
    event_files = [os.path.join(log_path, f) for f in os.listdir(log_path) if f.startswith('events')]
    # Create an EventAccumulator
    ea = event_accumulator.EventAccumulator(event_files[0])

    # Optionally: Load the data
    ea.Reload()

    # Get the available tags or keys in the logs
    tags = ea.Tags()
    tensors = tags['tensors']
    log_attrs = tensors
    parsed_logs = {}
    for attr in log_attrs:
        parsed_logs[attr] = np.array([tf.make_ndarray(x.tensor_proto) for x in ea.Tensors(attr)])
    return pd.DataFrame(parsed_logs)


def get_runs(logs_path, model_name):
    paths = [os.path.join(logs_path, x) for x in os.listdir(logs_path) if x.startswith(model_name)]
    dates = [os.listdir(x)[0] for x in paths]
    # print(dates)
    instance_logs_dirs = sorted([os.path.join(x, y) for (x, y) in zip(paths, dates)])
    test_logs = []
    for x in instance_logs_dirs:
        test_logs.append(test_tb_logs(os.path.join(x, 'test/validation')))

    instance_train_logs = [pd.read_csv(os.path.join(x, 'logs.csv'), sep=';') for x in instance_logs_dirs]
    return paths, instance_train_logs, test_logs


def get_logs(log_dir, model_name):
    return get_runs(log_dir, model_name)



