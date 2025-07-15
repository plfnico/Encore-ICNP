import os
import numpy as np
import pandas as pd
import torch

def get_data(size_dir: str, interval_dir: str, file: str, n_interval: int) -> list:
    """
    Retrieve and process data from files.

    Args:
        flowsize_dir (str): Directory path where flow size files are located.
        interval_dir (str): Directory path where interval files are located.
        file (str): File name.
        n_interval (int): Number of intervals.

    Returns:
        list: Processed data list.
    """
    data = []
    # Open flow size file and interval file
    with open(os.path.join(size_dir, file), 'r') as size_f, \
            open(os.path.join(interval_dir, file), 'r') as interval_f:
        # Iterate through lines of flow size and interval files, and process the data
        for size_line, interval_line in zip(size_f, interval_f):
            size = np.array(list(map(int, size_line.split(','))))
            interval = np.array(list(map(int, interval_line.split(','))))
            # Combine size and interval data and append to the list
            interval = np.concatenate((interval, [0]))
            data.append(size * n_interval + interval)
    return data


def get_loads(meta_file: str, normalize: bool) -> np.ndarray:
    """
    Retrieve load data from file, apply log10 transformation, and then
    min-max normalize each column.

    Args:
        load_file (str): File name containing load data.
        normalize (bool): Whether to normalize the data.
    
    Returns
        np.ndarray: Transformed and normalized load data.
    """

    metadata = pd.read_csv(meta_file)
    # loads = metadata[['load', 'mean_size', 'mean_interval']].values 
    loads = metadata[['load']].values 
    if normalize:
        loads = np.log10(loads)
        loads = (loads - loads.min(axis=0)) / (loads.max(axis=0) - loads.min(axis=0))
    return loads


def initialize_random_seeds(seed: int):
    """
    Initializes random seeds for NumPy and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value for random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)