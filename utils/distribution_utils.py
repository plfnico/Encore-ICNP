import os
import numpy as np
import pandas as pd


def compute_probability_distribution(sequence: np.ndarray, n_count: int) -> np.ndarray:
    """
    compute probability distribution of a sequence

    Parameters:
        sequence (list): input sequence
        n_count (int): number of unique elements in the sequence

    Returns:
        np.array: probability distribution
    """
    counts = np.bincount(sequence, minlength=n_count)
    prob = counts / np.sum(counts)
    return prob


def get_probability_distributions_from_sequence(sequence: np.ndarray, n_size: int, n_interval: int) -> tuple:
    """
    Computes the probability distribution for size and interval from a sequence.

    Args:
        sequence (np.ndarray): The input sequence array.
        n_size (int): The number of size classes.
        n_interval (int): The number of interval classes.

    Returns:
        tuple: Two np.ndarrays representing the probability distributions for size and interval.
    """
    size_seq = sequence // n_interval
    interval_seq = sequence % n_interval
    return compute_probability_distribution(size_seq, n_size), compute_probability_distribution(interval_seq, n_interval)


def normalize_distribution_excluding_minima(dist: np.ndarray) -> np.ndarray:
    """
    Normalizes a distribution by setting values below a threshold to zero and 
    then normalizing the distribution so that the sum equals 1.

    Args:
        dist (np.ndarray): The input distribution array.

    Returns:
        np.ndarray: The normalized distribution with minima excluded.
    """
    dist[dist < 1e-3] = 0
    dist /= np.sum(dist)
    return dist