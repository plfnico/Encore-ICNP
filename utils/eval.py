import numpy as np
import pandas as pd
from collections import Counter
import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import math


def compute_ngram_distribution(sequence: list, n: int = 2) -> dict:
    """
    compute n-gram distribution of a sequence
    Parameters:
        sequence (list): input sequence
        n (int): n in n-gram, default is 2

    Returns:
        dict: n-gram distribution
    """
    ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    ngram_counts = Counter(ngrams)
    total_count = sum(ngram_counts.values())
    ngram_dist = {k: v / total_count for k, v in ngram_counts.items()}
    return ngram_dist


def unify_distributions(dist1: dict, dist2: dict) -> tuple:
    """
    combine two distributions into one, with the same keys
    Parameters:
        dist1 (dict): first distribution
        dist2 (dict): second distribution
    Returns:
        unified_dist1 (dict): first unified distribution
        unified_dist2 (dict): second unified distribution
    """
    all_keys = set(dist1.keys()) | set(dist2.keys())
    unified_dist1 = {key: dist1.get(key, 0) for key in all_keys}
    unified_dist2 = {key: dist2.get(key, 0) for key in all_keys}
    return unified_dist1, unified_dist2
    

def compute_js_distance(dist1: dict, dist2: dict) -> float:
    """
    compute Jensen-Shannon distance between two distributions

    Parameters:
        dist1 (dict): the first distribution
        dist2 (dict): the second distribution

    Returns:
        float: Jensen-Shannon distance
    """
    unified_dist1, unified_dist2 = unify_distributions(dist1, dist2)
    js_distance = jensenshannon(list(unified_dist1.values()), list(unified_dist2.values()), base=np.e)
    return js_distance



def compute_js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """
    计算两个离散概率分布的 JS 散度。
    
    参数:
        p, q: 字典格式的分布，键为事件名，值为概率（需已归一化）。
    
    返回:
        JS 散度（以 2 为底的对数，单位：bits）。
    """
    p,q = unify_distributions(p,q)
    # 所有事件的并集
    all_events = set(p.keys()) | set(q.keys())
    
    # 创建中间分布 M = 0.5 * (P + Q)
    m = {event: (p.get(event, 0.0) + q.get(event, 0.0)) * 0.5 for event in all_events}
    
    # 计算 KL(P || M)
    kl_pm = 0.0
    for event in p:
        p_i = p[event]
        m_i = m[event]
        # 仅当 p_i > 0 时计算（避免 log(0)）
        if p_i > 0.0:
            kl_pm += p_i * math.log2(p_i / m_i)
    
    # 计算 KL(Q || M)
    kl_qm = 0.0
    for event in q:
        q_i = q[event]
        m_i = m[event]
        # 仅当 q_i > 0 时计算
        if q_i > 0.0:
            kl_qm += q_i * math.log2(q_i / m_i)
    
    # JS = 0.5 * (KL(P || M) + KL(Q || M))
    return (kl_pm + kl_qm) * 0.5



def cramer_dis_matrix(distributions_x: np.ndarray, distributions_y: np.ndarray) -> np.ndarray:
    """
    Computes the Cramer distance matrix between two sets of distributions.

    Args:
        distributions_x (np.ndarray): The first set of distributions.
        distributions_y (np.ndarray): The second set of distributions.

    Returns:
        np.ndarray: The Cramer distance matrix.
    """
    cdf_x = np.cumsum(distributions_x, axis=1)
    cdf_y = np.cumsum(distributions_y, axis=1)
    abs_diffs = np.abs(cdf_x[:, np.newaxis, :] - cdf_y[np.newaxis, :, :])
    cramer_distances = np.sum(abs_diffs, axis=-1) / distributions_x.shape[1]
    return cramer_distances


def get_model_params(model: nn.Module) -> tuple:
    """
    Get total and trainable parameters of a model.
    Args:
        model (nn.Module): Model.
    Returns:
        int: Total parameters.
        int: Trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def plot_cdf(data: np.ndarray, label: str = None, color: str = 'LightBlue', 
             alpha: float = 1.0, linewidth: float = 2.0, linestyle: str = '-') -> plt.Line2D:
    data.sort()
    yvals = np.arange(len(data)) / len(data)
    kwargs = {'color': color, 'alpha': alpha, 'linewidth': linewidth, 'linestyle': linestyle}
    if label is not None:
        kwargs['label'] = label
    line, = plt.plot(data, yvals, **kwargs)
    return line