import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import jensenshannon


def compute_probability_distribution(sequence: np.ndarray, n_count: int) -> np.ndarray:
    counts = np.bincount(sequence, minlength=n_count)
    prob = counts / np.sum(counts)
    return prob


def get_probability_distributions_from_sequence(sequence: np.ndarray, n_size: int, n_interval: int) -> tuple:
    size_seq = sequence // n_interval
    interval_seq = sequence % n_interval
    return compute_probability_distribution(size_seq, n_size), compute_probability_distribution(interval_seq, n_interval)


def compute_ngram_distribution(sequence: list, n: int = 2) -> dict:
    ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    ngram_counts = Counter(ngrams)
    total_count = sum(ngram_counts.values())
    ngram_dist = {k: v / total_count for k, v in ngram_counts.items()}
    return ngram_dist


def unify_distributions(dist1: dict, dist2: dict) -> tuple:
    all_keys = set(dist1.keys()) | set(dist2.keys())
    unified_dist1 = {key: dist1.get(key, 0) for key in all_keys}
    unified_dist2 = {key: dist2.get(key, 0) for key in all_keys}
    return unified_dist1, unified_dist2
    

def compute_js_distance(dist1: dict, dist2: dict) -> float:
    unified_dist1, unified_dist2 = unify_distributions(dist1, dist2)
    js_distance = jensenshannon(list(unified_dist1.values()), list(unified_dist2.values()), base=np.e)
    return js_distance
