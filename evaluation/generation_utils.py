import os
import sys
import time 
import numpy as np
import pandas as pd
from gensim import corpora
if '..' not in sys.path: sys.path.insert(0, '..')
from utils.initialization import *
from utils.distribution_utils import *
from utils.eval import *
# from encore.model import Decoder, Sequential


def gen_dists_encore(decoder: torch.nn.Module, loads: np.ndarray, device) -> tuple:
    num_samples = len(loads)
    latent_dim = decoder.latent_to_hidden.in_features
    load_tensor = torch.tensor(loads, dtype=torch.float32).to(device)
    z = torch.randn(num_samples, latent_dim).to(device)
    size_recon, interval_recon = decoder(z, load_tensor)
    size_recon = size_recon.squeeze().detach().cpu().numpy()
    interval_recon = interval_recon.squeeze().detach().cpu().numpy()
    size_recon = np.array([normalize_distribution_excluding_minima(size_recon[i]) for i in range(size_recon.shape[0])])
    interval_recon = np.array([normalize_distribution_excluding_minima(interval_recon[i]) for i in range(interval_recon.shape[0])])
    return size_recon, interval_recon
    

def sample_next(size_dist, interval_dist, start_token, block_size, model, device):
    n_size, n_interval = len(size_dist), len(interval_dist)
    model.eval()
    softmax = nn.Softmax(dim=2)
    size_tensor = torch.tensor(np.concatenate((size_dist, interval_dist))).unsqueeze(0).float().to(device)
    seq = [start_token]
    with torch.no_grad():
        for _ in range(block_size - 1):
            seq_tensor = torch.tensor(seq).long().unsqueeze(0).to(device)
            out = model((seq_tensor, size_tensor))
            out = softmax(out)[-1]
            p_size = out.detach().cpu().numpy().flatten()
            next_token = np.random.choice(n_size * n_interval + 1, p=p_size)
            seq.append(next_token)
    return seq[1:]


def generate_sequence_encore(model, size_dist, interval_dist, block_size, device, seq_len, seed=42):
    initialize_random_seeds(seed)
    n_size, n_interval = len(size_dist), len(interval_dist)
    start_token = n_size * n_interval
    seq_gen = []
    while len(seq_gen) < seq_len:
        next_block = sample_next(size_dist, interval_dist, start_token, block_size, model, device)
        seq_gen.extend(next_block)
        start_token = next_block[-1]
    return np.array(seq_gen)



def generate_sequence_sample(seq, n_interval, seed=42):
    np.random.seed(seed)
    size_seq = seq // n_interval
    interval_seq = seq % n_interval
    size_seq_permuted = np.random.permutation(size_seq)
    interval_seq_permuted = np.random.permutation(interval_seq)
    seq_gen = size_seq_permuted * n_interval + interval_seq_permuted
    return seq_gen


def generate_sequence_lomas(lomas_word_prob, seq_len, seed=42):
    np.random.seed(seed)
    return np.random.choice(len(lomas_word_prob), seq_len, p=lomas_word_prob)


def sample_sequence(sequence, cdf):
    sequence = np.array(sequence)
    random_values = np.random.rand(len(sequence))
    sampled_values = (cdf[sequence] + random_values * (cdf[sequence + 1] - cdf[sequence]))
    return sampled_values.tolist()


# def load_gru(model_path, device):
#     gru_path = './checkpoints/{model_path}.pt'.format(model_path=model_path)
#     gru_ckpt = torch.load(gru_path)
#     gru_params = gru_ckpt['gru_params']
#     s2h_params = gru_ckpt['s2h_params']
#     sequential = Sequential(gru_params, s2h_params).to(device)
#     sequential.load_state_dict(gru_ckpt['model'])
#     return sequential


# def load_cvae(model_path, n_size, n_interval, device):
#     cvae_path = './checkpoints/{model_path}.pt'.format(model_path=model_path)
#     cvae_ckpt = torch.load(cvae_path)
#     latent_dim = cvae_ckpt['params']['latent_dim']
#     hidden_dims = cvae_ckpt['params']['hidden_dims']
#     n_condition = cvae_ckpt['params']['n_condition']
#     decoder = Decoder(n_size, n_interval, n_condition, hidden_dims, latent_dim).to(device)
#     decoder.load_state_dict(cvae_ckpt['decoder'])
#     return decoder


def get_cdf(x, cdf):
    return np.sum([cdf <= x]) - 1


def gen_dists_common(data, loads, size_cdf, interval_cdf):
    n_size = len(size_cdf) - 1
    n_interval = len(interval_cdf) - 1
    all_data = np.concatenate(data)
    all_sizes = all_data // n_interval
    size_dist = compute_probability_distribution(all_sizes, n_size)
    size_index = np.random.choice(np.arange(n_size), size=len(all_sizes), p=size_dist)
    size_sequence = sample_sequence(size_index, size_cdf['size'].values)
    mean_size = np.mean(size_sequence)
    size_dists, interval_dists = [], []
    for i in range(len(loads)):
        size_dists.append(size_dist)
        mean_interval = mean_size / loads[i][0]
        interval_sequence = np.random.exponential(mean_interval, size=len(data[i]))
        interval_index = [get_cdf(interval, interval_cdf['interval'].values) for interval in interval_sequence]
        interval_index = np.clip(interval_index, 0, n_interval-1)
        interval_dist = compute_probability_distribution(interval_index, n_interval)
        interval_dists.append(interval_dist)
    return np.array(size_dists), np.array(interval_dists)


def gen_dists_lomas(model_dir, n_size, n_interval, app):
    dict_file = os.path.join(model_dir, 'dictionary', app + '.dict')
    topic_probs_file = os.path.join(model_dir, 'topic_probs', app + '.txt')
    topic_word_probs_file = os.path.join(model_dir, 'topic_word_probs', app + '.txt')
    dictionary = corpora.Dictionary.load(dict_file)
    topic_probs = np.loadtxt(topic_probs_file, delimiter=',')
    topic_word_probs = np.loadtxt(topic_word_probs_file, delimiter=',')
    topic_word_probs = topic_word_probs / topic_word_probs.sum(axis=1)[:, np.newaxis]

    num_samples = topic_probs.shape[0]
    word_prob = np.dot(topic_probs, topic_word_probs)
    word_prob = word_prob / word_prob.sum(axis=1)[:, np.newaxis]
    word_prob_matrix = word_prob.reshape(num_samples, n_size, n_interval)
    size_dists = np.sum(word_prob_matrix, axis=2)
    interval_dists = np.sum(word_prob_matrix, axis=1)
    return (size_dists, interval_dists), word_prob
