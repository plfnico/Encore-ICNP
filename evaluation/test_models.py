import os
import sys
import time 
import numpy as np
import pandas as pd
from tqdm import tqdm
if '/home/Encore-ICNP' not in sys.path: sys.path.insert(0, '/home/Encore-ICNP')
from encore.model import Decoder, Sequential
from utils.initialization import *
from utils.distribution_utils import *
from utils.eval import *
from evaluation.generation_utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dists(data: list, n_size: int, n_interval: int, block_size: int) -> tuple:
    size_dists, interval_dists = [], []
    for seq in data:
        seq = np.append(seq[:-1], seq[0:block_size-1])
        size_dist, interval_dist = get_probability_distributions_from_sequence(seq, n_size, n_interval)
        size_dists.append(size_dist)
        interval_dists.append(interval_dist)
    return np.array(size_dists), np.array(interval_dists)


def get_dist_error(dists_ori, dists_gen):
    size_dists_ori, interval_dists_ori = dists_ori
    size_dists_gen, interval_dists_gen = dists_gen
    size_cramer_distance = cramer_dis_matrix(size_dists_ori, size_dists_gen)
    interval_cramer_distance = cramer_dis_matrix(interval_dists_ori, interval_dists_gen)
    cramer_distance = size_cramer_distance + interval_cramer_distance
    return {'size_dist': size_cramer_distance, 'interval_dist': interval_cramer_distance, 'total_dist': cramer_distance}


def test_encore_dist(data: list, decoder: torch.nn.Module, norm_loads: np.ndarray, n_size: int, n_interval: int, block_size: int, device) -> dict:
    dists_ori = get_dists(data, n_size, n_interval, block_size)
    dists_encore = gen_dists_encore(decoder, norm_loads, device)
    error_encore = get_dist_error(dists_ori, dists_encore)
    return error_encore


def test_sequence_gen(sequence, sequence_gen, n_interval):
    size_sequence = sequence // n_interval
    interval_sequence = sequence % n_interval
    ori_size_interval = list(zip(size_sequence[:-1], interval_sequence[:-1]))
    size_sequence_gen = sequence_gen // n_interval
    interval_sequence_gen = sequence_gen % n_interval
    gen_size_interval = list(zip(size_sequence_gen[:-1], interval_sequence_gen[:-1]))
    jsds = {}
    ori_size_interval_dist = Counter(ori_size_interval)
    gen_size_interval_dist  = Counter(gen_size_interval)
    jsds['size_interval'] = compute_js_divergence(ori_size_interval_dist, gen_size_interval_dist)
    for n in [2, 3, 4]:
        n_gram_ori = compute_ngram_distribution(size_sequence, n)
        n_gram_gen = compute_ngram_distribution(size_sequence_gen, n)
        jsds['size_{}'.format(n)] = compute_js_divergence(n_gram_ori, n_gram_gen)
    return jsds


def save_results(errors, jsds, algo, app):
    directory = './results/{app}/{algo}'.format(app=app, algo=algo)
    os.makedirs(directory, exist_ok=True)
    for key, value in errors.items():
        result_file = f'{directory}/{key}.txt'
        np.savetxt(result_file, value, delimiter=',', fmt='%.3f')
    jsds.to_csv(f'{directory}/jsds.csv', index=False, float_format='%.4f')


if __name__ == "__main__":
    os.chdir('/home/Encore-ICNP/')
    size_dir = './data/size/'
    interval_dir = './data/interval/'
    metadata_dir = './data/metadata/'
    size_cdf = pd.read_csv('./data/cdf/size_cdf.csv')
    interval_cdf = pd.read_csv('./data/cdf/interval_cdf.csv')
    n_size = len(size_cdf) - 1
    n_interval = len(interval_cdf) - 1
    files = os.listdir(size_dir)
    block_size = 30

    model_date = '2024-5-15-9'
    for file in files:
        app = file.strip('.txt')
        s_time = time.time()
        print('start testing {}'.format(app))

        cvae_path = 'cvae-{date}/{app}'.format(date=model_date, app=app)
        gru_path = 'gru-{date}/{app}'.format(date=model_date, app=app)
        data = get_data(size_dir, interval_dir, file, n_interval)
        loads = get_loads(metadata_dir + app + '.csv', normalize=False)
        norm_loads = get_loads(metadata_dir + app + '.csv', normalize=True)
        dists_ori = get_dists(data, n_size, n_interval, block_size)
        size_dists_ori, interval_dists_ori = dists_ori
        num_samples = len(data)

        decoder = load_cvae(cvae_path, n_size, n_interval, device)
        sequential = load_gru(gru_path, device)

        dists, errors = {}, {}
        dists['common'] = gen_dists_common(data, loads, size_cdf, interval_cdf)
        dists['lomas'], lomas_word_prob = gen_dists_lomas('checkpoints/lomas', n_size, n_interval, app)
        dists['encore'] = gen_dists_encore(decoder, norm_loads, device)
        for algo in ['common', 'lomas', 'encore']:
            errors[algo] = get_dist_error(dists_ori, dists[algo])

        jsds = {}
        for algo in ['common', 'lomas', 'encore']:
            jsds[algo] = []
        for i in range(len(data)):
            sequence_encore = generate_sequence_encore(sequential, size_dists_ori[i], interval_dists_ori[i], block_size, device, 1000, seed=42)
            sequence_common = generate_sequence_sample(data[i], n_interval)
            sequence_lomas = generate_sequence_lomas(lomas_word_prob[i], 1000)
            jsds['common'].append(test_sequence_gen(data[i], sequence_common, n_interval))
            jsds['encore'].append(test_sequence_gen(data[i], sequence_encore, n_interval))
            jsds['lomas'].append(test_sequence_gen(data[i], sequence_lomas, n_interval))
        for algo in ['common', 'lomas', 'encore']:
            jsds[algo] = pd.DataFrame(jsds[algo])   
            save_results(errors[algo], jsds[algo], algo, app)

        print('finish testing {} in {:.2f} seconds'.format(app, time.time() - s_time))
        sys.stdout.flush()

