import os
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchsummary import summary
from typing import List, Tuple
from torch.nn import functional as F
from model import EncoreSequential, WeightNLLLoss
if '..' not in os.sys.path:os.sys.path.append('..')
from utils.dist import get_probability_distributions_from_sequence,compute_ngram_distribution, compute_js_distance
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


def get_trainset(pair_index_file, block_size, n_size, n_interval, min_samples, device):
    pair_index = []
    with open(pair_index_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pair_index.append([int(x) for x in line.strip().split(',')])
    
    trainset = []
    seqs = []
    size_probs, interval_probs = [], []
    for i, seq in enumerate(pair_index[0:100]):
        seq_trainset = []
        seq = np.concatenate((seq[:-1], seq[:1]))
        seqs.append(seq)
        size_prob, interval_prob = get_probability_distributions_from_sequence(seq, n_size, n_interval)
        size_probs.append(size_prob)
        interval_probs.append(interval_prob)
        size_seq = seq // n_interval
        permuted_seq = np.random.permutation(size_seq)
        size_pairs = compute_ngram_distribution(size_seq, 2)
        permuted_pairs = compute_ngram_distribution(permuted_seq, 2)
        jsd = compute_js_distance(size_pairs, permuted_pairs) ** 2
        jsd = jsd if jsd > 0.1 else 0.1
        weight = jsd * np.log10(len(seq))

        num_samples = max(min_samples, len(seq) - block_size)
        seq = np.append([start_token], seq)
        for i in range(num_samples):
            index = i % (len(seq) - block_size)
            sequence_block = seq[index:index+block_size]
            target_sequence = seq[index+1:index+block_size+1]
            probs = np.concatenate((size_prob, interval_prob))

            seq_trainset.append([
                torch.from_numpy(sequence_block).to(device, non_blocking=True).long(),
                torch.from_numpy(target_sequence).to(device, non_blocking=True).long(),
                torch.from_numpy(probs).to(device, non_blocking=True).float(),
                torch.tensor([weight], device=device, dtype=torch.float)
            ])
        trainset.extend(seq_trainset)
    size_probs, interval_probs = np.array(size_probs), np.array(interval_probs)
    return trainset, size_probs, interval_probs, seqs


def get_dataloader(trainset, subset_size, batch_size, seed):
    np.random.seed(seed)  
    subset_size = min(subset_size, len(trainset))  
    ran_index = np.random.choice(len(trainset), subset_size, replace=False)
    subset_trainset = [trainset[i] for i in ran_index]
    return DataLoader(subset_trainset, batch_size=batch_size, shuffle=True)


def get_model_params(model: nn.Module) -> tuple:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for seq_tensor, target_tensor, prob_tensor, weight_tensor in train_loader:
        optimizer.zero_grad()
        output = model(seq_tensor, prob_tensor)
        loss = criterion(output, target_tensor, weight_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(seq_tensor)
    return total_loss / len(train_loader.dataset)


def train(model: nn.Module, trainset: list, subset_size:int, batch_size:int, optimizer:torch.optim.Optimizer, n_epochs:int, plot_every:int, save_every:int, model_dir:str):
    start_time = time.time()
    criterion = WeightNLLLoss()
    avg_loss = 0
    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        train_loader = get_dataloader(trainset, subset_size, batch_size, epoch)
        loss = train_epoch(model, train_loader, optimizer, criterion)
        avg_loss += loss

        if epoch % plot_every == 0:
            avg_loss /= plot_every
            print(f'Epoch: {epoch}/{n_epochs}, Epoch time: {time.time() - epoch_start_time:.2f}s, Total time: {time.time() - start_time:.2f}s, Epoch Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}')
            print('-' * 80)
            sys.stdout.flush()
            avg_loss = 0

        if epoch % save_every == 0:
            model_save_path = os.path.join(model_dir, f'encore_transformer_{epoch}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')
            sys.stdout.flush()

    print(f'Training finished after {n_epochs} epochs in {time.time() - start_time:.2f}s')


if __name__ == '__main__':
    dataset = 'icbc'
    data_dir = f'/mnt/ssd1/hsj/encore/{dataset}'
    size_cdf_file = os.path.join(data_dir, f'{dataset}_cdf', f'{dataset}_size_cdf_coarse.csv')
    interval_cdf_file = os.path.join(data_dir, f'{dataset}_cdf', f'{dataset}_interval_cdf_coarse.csv')
    size_cdf = pd.read_csv(size_cdf_file)
    interval_cdf = pd.read_csv(interval_cdf_file)
    n_size, n_interval = len(size_cdf), len(interval_cdf)
    start_token = n_size * n_interval
    print(f'n_size: {n_size}, n_interval: {n_interval}, start_token: {start_token}')

    block_size = 16
    min_samples = 200
    pair_index_file = os.path.join(data_dir, f'{dataset}_pair_index.txt')
    trainset, size_probs, interval_probs, seqs = get_trainset(pair_index_file, block_size, n_size, n_interval, min_samples, device)
    print(f'Trainset size: {len(trainset)}')

    model = EncoreSequential(n_size, n_interval, 512, block_size, [128, 256, 512], device).to(device)
    get_model_params(model)

    subset_size = 64000
    batch_size = 256
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_epochs = 2000
    plot_every = 10
    save_every = 100
    model_dir = '/mnt/ssd1/hsj/encore/icbc/model/transformer-01-09/'
    os.makedirs(model_dir, exist_ok=True)
    train(model, trainset, subset_size, batch_size, optimizer, n_epochs, plot_every, save_every, model_dir)