import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torchsummary import summary
from torch.nn import functional as F
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import EncoreSequential, WeightNLLLoss
if '..' not in os.sys.path:os.sys.path.append('..')
from utils.dist import get_probability_distributions_from_sequence, compute_ngram_distribution, compute_js_distance
from typing import List, Tuple
import torch.multiprocessing as mp
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'


def get_trainset(pair_index_file, block_size, n_size, n_interval, min_samples):
    pair_index = []
    with open(pair_index_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pair_index.append([int(x) for x in line.strip().split(',')])
    
    trainset = []
    seqs = []
    size_probs, interval_probs = [], []
    start_token = n_size * n_interval
    for i, seq in enumerate(pair_index):
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
                torch.from_numpy(sequence_block).long(),
                torch.from_numpy(target_sequence).long(),
                torch.from_numpy(probs).float(),
                torch.tensor([weight]).float()
            ])
        trainset.extend(seq_trainset)
    size_probs, interval_probs = np.array(size_probs), np.array(interval_probs)
    return trainset, size_probs, interval_probs, seqs


def get_model_params(model: nn.Module) -> tuple:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for seq_tensor, target_tensor, prob_tensor, weight_tensor in train_loader:
        seq_tensor, target_tensor, prob_tensor, weight_tensor = seq_tensor.to(device), target_tensor.to(device), prob_tensor.to(device), weight_tensor.to(device)
        optimizer.zero_grad()
        output = model(seq_tensor, prob_tensor)
        loss = criterion(output, target_tensor, weight_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(seq_tensor)
    return total_loss / (len(train_loader.dataset) // 4)


def train(rank, world_size, device_ids):
    dataset = 'icbc'
    data_dir = f'/mnt/ssd1/hsj/encore/{dataset}'
    size_cdf_file = os.path.join(data_dir, f'{dataset}_cdf', f'{dataset}_size_cdf_coarse.csv')
    interval_cdf_file = os.path.join(data_dir, f'{dataset}_cdf', f'{dataset}_interval_cdf_coarse.csv')
    size_cdf = pd.read_csv(size_cdf_file)
    interval_cdf = pd.read_csv(interval_cdf_file)
    n_size, n_interval = len(size_cdf), len(interval_cdf)
    start_token = n_size * n_interval
    print(f'starting rank: {rank}, n_size: {n_size}, n_interval: {n_interval}, start_token: {start_token}')

    block_size = 16
    batch_size = 256
    min_samples = 100
    pair_index_file = os.path.join(data_dir, f'{dataset}_pair_index.txt')
    trainset, size_probs, interval_probs, seqs = get_trainset(pair_index_file, block_size, n_size, n_interval, min_samples)
    print(f'Rank: {rank}, Trainset size: {len(trainset)}')

    device = torch.device(f'cuda:{device_ids[rank]}')
    model = EncoreSequential(n_size, n_interval, 512, block_size, [128, 256, 512], device).to(device)
    total_params, trainable_params = get_model_params(model)
    print(f'Rank: {rank}, Total params: {total_params}, Trainable params: {trainable_params}')

    # load_model_dir = f'/mnt/ssd1/hsj/encore/{dataset}/model/transformer-01-09/'
    # load_model_path = os.path.join(load_model_dir, f'encore_transformer_200.pt')
    # if os.path.exists(load_model_path):
    #     model.load_state_dict(torch.load(load_model_path))
    #     print(f'Rank: {rank}, Model loaded from {load_model_path}')

    model = DDP(model, device_ids=[device_ids[rank]])
    sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_epochs = 200
    plot_every = 1
    save_every = 10
    model_dir = f'/mnt/ssd1/hsj/encore/{dataset}/model/transformer-01-10/'
    os.makedirs(model_dir, exist_ok=True)

    start_time = time.time()
    criterion = WeightNLLLoss()
    avg_loss = 0
    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        sampler.set_epoch(epoch)
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        avg_loss += loss

        if rank == 0 and epoch % plot_every == 0:
            avg_loss /= plot_every
            print(f'Epoch: {epoch}/{n_epochs}, Epoch time: {time.time() - epoch_start_time:.2f}s, Total time: {time.time() - start_time:.2f}s, Epoch Loss: {loss:.4f}, Avg Loss: {avg_loss:.4f}')
            print('-' * 80)
            sys.stdout.flush()
            avg_loss = 0

        if rank == 0 and epoch % save_every == 0:
            model_save_path = os.path.join(model_dir, f'encore_transformer_{epoch}.pt')
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')
            sys.stdout.flush()

    if rank == 0:
        print(f'Training finished after {n_epochs} epochs in {time.time() - start_time:.2f}s')


def setup(rank, world_size):
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '12355'      

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def run(rank, world_size, device_ids):
    setup(rank, world_size)
    train(rank, world_size, device_ids)
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = 4
    device_ids = [4, 5, 6, 7]
    mp.spawn(run, args=(world_size, device_ids), nprocs=world_size, join=True)
