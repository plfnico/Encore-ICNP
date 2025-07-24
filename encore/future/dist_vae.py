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
from model import EncoreVAE, CustomVAELoss
if '..' not in os.sys.path:os.sys.path.append('..')
from utils.dist import get_probability_distributions_from_sequence
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def get_trainset(pair_index_file:str, n_size:int, n_interval:int, device:torch.device) -> Tuple[torch.utils.data.TensorDataset, np.ndarray, np.ndarray]:
    pair_index = []
    with open(pair_index_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pair_index.append([int(x) for x in line.strip().split(',')])

    size_probs, interval_probs = [], []
    for seq in tqdm(pair_index):
        seq = np.concatenate((seq[:-1], seq[:1]))
        size_prob, interval_prob = get_probability_distributions_from_sequence(seq, n_size, n_interval)
        size_probs.append(size_prob)
        interval_probs.append(interval_prob)

    size_probs, interval_probs = np.array(size_probs), np.array(interval_probs)
    size_probe_tensor = torch.tensor(size_probs, dtype=torch.float32).to(device)
    interval_probe_tensor = torch.tensor(interval_probs, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(size_probe_tensor, interval_probe_tensor)
    return dataset, size_probs, interval_probs


def train_epoch(model:nn.Module, dataloader:DataLoader, optimizer:torch.optim.Optimizer, criterion:nn.Module, device:torch.device) -> Tuple[float, float, float]:
    model.train()
    running_loss, running_recon_loss, running_kld_loss = 0.0, 0.0, 0.0
    for size_batch, interval_batch in dataloader:
        size_batch, interval_batch = size_batch.to(device), interval_batch.to(device)
        input_tensor = torch.cat([size_batch, interval_batch], dim=1)
        size_recon, interval_recon, mu, log_var = model(input_tensor)
        loss, recon_loss, kld_loss = criterion(size_recon, size_batch, interval_recon, interval_batch, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * size_batch.size(0)
        running_recon_loss += recon_loss.item() * size_batch.size(0)
        running_kld_loss += kld_loss.item() * size_batch.size(0)
    average_loss = running_loss / len(dataloader.dataset)
    average_recon_loss = running_recon_loss / len(dataloader.dataset)
    average_kld_loss = running_kld_loss / len(dataloader.dataset)
    return average_loss, average_recon_loss, average_kld_loss


def train(model:nn.Module, dataloader:DataLoader, optimizer:torch.optim.Optimizer, scheduler:torch.optim.lr_scheduler, n_epochs:int, plot_every:int, save_every:int, model_dir:str, init_kld_weight:float, kld_update_every:int, device:torch.device):
    start_time = time.time()
    kld_weight = init_kld_weight
    for epoch in range(1, n_epochs+1):
        epoch_start_time = time.time()
        criterion = CustomVAELoss(kld_weight)
        train_loss, train_recon_loss, train_kld_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        if scheduler is not None and scheduler.get_last_lr()[0] > 1e-4:
            scheduler.step()

        if epoch % plot_every == 0:
            print(f'Epoch: {epoch}/{n_epochs}, Epoch time: {time.time() - epoch_start_time:.2f}s, Total time: {time.time() - start_time:.2f}s, LR: {scheduler.get_last_lr()[0]:.2e}')
            print(f'Train Loss: {train_loss:.4f}, Train Recon Loss: {train_recon_loss:.4f}, Train KLD Loss: {train_kld_loss:.4f}')
            print('-' * 80)

        if epoch % save_every == 0:
            model_save_path = os.path.join(model_dir, f'encore_vae_{epoch}.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')

        if epoch % kld_update_every == 0:
            kld_weight = min(kld_weight * 10, 1e-3)
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

    pair_index_file = os.path.join(data_dir, f'{dataset}_pair_index.txt')
    trainset, size_probs, interval_probs = get_trainset(pair_index_file, n_size, n_interval, device)
    print(f'Trainset size: {len(trainset)}')

    batch_size = 512
    hidden_dims = [256, 512, 1024, 1024, 512, 256]
    latent_dim = 64
    model = EncoreVAE(n_size, n_interval, hidden_dims, latent_dim, device)
    summary(model, input_size=(n_size + n_interval,), device='cpu')
    model = model.to(device)

    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    size_batch, interval_batch = next(iter(dataloader))
    input_tensor = torch.cat([size_batch, interval_batch], dim=1)
    size_recon, interval_recon, mu, log_var = model(input_tensor)

    model_dir = f'/mnt/ssd1/hsj/encore/{dataset}/model/vae-01-10/'
    os.makedirs(model_dir, exist_ok=True)
    init_kld_weight = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.8)
    n_epochs = 50000
    plot_every = 100
    save_every = 10000
    kld_update_every = 10000

    train(model, dataloader, optimizer, scheduler, n_epochs, plot_every, save_every, model_dir, init_kld_weight, kld_update_every, device)


    