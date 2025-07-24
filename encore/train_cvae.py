import os
import sys
import time
import datetime  
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchsummary import summary
from typing import List, Tuple
from torch.nn import functional as F
from model import Encoder, Decoder, reparameterize
if '/mnt/ssd1/encore/open-source' not in sys.path: sys.path.insert(0, '/mnt/ssd1/encore/open-source')
from utils.initialization import *
from utils.distribution_utils import *
from utils.eval import *
from evaluation.test_models import test_encore_dist
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(encoder: nn.Module, decoder: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, kld_weight: float) -> (float, float, float):
    encoder.train()
    decoder.train()

    # Get the next batch of data
    size_dist_tensor, interval_dist_tensor, load_tensor = next(iter(dataloader))
        
    # Reset gradients to zero
    optimizer.zero_grad()

    # Forward pass through the encoder and decoder
    mu, log_var = encoder(torch.cat([size_dist_tensor, interval_dist_tensor], dim=1), load_tensor)
    z = reparameterize(mu, log_var)
    size_recon, interval_recon = decoder(z, load_tensor)

    # Calculate losses
    recon_loss = F.l1_loss(size_recon, size_dist_tensor) + F.l1_loss(interval_recon, interval_dist_tensor)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / size_dist_tensor.size(0)
    total_loss = recon_loss + kld_weight * kld_loss

    # Backward pass and optimize
    total_loss.backward()
    optimizer.step()

    # Accumulate losses and the number of examples
    epoch_loss = total_loss.item() 
    epoch_kld = kld_loss.item() 
    epoch_recon = recon_loss.item() 

    return epoch_loss, epoch_recon, epoch_kld


def get_trainset(data, loads):
    size_dists, interval_dists = [], []
    for seq in data:
        # Append the beginning of the sequence to the end to ensure continuity for block processing
        seq = np.append(seq[:-1], seq[0:block_size-1])
        # Compute the size and interval sequences
        size_seq = seq // n_interval
        interval_seq = seq % n_interval
        # Compute and store the probability distributions for sizes and intervals
        size_dists.append(compute_probability_distribution(size_seq, n_size))
        interval_dists.append(compute_probability_distribution(interval_seq, n_interval))
    
    # Convert the lists of distributions into numpy arrays
    size_dists, interval_dists = np.array(size_dists), np.array(interval_dists)
    # Convert numpy arrays into tensors and move them to the specified device
    size_dists_tensor = torch.tensor(size_dists, dtype=torch.float32).to(device)
    interval_dists_tensor = torch.tensor(interval_dists, dtype=torch.float32).to(device)
    load_tensor = torch.from_numpy(loads).to(device).float()
    # Create a TensorDataset from the size and interval distribution tensors
    dataset = torch.utils.data.TensorDataset(size_dists_tensor, interval_dists_tensor, load_tensor)
    
    return dataset, size_dists, interval_dists


if __name__ == "__main__":
    # os.chdir('/mnt/ssd1/encore')#TODO:change to ur own dir

    parser = argparse.ArgumentParser()
    parser.add_argument('--app', '-a', type=str, help='Name of the app.', dest='app', required=True)
    parser.add_argument('--dir', '-d', type=str, help='Directory of the model.', dest='model_dir', required=True)
    args = parser.parse_args()
    app = args.app
    model_dir = args.model_dir

    print('start training app={} in {}-{}-{} {}:{}:{:02d}'.format(app, datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))
    sys.stdout.flush()

    size_dir = './data/size/'
    interval_dir = './data/interval/'
    metadata_dir = './data/metadata/'
    size_cdf = pd.read_csv('./data/cdf/size_cdf.csv')
    interval_cdf = pd.read_csv('./data/cdf/interval_cdf.csv')
    n_size = len(size_cdf) - 1
    n_interval = len(interval_cdf) - 1
    file = app + '.txt'
    block_size = 30
    batch_size = 64

    data = get_data(size_dir, interval_dir, file, n_interval)
    norm_loads = get_loads(metadata_dir + app + '.csv', normalize=True)
    trainset, size_dists, interval_dists = get_trainset(data, norm_loads)
    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # initialize model
    kld_weight = 1e-5
    lr = 1e-3
    latent_dim = 32
    hidden_dims = [256, 512, 256]
    n_condition = 1
    encoder = Encoder(n_size, n_interval, n_condition, hidden_dims, latent_dim).to(device)
    hidden_dims.reverse()
    decoder = Decoder(n_size, n_interval, n_condition, hidden_dims, latent_dim).to(device)
    print('encoder parameters: {}, decoder parameters: {}'.format(get_model_params(encoder)[0], get_model_params(decoder)[0]))
    optimizer = torch.optim.Adam([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.9)

    # train model
    min_epoch = 50000
    max_epoch = 300000
    test_every = 10000
    s_time = time.time()
    early_stop_recon = 0.001
    max_kld_weight = 1e-2
    update_loss = 0.002
    avg_loss, avg_recon, avg_kld = 0, 0, 0
    for epoch in range(max_epoch + 1):
        loss, recon, kld = train_model(encoder, decoder, dataloader, optimizer, kld_weight)
        avg_loss += loss
        avg_recon += recon
        avg_kld += kld
        if optimizer.param_groups[0]['lr'] > 1e-4:
            scheduler.step()
        if epoch and epoch % test_every == 0:
            avg_loss /= test_every
            avg_recon /= test_every
            avg_kld /= test_every
            cvae_errors = test_encore_dist(data, decoder, norm_loads, n_size, n_interval, block_size, device)
            accuracy, coverage = {}, {}
            for key, error in cvae_errors.items():
                accuracy[key] = np.sort(error.min(axis=1))
                coverage[key] = np.sort(error.min(axis=0))
            acc_percentile = {key: np.percentile(val, [50, 75, 90, 95, 99]) for key, val in accuracy.items()}
            cov_percentile = {key: np.percentile(val, [50, 75, 90, 95, 99]) for key, val in coverage.items()}
            print('epoch: {:d}, loss: {:.4f}, recon: {:.4f}, kld: {:.4f}, lr: {:.2e}, time: {:.2f}s'.format(epoch, avg_loss, avg_recon, avg_kld, optimizer.param_groups[0]['lr'], time.time() - s_time))
            print('accuracy: {:.2e}/{:.2e}/{:.2e}/{:.2e}/{:.2e}'.format(*acc_percentile['total_dist']))
            print('coverage: {:.2e}/{:.2e}/{:.2e}/{:.2e}/{:.2e}'.format(*cov_percentile['total_dist']))
            if epoch >= min_epoch and kld_weight >= max_kld_weight and avg_recon <= early_stop_recon:
                print(f'early stopping with loss={avg_loss} and kld_weight={kld_weight}')
                break
            if avg_recon <= update_loss:
                kld_weight *= 2
                kld_weight = min(kld_weight, max_kld_weight)
                print('kld_weight={:.2e}'.format(kld_weight))
            print('-' * 100)
            sys.stdout.flush()
            avg_loss, avg_recon, avg_kld = 0, 0, 0
    ckpt = {
        'params': {'latent_dim': latent_dim, 'hidden_dims': hidden_dims, 'n_condition': n_condition},
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'app': app
    }
    torch.save(ckpt, 'checkpoints/{dir}/{app}.pt'.format(dir=model_dir, app=app))
    print('Training finished with app={} in {}-{}-{} {}:{}:{:02d}'.format(app, datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))
    sys.stdout.flush()

 