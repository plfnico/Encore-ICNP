import os
import sys
import time 
import datetime
import argparse
import numpy as np
import pandas as pd
from model import SizeToHidden, GRU, Sequential, WeightNLLLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
if '/mnt/ssd1/encore/open-source' not in sys.path: sys.path.insert(0, '/mnt/ssd1/encore/open-source')
from utils.initialization import *
from utils.distribution_utils import *
from utils.eval import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_trainset(data, block_size, n_size, n_interval, max_len, min_samples, device):
    """
    Generate training set.

    Args:
        data (list): List of sequences.
        block_size (int): Size of sequence blocks.
        n_size (int): Number of size intervals.
        n_interval (int): Number of interval intervals.
        max_len (int): Maximum length of sequences.
        min_samples (int): Minimum number of samples.
        device (torch.device): Device for data placement.

    Returns:
        list: List of training samples.
    """
    trainset = []
    for seq in data:
        seq_trainset = []
        seq = np.append(seq[:-1], seq[0:block_size-1])
        size_seq = seq // n_interval
        interval_seq = seq % n_interval
        size_data = compute_probability_distribution(size_seq, n_size)
        interval_data = compute_probability_distribution(interval_seq, n_interval)
        permuted_seq = np.random.permutation(size_seq)
        jsd = 0
        for n in [2, 3, 4]:
            ngram_dist = compute_ngram_distribution(size_seq, n)
            permuted_ngram_dist = compute_ngram_distribution(permuted_seq, n)
            ngram_dist, permuted_ngram_dist = unify_distributions(ngram_dist, permuted_ngram_dist)
            js_divergence = compute_js_divergence(ngram_dist, permuted_ngram_dist)
            jsd += js_divergence * (2 ** (4 - n))
        weight = jsd * len(seq) / max_len
        weight = 0.01 if weight < 0.01 else weight
        seq = np.append([start_token], seq)
        num_samples = max(min_samples, len(seq) - block_size)
        for i in range(num_samples):
            index = i % (len(seq) - block_size)
            sequence_block = seq[index:index+block_size]
            target_sequence = seq[index+1:index+block_size+1]
            size_interval_data = np.concatenate((size_data, interval_data))
            seq_trainset.append([
                torch.from_numpy(sequence_block).to(device, non_blocking=True).long(),
                torch.from_numpy(target_sequence).to(device, non_blocking=True).long(),
                torch.from_numpy(size_interval_data).to(device, non_blocking=True).float(),
                torch.tensor(weight).to(device)
            ])
        trainset.extend(seq_trainset)
    return trainset
  

def train_batch(model, dloader, optimizer):
    """
    Train the model on a single batch of data.

    Args:
        model: The model to be trained.
        dloader: DataLoader providing batched data.
        optimizer: Optimizer used to update model parameters.

    Returns:
        float: The loss value for the current batch.
    """
    model.train()  # Set the model to training mode
    # Retrieve a batch of data
    seq_tensor, target_tensor, size_tensor, weight = next(iter(dloader))
    target_tensor = target_tensor.T  # Transpose the target tensor to match output shape
    optimizer.zero_grad()  # Clear previous gradients
    output = model((seq_tensor, size_tensor))  # Compute model output
    loss = loss_fn(output, target_tensor, weight)  # Calculate loss
    loss.backward()  # Backpropagate to compute gradients
    optimizer.step()  # Update model parameters
    return loss.item()  # Return the loss value


def get_dataloader(trainset, subset_size, batch_size, seed):
    """
    Generate a DataLoader from the training set.

    Args:
        trainset: Full dataset of training data.
        subset_size: Size of the subset to select from the trainset.
        batch_size: Number of samples per batch.
        seed: Random seed for deterministic sampling.

    Returns:
        DataLoader: Configured DataLoader for training process.
    """
    np.random.seed(seed)  # Set the random seed
    subset_size = min(subset_size, len(trainset))  # Ensure subset size does not exceed trainset size
    # Randomly select indices for the subset
    ran_index = np.random.choice(len(trainset), subset_size, replace=False)
    # Build the subset based on selected indices
    subset_trainset = [trainset[i] for i in ran_index]
    # Create and return the DataLoader
    return DataLoader(subset_trainset, batch_size=batch_size, shuffle=True)


def train_model(model, lr, plot_every, min_epoch, max_epoch, subset_size):
    """
    Main function to train the model.

    Args:
        model: The model to be trained.
        lr: Learning rate for the optimizer.
        plot_every: How often to print progress (every N epochs).
        max_epoch: Maximum number of epochs to train.
        subset_size: Size of the subset of training data to use in each epoch.

    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Initialize the optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)  # Learning rate scheduler
    s_time = time.time()  # Record start time
    avg_loss = 0  # Accumulate loss
    # Obtain a DataLoader
    train_loader = get_dataloader(trainset, subset_size, batch_size, seed=0)
    # Training loop
    for epoch in range(max_epoch + 1):
        loss = train_batch(model, train_loader, optimizer)  # Train a batch and get loss
        avg_loss += loss  # Accumulate loss
        # Print progress every plot_every epochs
        if epoch and epoch % plot_every == 0:
            print(f'epoch: {epoch}, loss: {avg_loss / plot_every:.4f}, lr={optimizer.param_groups[0]["lr"]:.2e}, time: {int(time.time() - s_time) // 60}min {int(time.time() - s_time) % 60}sec')
            sys.stdout.flush()
            if epoch >= min_epoch and avg_loss / plot_every < early_stop_loss:  # Early stopping if loss is below a threshold
                print(f'early stopping with loss {avg_loss / plot_every:.4f} at epoch {epoch}')
                break
            avg_loss = 0
            torch.cuda.empty_cache()  # Clear CUDA cache
            # Obtain a new DataLoader for the next round of training
            train_loader = get_dataloader(trainset, subset_size, batch_size, seed=epoch)
        if optimizer.param_groups[0]['lr'] > 1e-4:
            scheduler.step() # Adjust learning rate


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
    size_cdf = pd.read_csv('./data/cdf/size_cdf.csv')
    interval_cdf = pd.read_csv('./data/cdf/interval_cdf.csv')
    n_size = len(size_cdf) - 1
    n_interval = len(interval_cdf) - 1
    start_token = n_size * n_interval
    file = app + '.txt'
    block_size = 30
    batch_size = 64
    min_samples = 200

    data = get_data(size_dir, interval_dir, file, n_interval)
    max_len = max(len(seq) for seq in data) + block_size - 1
    trainset = get_trainset(data, block_size, n_size, n_interval, max_len, min_samples, device)

    gru_params = {
        'hidden_size': 256,
        'n_layer': 2,
        'embed_size': 64,
        'input_size': n_size * n_interval + 1,
    }
    s2h_params = {
        'n_layer': 2,
        'hidden_size': 256,
        'input_size': n_size + n_interval,
        'hidden_dims': [128, 256]
    }
    model = Sequential(gru_params, s2h_params).to(device)
    loss_fn = WeightNLLLoss().to(device)
    early_stop_loss = 0.01
    train_model(model, lr=1e-3, plot_every=10000, min_epoch=100000, max_epoch=200000, subset_size=50000)

    ckpt = {
        'model': model.state_dict(),
        'gru_params': gru_params,
        's2h_params': s2h_params,
        'app': app,
    }
    torch.save(ckpt, 'checkpoints/{date}/{app}.pt'.format(date=model_dir, app=app))
    print('Training finished with app={} in {}-{}-{} {}:{}:{:02d}'.format(app, datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second))