import torch
from torch import nn, Tensor
from typing import List
from torch.nn import functional as F


class SizeToHidden(nn.Module):
    def __init__(self, n_layer, hidden_size, input_size, hidden_dims):
        """
        Initialize SizeToHidden module.

        Args:
            n_layer (int): Number of layers.
            hidden_size (int): Size of hidden layers.
            input_size (int): Size of input.
            hidden_dims (list): List of sizes of hidden layers.
        """
        super(SizeToHidden, self).__init__()
        self.n_layer = n_layer
        layers = []
        in_dim = input_size
        for h_dim in hidden_dims:
            layers.append(nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU()
            ))
            in_dim = h_dim
        self.lins = nn.ModuleList(layers)
        self.output = nn.Linear(in_dim, hidden_size)

    def forward(self, x):
        """
        Forward pass SizeToHidden module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        for lin in self.lins:
            x = lin(x)
        x = self.output(x)
        x = x.unsqueeze(0).repeat(self.n_layer, 1, 1)
        return x


class GRU(nn.Module):
    def __init__(self, hidden_size, n_layer, embed_size, input_size):
        """
        Initialize GRU module.

        Args:
            hidden_size (int): Size of hidden layers.
            n_layer (int): Number of layers.
            embed_size (int): Size of embedding.
            input_size (int): Size of input.
        """
        super(GRU, self).__init__()
        self.gru = nn.GRU(embed_size, hidden_size, n_layer)
        self.h2o = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.embedding = nn.Embedding(input_size, embed_size)

    def forward(self, x, hidden):
        """
        Forward pass GRU module.

        Args:
            x (Tensor): Input tensor.
            hidden (Tensor): Hidden state tensor.

        Returns:
            Tensor: Output tensor.
            Tensor: Hidden state tensor.
        """
        x = self.embedding(x).permute(1, 0, 2)
        out, hidden = self.gru(x, hidden)
        out = self.h2o(F.relu(out))
        out = self.softmax(out)
        return out, hidden


class Sequential(nn.Module):
    def __init__(self, gru_params, s2h_params):
        """
        Initialize Model module.

        Args:
            gru_params (dict): Parameters for GRU module.
            s2h_params (dict): Parameters for SizeToHidden module.
        """
        super(Sequential, self).__init__()
        self.gru = GRU(**gru_params)
        self.s2h = SizeToHidden(**s2h_params)

    def forward(self, x):
        """
        Forward pass Model module.

        Args:
            x (tuple): Input tuple containing seq_tensor and size_interval.

        Returns:
            Tensor: Output tensor.
        """
        seq_tensor, size_interval = x
        hidden = self.s2h(size_interval)
        out, hidden = self.gru(seq_tensor, hidden)
        return out


class WeightNLLLoss(nn.Module):
    def __init__(self) -> None:
        """
        Initialize WeightNLLLoss module.
        """
        super(WeightNLLLoss, self).__init__()

    def forward(self, output: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        """
        Forward pass of the WeightNLLLoss module.

        Args:
            output (Tensor): The output tensor from the model, with shape (batch_size, seq_len, n_classes).
            target (Tensor): The ground truth tensor, with shape (batch_size, seq_len).
            weight (Tensor): A tensor of weights, with shape (batch_size, ), to apply to each batch element's loss.

        Returns:
            Tensor: The computed weighted negative log likelihood loss.
        """
        # Compute the NLL loss on the flattened output
        loss = F.nll_loss(output.contiguous().view(-1, output.size(-1)), 
                          target.contiguous().view(-1), 
                          reduction='none')
        # Apply weights to the loss
        weighted_loss = torch.mean(loss.view_as(target) * weight)
        return weighted_loss


class Encoder(nn.Module):
    def __init__(self, n_size, n_interval, n_condition, hidden_dims, n_latent):
        """
        Initializes the Encoder module of a Variational Autoencoder (VAE).

        Args:
            n_size (int): Dimensionality of the size input feature.
            n_interval (int): Dimensionality of the interval input feature.
            n_condition (int): Dimensionality of the conditional input feature.
            hidden_dims (List[int]): List containing the sizes of hidden layers.
            n_latent (int): Dimensionality of the latent representation.
        """
        super(Encoder, self).__init__()
        
        # Initial linear layers to transform the specific input features into a common feature space.
        self.dist_to_hidden = nn.Linear(n_size + n_interval, hidden_dims[0])
        self.condition_to_hidden = nn.Linear(n_condition, hidden_dims[0])

        # ModuleList to hold the sequence of hidden layers.
        self.encoder = nn.ModuleList()
        in_dim = hidden_dims[0]  # Input dimension for the first hidden layer
        for h_dim in hidden_dims:
            # Adding each hidden layer to the encoder, followed by a ReLU activation function.
            self.encoder.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU()))
            in_dim = h_dim  # Update input dimension for the next hidden layer

        # Linear layers to output the parameters of the variational distribution: mean (mu) and log variance (log_var).
        self.fc_mu = nn.Linear(hidden_dims[-1], n_latent)  # Outputs mean of the latent distribution
        self.fc_var = nn.Linear(hidden_dims[-1], n_latent)  # Outputs log variance of the latent distribution

    
    def forward(self, x: Tensor, c: Tensor) -> List[Tensor]:
        """
        Forward pass through the encoder.

        Args:
            x (Tensor): The input feature tensor (combining size and interval features).
            c (Tensor): The conditional input tensor.

        Returns:
            List[Tensor]: A list containing two tensors, [mu, log_var], representing
                          the mean and log variance of the latent distribution.
        """
        # Transform and combine the specific input features into a common feature space.
        x = self.dist_to_hidden(x) + self.condition_to_hidden(c)
        x = F.relu(x)  # Apply ReLU activation to the combined features
        
        # Pass the transformed input through the sequence of hidden layers.
        for module in self.encoder:
            x = module(x)
        
        # Compute the mean (mu) and log variance (log_var) of the latent distribution.
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        
        return [mu, log_var]  # Return the parameters of the latent distribution


class Decoder(nn.Module):
    def __init__(self, n_size, n_interval, n_condition, hidden_dims, n_latent):
        """
        Initializes the Decoder module of a Variational Autoencoder (VAE).

        Args:
            n_size (int): The output dimension for the size distribution.
            n_interval (int): The output dimension for the interval distribution.
            n_condition (int): Dimensionality of the conditional input feature.
            hidden_dims (List[int]): List containing the sizes of hidden layers.
            n_latent (int): Dimensionality of the latent representation.
        """
        super(Decoder, self).__init__()

        # Initial linear layers to transform the latent and conditional inputs.
        self.latent_to_hidden = nn.Linear(n_latent, hidden_dims[0])  # Transforms latent vector to initial hidden state
        self.condition_to_hidden = nn.Linear(n_condition, hidden_dims[0])  # Transforms condition to initial hidden state
        
        # ModuleList to hold the sequence of hidden layers.
        self.decoder = nn.ModuleList()
        in_dim = hidden_dims[0]  # Input dimension for the first hidden layer
        for h_dim in hidden_dims:
            # Adding each hidden layer to the decoder, followed by a ReLU activation function.
            self.decoder.append(nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU()))
            in_dim = h_dim  # Update input dimension for the next hidden layer
        
        # Linear layers to output the reconstructed size and interval distributions.
        self.fc_size = nn.Linear(hidden_dims[-1], n_size)  # Outputs reconstructed size distribution
        self.fc_interval = nn.Linear(hidden_dims[-1], n_interval)  # Outputs reconstructed interval distribution
    
    def forward(self, z: Tensor, c: Tensor) -> List[Tensor]:
        """
        Forward pass through the decoder.

        Args:
            z (Tensor): The latent vector.
            c (Tensor): The conditional input tensor.

        Returns:
            List[Tensor]: A list containing two tensors, [size, interval], representing
                          the reconstructed size and interval distributions.
        """
        # Transform and combine the latent and conditional inputs into initial hidden state.
        z = self.latent_to_hidden(z) + self.condition_to_hidden(c)
        z = F.relu(z)  # Apply ReLU activation to the combined features
        
        # Pass the transformed input through the sequence of hidden layers.
        for module in self.decoder:
            z = module(z)
        
        # Compute the reconstructed size and interval distributions, applying softmax to normalize the outputs.
        size = F.softmax(self.fc_size(z), dim=1)  # Normalize the reconstructed size distribution
        interval = F.softmax(self.fc_interval(z), dim=1)  # Normalize the reconstructed interval distribution
        
        return [size, interval]  # Return the reconstructed distributions


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Performs the reparameterization trick to sample from a Gaussian distribution.

    Args:
        mu (Tensor): Mean of the distribution.
        logvar (Tensor): Log variance of the distribution.

    Returns:
        Tensor: Sampled tensor using the reparameterization trick.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu