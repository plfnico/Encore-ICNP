import numpy as np
from typing import List, Tuple 
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class EncoreVAE(nn.Module):
    def __init__(self, n_size, n_interval, hidden_dims, n_latent, device):
        super(EncoreVAE, self).__init__()
        # initialize the parameters
        self.n_size = n_size
        self.n_interval = n_interval
        self.n_latent = n_latent
        self.encorer_hidden_dims = hidden_dims
        self.decoder_hidden_dims = hidden_dims
        self.decoder_hidden_dims.reverse()
        self.fc_mu = nn.Linear(self.encorer_hidden_dims[-1], n_latent)
        self.fc_var = nn.Linear(self.encorer_hidden_dims[-1], n_latent)
        self.fc_size = nn.Linear(self.decoder_hidden_dims[-1], n_size)
        self.fc_interval = nn.Linear(self.decoder_hidden_dims[-1], n_interval)
        self.device = device

        # construct encoder
        encoder_layers = [nn.Linear(n_size + n_interval, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            encoder_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # construct decoder
        decoder_layers = [nn.Linear(n_latent, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            decoder_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x:Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu:Tensor, log_var:Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z:Tensor) -> Tensor:
        h = self.decoder(z)
        return self.fc_size(h), self.fc_interval(h)

    def forward(self, x:Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        size_logits, interval_logits = self.decode(z)
        size_recon = F.softmax(size_logits, dim=-1)
        interval_recon = F.softmax(interval_logits, dim=-1)
        return size_recon, interval_recon, mu, log_var
    
    def generate(self, n:int) -> Tuple[Tensor, Tensor]:
        z = torch.randn(n, self.n_latent).to(self.device)
        size_logits, interval_logits = self.decode(z)
        size_recon = F.softmax(size_logits, dim=-1)
        interval_recon = F.softmax(interval_logits, dim=-1)
        return size_recon, interval_recon
    

class CustomVAELoss(nn.Module):
    def __init__(self, kld_weight:float=1.0):
        super(CustomVAELoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, size_recon:Tensor, size_target:Tensor, interval_recon:Tensor, interval_target:Tensor, mu:Tensor, log_var:Tensor) -> Tensor:
        recon_loss = F.l1_loss(size_recon, size_target) + F.l1_loss(interval_recon, interval_target)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)
        weighted_loss = recon_loss + self.kld_weight * kld_loss
        return weighted_loss, recon_loss, kld_loss



class ProbToHidden(nn.Module):
    def __init__(self, hidden_size, input_size, hidden_dims):
        super(ProbToHidden, self).__init__()
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
        for lin in self.lins:
            x = lin(x)
        x = self.output(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  
        return x


class SequenceDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_seq_len=16):
        super(SequenceDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        self.transfomer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=8
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, output_dim)
        )
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

    def forward(self, x, memory, tgt_mask=None):
        x = self.positional_encoding(x)
        output = self.transfomer(x, memory, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask
    

class EncoreSequential(nn.Module):
    def __init__(self, n_size, n_interval, hidden_dim, seq_len, prob_hidden_dims, device):
        super(EncoreSequential, self).__init__()
        self.prob_to_hidden = ProbToHidden(hidden_dim, n_size + n_interval, prob_hidden_dims)
        self.embedding = nn.Embedding(n_size * n_interval + 1, hidden_dim)
        self.sequence_decoder = SequenceDecoder(hidden_dim, hidden_dim, seq_len)
        self.fc = nn.Linear(hidden_dim, n_size * n_interval)
        self.device = device

    def forward(self, x, prob):
        x, prob = x.long().to(self.device), prob.float().to(self.device)
        x = self.embedding(x)
        memory = self.prob_to_hidden(prob).unsqueeze(1).repeat(1, x.size(1), 1)
        tgt_mask = self.sequence_decoder.generate_square_subsequent_mask(x.size(1)).to(self.device)
        output = self.sequence_decoder(x, memory, tgt_mask)
        output = self.fc(output)
        output = F.log_softmax(output, dim=-1)
        return output 
    
    # def generate(self, prob, start_token, max_len):
    #     self.eval()
    #     seq = torch.tensor([start_token]).unsqueeze(0).to(self.device)
    #     x = self.embedding(seq)
    #     prob_tensor = torch.from_numpy(prob).unsqueeze(0).to(self.device)
    #     for _ in range(max_len):
    #         memory = self.prob_to_hidden(prob_tensor).unsqueeze(1).repeat(1, x.size(1), 1)
    #         tgt_mask = self.sequence_decoder.generate_square_subsequent_mask(x.size(1)).to(self.device)
    #         output = self.sequence_decoder(x, memory, tgt_mask)
    #         output = self.fc(output).squeeze(0)
    #         logits = F.softmax(output[-1], dim=-1).cpu().detach().numpy()
    #         next_token = np.random.choice(np.arange(len(logits)), p=logits)
    #         seq = torch.cat([seq, torch.tensor([next_token]).unsqueeze(0).to(self.device)], dim=1)
    #         x = self.embedding(seq)
    #     return seq[:, 1:].squeeze(0).detach().cpu().numpy()

    def generate(self, probs, start_tokens, max_len):
        batch_size = len(probs)
        assert len(start_tokens) == batch_size
        self.eval()
        seq = torch.tensor(start_tokens).unsqueeze(1).to(self.device)
        x = self.embedding(seq)
        prob_tensor = torch.from_numpy(probs).to(self.device)
        for _ in range(max_len):
            memory = self.prob_to_hidden(prob_tensor).unsqueeze(1).repeat(1, x.size(1), 1)
            tgt_mask = self.sequence_decoder.generate_square_subsequent_mask(x.size(1)).to(self.device)
            output = self.sequence_decoder(x, memory, tgt_mask)
            output = self.fc(output)[:,-1,:]
            logits = F.softmax(output, dim=-1).cpu().detach().numpy()
            next_tokens = [np.random.choice(np.arange(len(logit)), p=logit) for logit in logits]
            seq = torch.cat([seq, torch.tensor(next_tokens).unsqueeze(1).to(self.device)], dim=1)
            x = self.embedding(seq)
        return seq[:, 1:].squeeze(0).detach().cpu().numpy()


class WeightNLLLoss(nn.Module):
    def __init__(self) -> None:
        super(WeightNLLLoss, self).__init__()

    def forward(self, output: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        loss = F.nll_loss(output.contiguous().view(-1, output.size(-1)), target.contiguous().view(-1), reduction='none')
        weighted_loss = torch.mean(loss.view_as(target) * weight)
        return weighted_loss