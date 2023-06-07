import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as distrib


def L1_loss(p, z, average=True):
    loss = torch.abs(p-z)
    loss = loss.sum(dim=1)
    if average:
        loss = loss.mean()
    return loss


def L2_loss(p, z, average=True):
    loss = (p-z)**2
    loss = loss.sum(dim=1)
    if average:
        loss = loss.mean()
    return loss


def cosine_similarity(p, z, average=True):
    p = F.normalize(p, p=2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    loss = -(p * z).sum(dim=1)
    if average:
        loss = loss.mean()
    return loss


class AutoEncoder(torch.nn.Module):
    def __init__(self, emb_dim, loss, detach_target, beta=1):
        super(AutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.loss = loss
        self.detach_target = detach_target
        self.beta = beta

        self.criterion = None
        if loss == 'l1':
            self.criterion = nn.L1Loss()
            # self.criterion = L1_loss
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
            # self.criterion = L2_loss
        elif loss == 'cosine':
            self.criterion = cosine_similarity

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        return

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()

        z = x
        y_hat = self.decoder(z)

        reconstruction_loss = self.criterion(y_hat, y)

        loss = reconstruction_loss
        return loss


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, emb_dim, loss, detach_target, beta=1):
        super(VariationalAutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.loss = loss
        self.detach_target = detach_target
        self.beta = beta

        self.criterion = None
        if loss == 'l1':
            self.criterion = nn.L1Loss()
            # self.criterion = L1_loss
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
            # self.criterion = L2_loss
        elif loss == 'cosine':
            self.criterion = cosine_similarity

        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        return

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decoder(z)

        reconstruction_loss = self.criterion(y_hat, y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # negative reconstruction + KL
        loss = reconstruction_loss + self.beta * kl_loss

        return loss
