"""MDN-RNN (M model) for the World Models reproduction.

Paper: Ha & Schmidhuber (2018).
Specified by the paper:
  * a recurrent network models P(z_{t+1} | a_t, z_t, h_t)
  * the output is a mixture of Gaussians, not a point estimate
  * the paper's CarRacing variant uses LSTM with 256 hidden units
    ("In the Car Racing task, the LSTM used 256 hidden units")
  * a temperature tau controls the uncertainty of the sampled z during
    controller training; tau = 1.0 while training M itself
  * the CarRacing M does NOT predict `done` (only the VizDoom variant does)

NOT specified by the paper (our choices -- see the deviation log):
  * number of mixture components (we use 5)
  * sigma parameterisation and its numeric clamps
  * the exact way tau enters the distribution
  * sequence length and optimiser settings
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MDNRNNConfig:
    z_dim: int = 32
    action_dim: int = 3
    hidden: int = 256                 # paper: 256 for CarRacing
    n_components: int = 5             # paper does not state this
    num_layers: int = 1
    predict_done: bool = False        # True only for the VizDoom variant
    sigma_min: float = 1e-3
    sigma_max: float = 10.0
    # how temperature enters sampling:
    #   "sqrt"   : sigma <- sigma * sqrt(tau)   (default; matches the common
    #              reference implementation, which adds 0.5*log(tau) to logstd)
    #   "linear" : sigma <- sigma * tau
    #   "pi"     : sharpen the mixture weights, pi^(1/tau), renormalised
    #   "both"   : "sqrt" on sigma and "pi" on the weights
    tau_mode: str = "sqrt"


class MDNRNN(nn.Module):
    def __init__(self, cfg: MDNRNNConfig | None = None):
        super().__init__()
        self.cfg = cfg or MDNRNNConfig()
        c = self.cfg
        self.lstm = nn.LSTM(
            input_size=c.z_dim + c.action_dim,
            hidden_size=c.hidden,
            num_layers=c.num_layers,
            batch_first=True,
        )
        self.n_out = c.n_components * (1 + 2 * c.z_dim) + int(c.predict_done)
        self.head = nn.Linear(c.hidden, self.n_out)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(p)
                # forget-gate bias to 1 for stable long sequences
                n = p.numel() // 4
                p.data[n:2 * n].fill_(1.0)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)
        # start with small sigma: log(sigma) ~ 0 -> sigma = 1 before scaling
        c = self.cfg
        with torch.no_grad():
            self.head.bias.view(-1)[c.n_components + c.n_components * c.z_dim:
                                    c.n_components + 2 * c.n_components * c.z_dim] \
                .zero_()

    # ------------------------------------------------------------------ #
    def init_hidden(self, batch_size: int, device=None):
        c = self.cfg
        z = torch.zeros(c.num_layers, batch_size, c.hidden, device=device)
        return (z, torch.zeros_like(z))

    def split(self, y: torch.Tensor):
        """(B,T,K*(1+2D)[+1]) -> pi_logit (B,T,K), mu (B,T,K,D), log_sigma."""
        c = self.cfg
        K, D = c.n_components, c.z_dim
        pi_logit = y[..., :K]
        mu = y[..., K:K + K * D].view(*y.shape[:-1], K, D)
        log_sigma = y[..., K + K * D:K + 2 * K * D].view(*y.shape[:-1], K, D)
        log_sigma = log_sigma.clamp(math.log(c.sigma_min), math.log(c.sigma_max))
        done = y[..., -1] if c.predict_done else None
        return pi_logit, mu, log_sigma, done

    def forward(self, z: torch.Tensor, a: torch.Tensor, h=None):
        """z (B,T,D), a (B,T,A) -> pi_logit, mu, log_sigma, done, h."""
        if h is None:
            h = self.init_hidden(z.size(0), z.device)
        x = torch.cat([z, a], dim=-1)
        out, h = self.lstm(x, h)
        y = self.head(out)
        pi_logit, mu, log_sigma, done = self.split(y)
        return pi_logit, mu, log_sigma, done, h

    # ------------------------------------------------------------------ #
    def nll(self, pi_logit, mu, log_sigma, target):
        """Negative log-likelihood of a Gaussian mixture. target (B,T,D).

        -log sum_k pi_k N(target; mu_k, sigma_k), computed with logsumexp.
        """
        t = target.unsqueeze(-2)                              # (B,T,1,D)
        sigma = torch.exp(log_sigma)
        sq = ((t - mu) / sigma) ** 2
        log_comp = -0.5 * (sq + 2.0 * log_sigma
                           + math.log(2.0 * math.pi)).sum(-1)  # (B,T,K)
        log_pi = F.log_softmax(pi_logit, dim=-1)
        return -torch.logsumexp(log_pi + log_comp, dim=-1)     # (B,T)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, pi_logit, mu, log_sigma, temperature: float = 1.0):
        """Draw z from the mixture, optionally temperature-scaled.

        Shapes: pi_logit (...,K), mu/log_sigma (...,K,D) -> z (...,D).
        Note that tau changes only the *sampling* distribution; the mixture
        parameters produced during training always use tau = 1.
        """
        assert temperature > 0, temperature
        mode = self.cfg.tau_mode

        if mode in ("sqrt", "both"):
            log_sigma = log_sigma + 0.5 * math.log(temperature)
        elif mode == "linear":
            log_sigma = log_sigma + math.log(temperature)
        elif mode != "pi":
            raise ValueError(f"unknown tau_mode {mode!r}")

        log_pi = F.log_softmax(pi_logit, dim=-1)
        if mode in ("pi", "both"):
            log_pi = log_pi / temperature
            log_pi = log_pi - torch.logsumexp(log_pi, dim=-1, keepdim=True)

        idx = torch.distributions.Categorical(logits=log_pi).sample()   # (...)
        D = self.cfg.z_dim
        sel = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, D)
        mu_k = torch.gather(mu, -2, sel).squeeze(-2)
        ls_k = torch.gather(log_sigma, -2, sel).squeeze(-2)
        eps = torch.randn_like(mu_k)
        return mu_k + torch.exp(ls_k) * eps

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
