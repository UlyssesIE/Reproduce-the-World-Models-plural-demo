"""Linear controller C (Ha & Schmidhuber 2018, §2.3 and Appendix).

    y_t = W_c [z_t; h_t] + b_c        W_c: 3 x 288,  b_c: 3   -> 867 params
    a_t = low + (tanh(y_t) + 1) / 2 * (high - low)

The tanh + affine map is the paper's own description ("we applied tanh
nonlinearities to clip and bound the action space to the appropriate ranges"),
which for steer/low=-1, high=1 reduces to tanh, and for gas/brake (low=0,
high=1) reduces to (tanh+1)/2.

CarRacing uses the LSTM's *output* vector h_t only; c_t is used for VizDoom.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

PAPER_PARAMS = 867
# Initial bias: steer 0, gas ~0.5, brake ~0.12 (tanh(0)=0, tanh(-2)=-0.964).
# Weights start at exactly zero, so the initial population is a perturbation of
# one benign "roll forward slowly" policy instead of a saturated random one.
BIAS0 = (0.0, 0.0, -2.0)


class Controller(nn.Module):
    def __init__(self, z_dim: int = 32, hidden: int = 256, action_dim: int = 3,
                 use_hidden: bool = True, weight_init: float = 0.0,
                 bias0=BIAS0):
        super().__init__()
        self.z_dim, self.hidden, self.action_dim = z_dim, hidden, action_dim
        self.use_hidden = bool(use_hidden)
        self.in_dim = z_dim + (hidden if self.use_hidden else 0)

        self.fc = nn.Linear(self.in_dim, action_dim)
        if weight_init == 0.0:
            nn.init.zeros_(self.fc.weight)
        else:
            nn.init.normal_(self.fc.weight, std=float(weight_init))
        with torch.no_grad():
            b = torch.zeros(action_dim)
            for i, v in enumerate(tuple(bias0)[:action_dim]):
                b[i] = float(v)
            self.fc.bias.copy_(b)

        self.register_buffer("low", torch.tensor([-1.0, 0.0, 0.0]))
        self.register_buffer("high", torch.tensor([1.0, 1.0, 1.0]))

    # ---------------- spec ---------------- #
    @property
    def n_params(self) -> int:
        return self.in_dim * self.action_dim + self.action_dim

    def check_spec(self, expect_paper: bool = True) -> int:
        n = int(sum(p.numel() for p in self.parameters()))
        assert n == self.n_params, (n, self.n_params)
        if expect_paper:
            assert n == PAPER_PARAMS, (
                f"C has {n} params, paper reports {PAPER_PARAMS} "
                f"(= 288*3+3); use_hidden must be True")
        return n

    def set_bounds(self, low, high) -> None:
        with torch.no_grad():
            self.low.copy_(torch.as_tensor(low, dtype=self.low.dtype))
            self.high.copy_(torch.as_tensor(high, dtype=self.high.dtype))
        assert torch.all(self.low < self.high)

    # ---------------- forward / act ---------------- #
    def forward(self, z, h=None):
        x = z if not self.use_hidden else torch.cat([z, h], dim=-1)
        return torch.tanh(self.fc(x))

    def act(self, z, h=None):
        """z (D,) or (B,D); h (H,) or (B,H) -> bounded action(s), same batch."""
        y = self.forward(z, h)                       # in [-1, 1]
        return self.low + (y + 1.0) * 0.5 * (self.high - self.low)

    # ---------------- CMA-ES flat vector ---------------- #
    def get_flat(self) -> np.ndarray:
        return np.concatenate([self.fc.weight.detach().cpu().numpy().ravel(),
                               self.fc.bias.detach().cpu().numpy().ravel()]
                              ).astype(np.float64)

    def set_flat(self, theta) -> None:
        th = np.asarray(theta, dtype=np.float64).reshape(-1)
        assert th.size == self.n_params, (th.size, self.n_params)
        n = self.fc.weight.numel()
        with torch.no_grad():
            self.fc.weight.copy_(torch.as_tensor(th[:n]).view_as(self.fc.weight))
            self.fc.bias.copy_(torch.as_tensor(th[n:]).view_as(self.fc.bias))

    def save(self, path, **meta) -> None:
        np.savez(path, theta=self.get_flat(), z_dim=self.z_dim,
                 hidden=self.hidden, action_dim=self.action_dim,
                 use_hidden=self.use_hidden, meta=np.array([meta], dtype=object))

    @classmethod
    def from_file(cls, path):
        d = np.load(path, allow_pickle=True)
        c = cls(int(d["z_dim"]), int(d["hidden"]), int(d["action_dim"]),
                bool(d["use_hidden"]))
        c.set_flat(d["theta"])
        return c
