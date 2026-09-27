"""Run train_mdnrnn with the a-slice of lstm.weight_ih_l0 projected to a fixed norm.

No edit to train_mdnrnn.py: the projection is installed as an LSTM forward
pre-hook, so the weight actually used by every forward pass has the target norm.

    python scripts/mdn/train_mdnrnn_norm.py --norm-target 5.242 \
        --latents data/pilot_latents_shuf --out runs/mdn_pilot_shufup \
        --epochs 30 --seed 0

NOTE: everything runs under `if __name__ == "__main__":` because DataLoader
workers are started with the Windows "spawn" method and re-import this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from src.models.mdn_rnn import MDNRNN

ACTION_DIM = 3


def _parse_target(argv):
    """Return (argv without --norm-target, value)."""
    keep, target, i = [], 0.0, 0
    while i < len(argv):
        if argv[i] == "--norm-target":
            target = float(argv[i + 1]); i += 2
        elif argv[i].startswith("--norm-target="):
            target = float(argv[i].split("=", 1)[1]); i += 1
        else:
            keep.append(argv[i]); i += 1
    return keep, target


def _install_hook(target: float) -> None:
    orig_init = MDNRNN.__init__

    def _init(self, cfg, *a, **kw):
        orig_init(self, cfg, *a, **kw)
        assert int(cfg.action_dim) == ACTION_DIM, cfg.action_dim

        def _pre(mod, inputs):
            with torch.no_grad():
                W = mod.weight_ih_l0
                sl = W[:, mod.input_size - ACTION_DIM:]
                sl.mul_(target / (float(sl.norm()) + 1e-8))
            return None

        self.lstm.register_forward_pre_hook(_pre)

    MDNRNN.__init__ = _init


if __name__ == "__main__":
    argv, target = _parse_target(sys.argv[1:])
    sys.argv = ["train_mdnrnn", *argv]
    if target > 0.0:
        _install_hook(target)
        print(f"[wrap] a-slice projected to ||Wa||={target:.4f} before every forward",
              flush=True)
    import train_mdnrnn
    raise SystemExit(train_mdnrnn.main())
