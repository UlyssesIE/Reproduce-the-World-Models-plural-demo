"""Robust checkpoint loading.

The training scripts may save: the nn.Module itself, a dict holding it under
'model'/'net', or a dict with 'state_dict' (+ 'args'/'config') that needs the
class to be rebuilt.  All three are handled; anything else fails loudly.
"""
from __future__ import annotations

import inspect
import torch
import torch.nn as nn


def torch_load(path, dev="cpu"):
    try:
        return torch.load(path, map_location=dev, weights_only=False)
    except TypeError:                      # torch < 2.0
        return torch.load(path, map_location=dev)


def _as_kwargs(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def _ctor_kwargs(ctor, cfg):
    if ctor is None:
        return {}
    cfg = _as_kwargs(cfg)
    params = inspect.signature(ctor).parameters
    kw = {k: v for k, v in cfg.items() if k in params}
    missing = [k for k, p in params.items()
               if p.default is inspect.Parameter.empty
               and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
               and k not in kw]
    if missing:
        print(f"[load] WARNING: {ctor.__name__} needs {missing} but they are "
              f"not in the checkpoint; pass ctor_kwargs= explicitly")
    return kw


def load_module(path, dev="cpu", ctor=None, ctor_kwargs=None, name="model"):
    """Return the checkpoint's nn.Module in eval() mode."""
    ckpt = torch_load(path, dev)

    # case 1: the module itself
    if isinstance(ckpt, nn.Module):
        print(f"[load] {name}: nn.Module ({type(ckpt).__name__})")
        return ckpt.eval().to(dev)

    if not isinstance(ckpt, dict):
        raise TypeError(f"{name}: unsupported checkpoint type {type(ckpt)}")

    print(f"[load] {name}: dict keys = {list(ckpt.keys())}")

    # case 2: module nested in the dict
    for k in ("model", "net", "module", "network", "vae", "mdn", "rnn"):
        v = ckpt.get(k)
        if isinstance(v, nn.Module):
            print(f"[load] {name}: found nn.Module under {k!r}")
            return v.eval().to(dev)

    # case 3: raw state_dict, rebuild from the class
    sd = next((ckpt[k] for k in ("state_dict", "sd", "model_state_dict")
               if isinstance(ckpt.get(k), dict)), None)
    if sd is None:
        raise KeyError(f"{name}: no module and no state_dict in "
                       f"{list(ckpt.keys())}; paste this line back")
    if ctor is None:
        raise ValueError(f"{name}: checkpoint holds a state_dict; a class is "
                         f"required (ctor=...) -- see the file list below")
    cfg = ctor_kwargs if ctor_kwargs is not None else (
        ckpt.get("args") or ckpt.get("config") or ckpt.get("hparams"))
    obj = ctor(**_ctor_kwargs(ctor, cfg))
    obj.load_state_dict(sd, strict=True)
    print(f"[load] {name}: rebuilt {type(obj).__name__} from state_dict")
    return obj.eval().to(dev)


def build_meta(ckpt_or_path):
    """Small helper for reporting z_dim / hidden / etc. from a checkpoint."""
    d = ckpt_or_path
    if not isinstance(d, dict):
        d = torch_load(ckpt_or_path)
    cfg = _as_kwargs(d.get("args") or d.get("config") or d.get("hparams"))
    return {k: d[k] for k in ("z_dim", "hidden", "n_components", "epoch")
            if k in d} | cfg
