# Reproducing World Models (Ha & Schmidhuber, 2018)

From-scratch reimplementation of the V–M–C architecture, without referencing
the official or any third-party implementation. Code rewritten from the paper.

## Result
- V (VAE): R² = 0.916 on held-out frames
- M (MDN-RNN): ...
- C (controller): ...

## Pipeline
data collection → frame cache → VAE → latent encoding → MDN-RNN → controller

## Deviation log
See `notes/` — Note1 (initial data observations), Note2 (source-level
corrections), Note3 (VAE fixes and acceptance).

## Setup
python 3.11; pip install -r requirements.txt; python check_env.py

## Reproducing from scratch

Data and checkpoints are not tracked (see .gitignore). To rebuild:

```bash
python scripts/data/collect_data.py --out data/raw --num-rollouts 2000 --compress
python scripts/data/build_frame_cache.py --data data/raw --out data/raw_frames.npy
python scripts/vae/train_vae.py --data data/raw --frame-cache data/raw_frames.npy --out runs/vae
python scripts/vae/encode_latent.py --vae runs/vae/best.pt --data data/raw \
    --cache data/raw_frames.npy --out data/raw_latents
python scripts/mdn/train_mdnrnn.py --latents data/raw_latents --out runs/mdn
python scripts/mdn/dream_rollout.py --vae runs/vae/best.pt --mdn runs/mdn/best.pt \
    --latents data/raw_latents --steps 64 --tau 1.15
