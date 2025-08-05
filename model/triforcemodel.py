
import random, json, os
from typing import Tuple, Sequence, Optional

import numpy as np
import h5py
import torch, torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from sklearn.model_selection import train_test_split


class MultiFileXelaDataset(Dataset):
    def __init__(
        self,
        h5_paths: Sequence[str],
        offsets: np.ndarray,
        indices: np.ndarray,
        labels: np.ndarray,
        *,
        train: bool = False,
        normalize_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        normal_only: bool = False,
    ):
        self.h5_paths, self.offsets, self.indices = h5_paths, offsets, indices
        self.labels, self.train, self.stats = labels, train, normalize_stats
        self.normal_only = normal_only
        self.hfs = [h5py.File(p, "r") for p in h5_paths]  # keep open!

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i):
        gidx = self.indices[i]
        fi   = np.searchsorted(self.offsets, gidx, side="right") - 1
        lidx = int(gidx - self.offsets[fi])
        hf   = self.hfs[fi]

        x = hf["X"][lidx].astype(np.float32)         # (3,4,4,T)
        if self.normal_only:
            x = x[2:3]                               # keep Z only
        x = x.mean(-1)                               # (C,4,4)

        if self.stats is not None:
            m, s = self.stats
            x = (x - m) / (s + 1e-8)

        if self.train:
            if random.random() < 0.5:                # random 90deg rotation
                k = random.randint(0, 3)
                x = np.rot90(x, k, axes=(1, 2)).copy()
            if random.random() < 0.4:                # gaussian noise
                x += np.random.normal(0, 0.01 * np.std(x), x.shape)

        y = int(self.labels[gidx])
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y)



def compute_normalization_stats(
    h5_paths, offsets, indices, normal_only=False, max_samples=1000
):
    hfs = [h5py.File(p, "r") for p in h5_paths]
    step = max(1, len(indices) // max_samples)
    sample_idx = indices[::step]

    samples = []
    for gidx in sample_idx:
        fi   = np.searchsorted(offsets, gidx, side="right") - 1
        lidx = int(gidx - offsets[fi])
        samples.append(hfs[fi]["X"][lidx].astype(np.float32))
    for hf in hfs:
        hf.close()

    samples = np.stack(samples)                      # (N,3,4,4,T)
    if normal_only:
        samples = samples[:, 2:3]                    # (N,1,4,4,T)
    samples = samples.mean(-1)                       # (N,C,4,4)

    mean = samples.mean(axis=(0, 2, 3)).reshape(-1, 1, 1)
    std  = samples.std (axis=(0, 2, 3)).reshape(-1, 1, 1)
    return mean, std

def create_balanced_sampler(labels: np.ndarray):
    uniq, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    return WeightedRandomSampler(weights[labels], len(labels))


def split_indices(
    y: np.ndarray, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices into train / val / test in a stratified way.

    • val_frac  – fraction of the whole set reserved for validation  
    • test_frac – fraction reserved for testing (can be 0)  

    When either val_frac or test_frac is 0, the corresponding split is returned
    as an empty array. The remaining samples are assigned to the existing split.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1.0:
        raise ValueError("val_frac and test_frac must be ≥0 and sum to < 1.")

    idx = np.arange(len(y))

    # No hold-out at all -> everything is training
    if val_frac == 0 and test_frac == 0:
        return idx, np.empty(0, dtype=int), np.empty(0, dtype=int)

    hold_frac = val_frac + test_frac
    tr_idx, hold_idx = train_test_split(
        idx, test_size=hold_frac, stratify=y, random_state=seed
    )

    if test_frac == 0:
        return tr_idx, hold_idx, np.empty(0, dtype=int)

    if val_frac == 0:
        return tr_idx, np.empty(0, dtype=int), hold_idx

    rel_val = val_frac / hold_frac  # fraction of holdout that will be validation
    val_idx, te_idx = train_test_split(
        hold_idx, test_size=1 - rel_val, stratify=y[hold_idx], random_state=seed
    )
    return tr_idx, val_idx, te_idx

def mixup_collate(batch, alpha: float = 0.4):
    xs, ys = zip(*batch)
    xs, ys = torch.stack(xs), torch.tensor(ys)
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(xs.size(0))
    return lam * xs + (1 - lam) * xs[idx], (ys, ys[idx], lam)

def run_epoch(model, loader, optim, device, loss_fn, mixup: bool = False):
    train = optim is not None
    model.train() if train else model.eval()

    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for data in loader:
            if mixup:
                xs, (ya, yb, lam) = data
                xs, ya, yb = xs.to(device), ya.to(device), yb.to(device)
                out = model(xs)
                loss = lam * loss_fn(out, ya) + (1 - lam) * loss_fn(out, yb)
                y = (
                    lam * nn.functional.one_hot(ya, out.size(1))
                    + (1 - lam) * nn.functional.one_hot(yb, out.size(1))
                ).argmax(1)
            else:
                xs, y = data
                xs, y = xs.to(device), y.to(device)
                out = model(xs)
                loss = loss_fn(out, y)

            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            total   += xs.size(0)
            correct += (out.argmax(1) == y).sum().item()
            loss_sum+= loss.item()
    return loss_sum / len(loader), correct / total

class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, r: int = 4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // r),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x).unsqueeze(-1).unsqueeze(-1)


class MixerBlock(nn.Module):
    def __init__(self, dim: int, tokens: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlp1  = nn.Sequential(nn.Linear(tokens, 64), nn.GELU(),
                                   nn.Linear(64, tokens))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2  = nn.Sequential(nn.Linear(dim, 64), nn.GELU(),
                                   nn.Linear(64, dim))

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        x = x + self.mlp1(y).transpose(1, 2)  # token mixing
        x = x + self.mlp2(self.norm2(x))       # channel mixing
        return x


class SpatialMixerXS(nn.Module):
    """A tiny spatial MLP-Mixer for 4×4 tactile maps."""
    def __init__(self, n_classes: int, in_ch: int = 3, dropout: float = 0.3):
        super().__init__()
        self.proj   = nn.Conv2d(in_ch, 32, 1)
        self.blocks = nn.ModuleList(MixerBlock(32, tokens=16) for _ in range(4))
        self.se     = SqueezeExcite(32)
        self.head   = nn.Linear(32, n_classes)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)                             # (B,32,4,4)
        b, c, h, w = x.shape
        tokens = x.reshape(b, c, h * w).permute(0, 2, 1)
        for blk in self.blocks:
            tokens = blk(tokens)
        x = tokens.permute(0, 2, 1).reshape(b, c, h, w)
        x = self.se(x).mean(dim=[2, 3])              # global pool
        return self.head(self.drop(x))
