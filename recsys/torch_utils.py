"""Shared PyTorch plumbing: device choice, left-padded batches, uniform negatives, an early-stopping loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import FIRST_ITEM, PAD, SEED, UNK


def pick_device(name: str | None = None) -> torch.device:
    """cuda (Kaggle) > mps (Apple Silicon) > cpu, unless a name is forced."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def left_pad(seqs: list[np.ndarray], max_len: int) -> np.ndarray:
    out = np.full((len(seqs), max_len), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        k = min(len(s), max_len)
        if k:
            out[i, max_len - k :] = s[-k:]
    return out


def sample_negatives(n: int, n_items: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    """Uniform item indices in [FIRST_ITEM, n_items)."""
    return torch.randint(FIRST_ITEM, n_items, (n,), generator=generator, device="cpu").to(device)


def valid_targets(target: torch.Tensor) -> torch.Tensor:
    """Positions whose target can be predicted: not padding, not an out-of-vocabulary item."""
    return (target != PAD) & (target != UNK)


@dataclass
class TrainLog:
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_metric: float = -1.0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"epochs": self.epochs, "best_epoch": self.best_epoch, "best_metric": self.best_metric, "seconds": round(self.seconds, 1)}


def run_epochs(net: torch.nn.Module, epochs: int, train_epoch, evaluate=None, patience: int = 3, log_fn=print) -> TrainLog:
    """Generic loop: train_epoch() → mean loss; evaluate() → a metric to maximise (dev recall@10). Keeps the best
    weights; stops after `patience` epochs without improvement. Without `evaluate`, runs all epochs."""
    log, best_state, t0 = TrainLog(), None, time.time()
    bad = 0
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        loss = train_epoch()
        metric = evaluate() if evaluate is not None else None
        row = {"epoch": ep, "loss": round(float(loss), 5), "metric": None if metric is None else round(float(metric), 5), "seconds": round(time.time() - t_ep, 1)}
        log.epochs.append(row)
        if log_fn:
            log_fn(f"epoch {ep:3d}  loss {loss:.4f}" + (f"  dev {metric:.4f}" if metric is not None else "") + f"  {row['seconds']}s")
        if metric is None or metric > log.best_metric:
            log.best_metric, log.best_epoch, bad = (metric if metric is not None else -1.0), ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    log.seconds = time.time() - t0
    return log
