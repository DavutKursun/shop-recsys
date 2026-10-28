"""Two-tower retrieval model (the YouTube / Google "sampled softmax with in-batch negatives" recipe).

    user tower   u = MLP(mean-pool(E_hist[history] + position-from-end)) → L2-normalised
    item tower   v = MLP([E_item[item]; E_cat[category]; E_parent[parent]]) → L2-normalised
    logits_ij = u_i · v_j / temperature − log q(item_j)       every other positive in the batch is a negative
    loss = cross-entropy with the diagonal as the target

The logQ correction (Yi et al., RecSys 2019) removes the bias of in-batch sampling, which otherwise punishes
popular items for being sampled as negatives more often. Items share a tower that also reads the category, so
an item outside the id vocabulary can still be embedded from its category (the cold-start path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import Recommender
from .config import FIRST_ITEM, PAD, SEED, TWO_TOWER, UNK, TwoTowerConfig
from .data import Dataset
from .torch_utils import left_pad, pick_device, run_epochs, seed_everything


class ItemTower(nn.Module):
    def __init__(self, n_items: int, n_cats: int, n_parents: int, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(n_items, cfg.dim, padding_idx=PAD)
        self.cat_emb = nn.Embedding(n_cats, cfg.dim // 2)
        self.parent_emb = nn.Embedding(n_parents, cfg.dim // 4)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim + cfg.dim // 2 + cfg.dim // 4, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, cfg.dim))
        for e in (self.item_emb, self.cat_emb, self.parent_emb):
            nn.init.normal_(e.weight, std=0.05)

    def forward(self, item: torch.Tensor, cat: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.item_emb(item), self.cat_emb(cat), self.parent_emb(parent)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    """Mean-pooled history (with position-from-end embeddings) concatenated with the last item's embedding, so the
    tower keeps both the long-term taste and the immediate intent."""

    def __init__(self, n_items: int, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.hist_emb = nn.Embedding(n_items, cfg.dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.dim)  # position counted from the end: the last item is 0
        self.mlp = nn.Sequential(nn.Linear(2 * cfg.dim, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, cfg.dim))
        nn.init.normal_(self.hist_emb.weight, std=0.05)
        nn.init.normal_(self.pos_emb.weight, std=0.01)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        b, L = history.shape
        mask = (history != PAD).float()[..., None]
        pos = torch.arange(L - 1, -1, -1, device=history.device)[None, :].expand(b, L)
        emb = self.hist_emb(history)
        x = (emb + self.pos_emb(pos)) * mask
        pooled = x.sum(1) / mask.sum(1).clamp(min=1.0)
        return F.normalize(self.mlp(torch.cat([pooled, emb[:, -1]], dim=-1)), dim=-1)


class TwoTowerNet(nn.Module):
    def __init__(self, n_items: int, cat_of_item: np.ndarray, parent_of_item: np.ndarray, cfg: TwoTowerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("cat_of_item", torch.as_tensor(cat_of_item, dtype=torch.long))
        self.register_buffer("parent_of_item", torch.as_tensor(parent_of_item, dtype=torch.long))
        self.user = UserTower(n_items, cfg)
        self.item = ItemTower(n_items, int(cat_of_item.max()) + 1, int(parent_of_item.max()) + 1, cfg)

    def item_vectors(self, items: torch.Tensor) -> torch.Tensor:
        return self.item(items, self.cat_of_item[items], self.parent_of_item[items])


class TwoTower(Recommender):
    name = "two_tower"

    def __init__(self, cfg: TwoTowerConfig = TWO_TOWER, device: str | None = None, seed: int = SEED) -> None:
        super().__init__()
        self.cfg, self.device_name, self.seed = cfg, device, seed
        self.net: TwoTowerNet | None = None
        self.item_matrix: np.ndarray | None = None
        self.log_q: np.ndarray | None = None
        self.log = None

    # training -----------------------------------------------------------------------------------------------------
    def _fit(self, ds: Dataset, eval_cases=None, patience: int = 3, log_fn=print) -> None:
        seed_everything(self.seed)
        device = pick_device(self.device_name)
        cfg = self.cfg
        seqs = [s for s in ds.train_sequences() if len(s) >= 2]
        cat = np.zeros(ds.n_items, dtype=np.int64)
        parent = np.zeros(ds.n_items, dtype=np.int64)
        cat[ds.items["iidx"].to_numpy()] = ds.items["category"].to_numpy()
        parent[ds.items["iidx"].to_numpy()] = ds.items["parent"].to_numpy()
        self.net = TwoTowerNet(ds.n_items, cat, parent, cfg).to(device)
        # sampling probability of each item as a positive (for the logQ correction)
        counts = np.bincount(np.concatenate([s[1:] for s in seqs]), minlength=ds.n_items).astype(float)
        self.log_q = np.log((counts + 1.0) / (counts.sum() + ds.n_items))
        log_q = torch.as_tensor(self.log_q, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        rng = np.random.default_rng(self.seed)
        # every prediction point of every sequence (the last max_len of them), like SASRec's per-position targets
        hists, positives = [], []
        for s in seqs:
            for q in range(max(1, len(s) - cfg.max_len), len(s)):
                if s[q] != UNK:
                    hists.append(s[max(0, q - cfg.max_len) : q])
                    positives.append(s[q])
        hist_all, pos_all = left_pad(hists, cfg.max_len).astype(np.int32), np.array(positives, dtype=np.int64)
        del hists, positives
        if len(pos_all) > cfg.max_examples:  # bound memory and epoch time on large data
            keep = np.sort(rng.choice(len(pos_all), cfg.max_examples, replace=False))
            hist_all, pos_all = hist_all[keep], pos_all[keep]
        if log_fn:
            log_fn(f"two-tower: {len(pos_all):,} training examples from {len(seqs):,} sequences")

        def epoch_examples():
            return hist_all, pos_all

        def train_epoch() -> float:
            self.net.train()
            hist, pos = epoch_examples()
            order = rng.permutation(len(pos))
            total, count = 0.0, 0
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                h = torch.as_tensor(hist[idx], device=device).long()
                items = torch.as_tensor(pos[idx], device=device)
                u = self.net.user(h)
                v = self.net.item_vectors(items)
                logits = (u @ v.T) / cfg.temperature
                if cfg.logq_correction:
                    logits = logits - log_q[items][None, :]
                same = items[None, :] == items[:, None]
                same.fill_diagonal_(False)
                logits = logits.masked_fill(same, float("-inf"))  # the same item elsewhere in the batch is not a negative
                loss = F.cross_entropy(logits, torch.arange(len(idx), device=device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, count = total + float(loss.detach()), count + 1
            return total / max(count, 1)

        evaluate = None
        if eval_cases is not None:
            from .metrics import ranks_in_topk

            def evaluate() -> float:
                self._refresh_items()
                idx, _ = self.topk(eval_cases.history, 10)
                return float((ranks_in_topk(idx, eval_cases.target) > 0).mean())

        self.log = run_epochs(self.net, cfg.epochs, train_epoch, evaluate, patience, log_fn)
        self.net.to("cpu").eval()
        self._refresh_items()

    # inference ----------------------------------------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device

    @torch.no_grad()
    def _refresh_items(self) -> None:
        self.net.eval()
        all_items = torch.arange(self.n_items, device=self._device())
        self.item_matrix = self.net.item_vectors(all_items).cpu().numpy().astype(np.float32)

    def item_vectors(self) -> np.ndarray:
        return self.item_matrix

    @torch.no_grad()
    def embed_new_items(self, categories: np.ndarray, parents: np.ndarray) -> np.ndarray:
        """Vectors for items outside the id vocabulary: UNK id embedding plus their category — the cold-start path."""
        self.net.eval()
        dev = self._device()
        unk = torch.full((len(categories),), UNK, dtype=torch.long, device=dev)
        return self.net.item(unk, torch.as_tensor(categories, device=dev), torch.as_tensor(parents, device=dev)).cpu().numpy()

    def _fit_len(self, history: np.ndarray) -> np.ndarray:
        L = self.cfg.max_len
        return np.ascontiguousarray(history[:, -L:]) if history.shape[1] >= L else left_pad(list(history), L)

    @torch.no_grad()
    def user_vectors(self, history: np.ndarray) -> np.ndarray:
        self.net.eval()
        return self.net.user(torch.as_tensor(self._fit_len(history), device=self._device())).cpu().numpy()

    def score_batch(self, history: np.ndarray) -> np.ndarray:
        s = self.user_vectors(history) @ self.item_matrix.T
        s[:, :FIRST_ITEM] = -np.inf
        return s.astype(np.float32)

    # persistence --------------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg.__dict__, "n_items": self.n_items, "state": self.net.state_dict(), "cat": self.net.cat_of_item.cpu().numpy(),
                    "parent": self.net.parent_of_item.cpu().numpy(), "log_q": self.log_q, "log": self.log.as_dict() if self.log else None}, path)

    @classmethod
    def load(cls, path: Path) -> TwoTower:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(TwoTowerConfig(**blob["cfg"]))
        m.n_items = blob["n_items"]
        m.net = TwoTowerNet(m.n_items, blob["cat"], blob["parent"], m.cfg)
        m.net.load_state_dict(blob["state"])
        m.net.eval()
        m.log_q = blob["log_q"]
        m._refresh_items()
        m.fitted = True
        return m
