"""SASRec (Kang & McAuley, ICDM 2018) written from scratch: item + position embeddings, causal self-attention
blocks, next-item prediction at every position, sampled-softmax loss with uniform negatives.

    h = TransformerBlocks(E[item_1..item_L] + P)          causal: position i sees items ≤ i
    score(i, j) = h_i · E[j]                              tied output embeddings
    loss_i = −log softmax([h_i·E[target_i], h_i·E[neg_1], …, h_i·E[neg_N]])[0]

Uniform negatives need no correction; a full softmax (negatives = 0) is exact but needs batch × L × n_items
memory. Cross-entropy with many negatives beats the paper's one-negative BCE (Klenitskiy & Vasilev, RecSys 2023).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import Recommender
from .config import FIRST_ITEM, PAD, SASREC, SEED, SASRecConfig
from .data import Dataset
from .torch_utils import left_pad, pick_device, run_epochs, sample_negatives, seed_everything, valid_targets


class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """allowed: (b, L, L) bool — query i may attend to key j."""
        b, L, d = x.shape
        q, k, v = self.qkv(x).view(b, L, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)  # each (b, heads, L, dh)
        att = (q @ k.transpose(-1, -2)) / self.dh**0.5  # (b, heads, L, L)
        att = att.masked_fill(~allowed[:, None], float("-inf"))
        att = self.drop(torch.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(b, L, d)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.att = CausalAttention(dim, heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.att(self.ln1(x), allowed))
        return x + self.drop(self.ffn(self.ln2(x)))


class SASRecNet(nn.Module):
    def __init__(self, n_items: int, cfg: SASRecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(n_items, cfg.dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg.dim, cfg.heads, cfg.dropout) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """(b, L) left-padded → (b, L, dim) hidden states; position i predicts the item after item i."""
        b, L = seq.shape
        pad = seq == PAD
        causal = torch.ones(L, L, dtype=torch.bool, device=seq.device).tril()
        allowed = causal[None] & ~pad[:, None, :]  # (b, L, L): causal, and never a padding key
        allowed = allowed | torch.eye(L, dtype=torch.bool, device=seq.device)[None]  # a position may always see itself: no empty rows
        pos = torch.arange(L, device=seq.device)[None, :].expand(b, L)
        x = self.drop(self.item_emb(seq) * self.cfg.dim**0.5 + self.pos_emb(pos))
        x = x.masked_fill(pad[..., None], 0.0)
        for blk in self.blocks:
            x = blk(x, allowed).masked_fill(pad[..., None], 0.0)
        return self.ln(x)

    def user_vectors(self, seq: torch.Tensor) -> torch.Tensor:
        return self.forward(seq)[:, -1]

    def all_scores(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.item_emb.weight.T


class SASRec(Recommender):
    name = "sasrec"

    def __init__(self, cfg: SASRecConfig = SASREC, device: str | None = None, seed: int = SEED) -> None:
        super().__init__()
        self.cfg, self.device_name, self.seed = cfg, device, seed
        self.net: SASRecNet | None = None
        self.log = None

    # training -----------------------------------------------------------------------------------------------------
    def _fit(self, ds: Dataset, eval_cases=None, patience: int = 3, log_fn=print) -> None:
        seed_everything(self.seed)
        device = pick_device(self.device_name)
        cfg = self.cfg
        seqs = [s for s in ds.train_sequences() if len(s) >= 2]
        inputs = left_pad([s[:-1] for s in seqs], cfg.max_len)
        targets = left_pad([s[1:] for s in seqs], cfg.max_len)
        self.net = SASRecNet(ds.n_items, cfg).to(device)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)
        order_rng = np.random.default_rng(self.seed)

        def train_epoch() -> float:
            self.net.train()
            order = order_rng.permutation(len(inputs))
            total, count = 0.0, 0
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                seq = torch.as_tensor(inputs[idx], device=device)
                tgt = torch.as_tensor(targets[idx], device=device)
                loss = self._loss(seq, tgt, gen, device)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, count = total + float(loss.detach()), count + 1
            return total / max(count, 1)

        evaluate = None
        if eval_cases is not None:
            from .metrics import ranks_in_topk

            def evaluate() -> float:
                idx, _ = self.topk(eval_cases.history, 10)
                return float((ranks_in_topk(idx, eval_cases.target) > 0).mean())

        self.log = run_epochs(self.net, cfg.epochs, train_epoch, evaluate, patience, log_fn)
        self.net.to("cpu").eval()

    def _loss(self, seq: torch.Tensor, tgt: torch.Tensor, gen: torch.Generator, device: torch.device) -> torch.Tensor:
        h = self.net(seq)  # (b, L, d)
        valid = valid_targets(tgt)
        h, tgt = h[valid], tgt[valid]  # (m, d), (m,)
        E = self.net.item_emb.weight
        pos = (h * E[tgt]).sum(-1, keepdim=True)  # (m, 1)
        if self.cfg.negatives > 0:
            neg = sample_negatives(self.cfg.negatives, self.n_items, gen, device)  # shared by the batch
            neg_logits = (h @ E[neg].T).masked_fill(neg[None, :] == tgt[:, None], float("-inf"))  # drop accidental hits
        else:
            neg_logits = (h @ E.T).scatter(1, tgt[:, None], float("-inf"))
            neg_logits[:, :FIRST_ITEM] = float("-inf")
        logits = torch.cat([pos, neg_logits], dim=1)
        return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=device))

    # inference ----------------------------------------------------------------------------------------------------
    def _device(self) -> torch.device:
        return next(self.net.parameters()).device

    def _fit_len(self, history: np.ndarray) -> np.ndarray:
        L = self.cfg.max_len
        return np.ascontiguousarray(history[:, -L:]) if history.shape[1] >= L else left_pad(list(history), L)

    @torch.no_grad()
    def user_vectors(self, history: np.ndarray) -> np.ndarray:
        self.net.eval()
        return self.net.user_vectors(torch.as_tensor(self._fit_len(history), device=self._device())).cpu().numpy()

    def item_vectors(self) -> np.ndarray:
        return self.net.item_emb.weight.detach().cpu().numpy()

    @torch.no_grad()
    def _scores_tensor(self, history: np.ndarray) -> torch.Tensor:
        self.net.eval()
        u = self.net.user_vectors(torch.as_tensor(self._fit_len(history), device=self._device()))
        s = self.net.all_scores(u)
        s[:, :FIRST_ITEM] = float("-inf")
        return s

    def score_batch(self, history: np.ndarray) -> np.ndarray:
        return self._scores_tensor(history).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def topk(self, history: np.ndarray, k: int, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        idx_out, score_out = [], []
        for start in range(0, len(history), batch_size):
            s = self._scores_tensor(history[start : start + batch_size])
            v, i = torch.topk(s, min(k, s.shape[1] - FIRST_ITEM), dim=1)
            idx_out.append(i.cpu().numpy())
            score_out.append(v.cpu().numpy())
        return np.vstack(idx_out), np.vstack(score_out)

    # persistence --------------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg.__dict__, "n_items": self.n_items, "state": self.net.state_dict(), "log": self.log.as_dict() if self.log else None}, path)

    @classmethod
    def load(cls, path: Path) -> SASRec:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(SASRecConfig(**blob["cfg"]))
        m.n_items = blob["n_items"]
        m.net = SASRecNet(m.n_items, m.cfg)
        m.net.load_state_dict(blob["state"])
        m.net.eval()
        m.fitted = True
        return m
