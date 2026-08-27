#!/usr/bin/env python3
"""Train a character-level GPT on antimicrobial peptide sequences.

The generative direction: a small GPT (nanoGPT-style) learns the
distribution of known AMP sequences, then samples novel candidates.
This is the same family as the generative AMP models in the literature
(Park et al. 2025 RL+LoRA, MOFormer 2025) — a real training run on
MPS, not frozen embeddings.

Model: character-level transformer (block size 64, 4 heads, 4 layers,
embed 128) — ~1.5M params, trains in minutes on an Apple MPS GPU.

Usage:
    /opt/anaconda3/bin/python3 scripts/train_gpt.py --data data --out models
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- tiny GPT (nanoGPT-style) ---
class GPTConfig:
    block_size = 64
    n_layer = 6
    n_head = 6
    n_embd = 192
    dropout = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--out", default="models", type=Path)
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--batch-size", default=64, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--warmup", default=200, type=int,
                    help="warmup steps (default 200)")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # Load AMP sequences, build vocab.
    seqs = []
    with open(args.data / "grampa.csv") as fh:
        for r in csv.DictReader(fh):
            s = r["sequence"].upper()
            if 5 <= len(s) <= 64:
                seqs.append(s)
    print(f"training sequences: {len(seqs)}")

    chars = sorted(set("".join(seqs)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    print(f"vocab: {vocab_size} amino acids: {''.join(chars)}")

    config = GPTConfig()
    config.block_size = min(config.block_size, max(len(s) for s in seqs))
    model = GPT(config, vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params, block size {config.block_size}")

    # Tokenize: 90/10 train/val split by sequence.
    rng = torch.Generator().manual_seed(args.seed)
    n_val = max(1, int(0.1 * len(seqs)))
    val_seqs = seqs[:n_val]
    train_seqs = seqs[n_val:]
    data = torch.tensor([stoi[c] for s in train_seqs for c in s], dtype=torch.long)
    val_data = torch.tensor([stoi[c] for s in val_seqs for c in s], dtype=torch.long)
    print(f"train tokens: {data.numel()}, val tokens: {val_data.numel()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_batches = max(1, (data.numel() - 1) // (args.batch_size * config.block_size))
    total_steps = n_batches * args.epochs
    print(f"batches per epoch: {n_batches}, total steps: {total_steps}")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        # cosine decay to 10% of peak
        frac = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

    @torch.no_grad()
    def val_loss():
        model.eval()
        total, n = 0.0, 0
        for _ in range(20):
            ix = torch.randint(0, val_data.numel() - config.block_size - 1, (args.batch_size,))
            x = torch.stack([val_data[i:i + config.block_size] for i in ix]).to(device)
            y = torch.stack([val_data[i + 1:i + 1 + config.block_size] for i in ix]).to(device)
            _, loss = model(x, y)
            total += loss.item() * len(x)
            n += len(x)
        model.train()
        return total / n

    t0 = time.time()
    step = 0
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for _ in range(n_batches):
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            ix = torch.randint(0, data.numel() - config.block_size - 1, (args.batch_size,))
            x = torch.stack([data[i:i + config.block_size] for i in ix]).to(device)
            y = torch.stack([data[i + 1:i + 1 + config.block_size] for i in ix]).to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            step += 1
        avg = total_loss / n_batches
        vl = val_loss()
        if vl < best_val:
            best_val = vl
            torch.save({"model": model.state_dict(), "config": config,
                        "stoi": stoi, "itos": itos},
                       args.out / "amp_gpt.pt")
        print(f"epoch {epoch:3d} | train {avg:.4f} | val {vl:.4f} | "
              f"ppl {math.exp(vl):.2f} | best {best_val:.4f} | "
              f"{time.time() - t0:.0f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"saved best (val {best_val:.4f}) -> {args.out / 'amp_gpt.pt'}")


if __name__ == "__main__":
    main()
