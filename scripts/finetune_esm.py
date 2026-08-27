#!/usr/bin/env python3
"""Fine-tune ESM-2 with a classification head for AMP detection.

Strategy 2 from the bio-LLM playbook: instead of frozen embeddings +
logistic regression, fine-tune the ESM-2 model itself with a
classification head. The 8M-parameter model fits on MPS and trains in
minutes. This is the "fine-tune an existing open-source model" path.

Usage:
    /opt/anaconda3/bin/python3 scripts/finetune_esm.py --data data --out models
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"


class PeptideDataset(Dataset):
    def __init__(self, seqs, labels, tokenizer, max_len=64):
        self.enc = tokenizer(seqs, padding="max_length", truncation=True,
                             max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}, self.labels[i]


class ESMClassifier(nn.Module):
    """ESM-2 + mean-pooled classification head (fine-tuned end to end)."""

    def __init__(self, esm, n_classes=2, hidden=64):
        super().__init__()
        self.esm = esm
        self.head = nn.Sequential(
            nn.Linear(esm.config.hidden_size, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes))

    def forward(self, input_ids, attention_mask):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return self.head(pooled)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--out", default="models", type=Path)
    ap.add_argument("--epochs", default=6, type=int)
    ap.add_argument("--batch-size", default=32, type=int)
    ap.add_argument("--lr", default=3e-5, type=float)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    # Data: AMPs (GRAMPA) + short non-AMPs, balanced.
    pos = []
    with open(args.data / "grampa.csv") as fh:
        for r in csv.DictReader(fh):
            pos.append(r["sequence"].upper())
    neg = []
    for line in (args.data / "non_amp_short.fasta").read_text().splitlines():
        if line.startswith(">"):
            continue
        neg.append(line.strip().upper())
    neg = neg[: len(pos)]
    seqs = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    print(f"data: {len(pos)} AMP + {len(neg)} non-AMP")

    # Split.
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(seqs))
    n_tr = int(0.8 * len(seqs))
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
    tr_seqs = [seqs[i] for i in tr_idx]
    tr_lab = [labels[i] for i in tr_idx]
    te_seqs = [seqs[i] for i in te_idx]
    te_lab = [labels[i] for i in te_idx]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    esm = AutoModel.from_pretrained(MODEL_NAME)
    model = ESMClassifier(esm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params (fine-tuned end to end)")

    train_ds = PeptideDataset(tr_seqs, tr_lab, tokenizer)
    test_ds = PeptideDataset(te_seqs, te_lab, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch, y in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = y.to(device)
            optimizer.zero_grad()
            out = model(**batch)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        print(f"epoch {epoch} | loss {total_loss / len(tr_seqs):.4f} | "
              f"{time.time() - t0:.0f}s")

    # Evaluate.
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch, y in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
            ys.extend(y.numpy())
    auc_ = roc_auc_score(ys, probs)
    acc = np.mean((np.array(probs) >= 0.5) == np.array(ys))
    print(f"fine-tuned ESM-2: acc {acc:.3f} | AUC {auc_:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_name": MODEL_NAME},
               args.out / "esm_finetuned.pt")
    print(f"saved -> {args.out / 'esm_finetuned.pt'}")


if __name__ == "__main__":
    main()
