#!/usr/bin/env python3
"""Train an ESM-2 (protein language model) classifier for AMPs.

Replaces the hand-crafted physicochemical features with learned
representations from ESM-2 (facebook/esm2_t6_8M_UR50D) — a transformer
trained on 220M protein sequences. This is the same model family used
by the AllTheBacteria/APEX AMP-discovery pipeline (Hunt et al. 2026).

Pipeline:
  1. Embed each peptide with ESM-2 (mean-pooled, 320-dim)
  2. Train a logistic regression on the embeddings
  3. Report AUC / PR-AUC, compare to the physicochemical RF

Usage:
    python3 scripts/train_esm.py --data data --out models
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"


def embed_sequences(seqs: list[str], tokenizer, model, batch_size: int = 32) -> np.ndarray:
    """Mean-pooled ESM-2 embeddings for a list of sequences."""
    model.eval()
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True)
            out = model(**enc)
            # mask out padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_emb.append(emb.numpy())
    return np.vstack(all_emb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--out", default="models", type=Path)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load data.
    pos_seqs, pos_mic = [], []
    with open(args.data / "grampa.csv") as fh:
        for r in csv.DictReader(fh):
            pos_seqs.append(r["sequence"])
            pos_mic.append(float(r["mic"]))
    neg_seqs = []
    for line in (args.data / "non_amp.fasta").read_text().splitlines():
        if line.startswith(">"):
            continue
        neg_seqs.append(line.strip().upper())
    print(f"data: {len(pos_seqs)} AMP + {len(neg_seqs)} non-AMP")

    # Embed.
    print(f"loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    print("embedding positives...")
    X_pos = embed_sequences(pos_seqs, tokenizer, model)
    print("embedding negatives...")
    X_neg = embed_sequences(neg_seqs, tokenizer, model)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))
    print(f"embeddings: {X.shape}")

    # Train classifier on embeddings.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)
    clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, (prob >= 0.5).astype(int))
    auc_ = roc_auc_score(y_te, prob)
    prec, rec, _ = precision_recall_curve(y_te, prob)
    print(f"ESM-2 classifier: acc {acc:.3f} | AUC {auc_:.3f} | "
          f"PR-AUC {auc(rec, prec):.3f}")

    # Save.
    import joblib
    joblib.dump({"clf": clf, "model_name": MODEL_NAME}, args.out / "esm_amp.joblib")
    print(f"saved -> {args.out / 'esm_amp.joblib'}")


if __name__ == "__main__":
    main()
