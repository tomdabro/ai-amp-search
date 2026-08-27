#!/usr/bin/env python3
"""Train a hemolysis (toxicity) predictor for AMPs.

A drug candidate needs potency AND low toxicity. This trains a
classifier on the ConsAMPHemo S1 dataset (884 peptides, hemolytic vs
non-hemolytic) using ESM-2 embeddings — the same approach as the AMP
classifier. The result: score a peptide for both activity and safety.

Data: ConsAMPHemo (Xie et al. 2025, Protein Science) — public at
https://github.com/Cpillar/ConsAMPHemo (Dataset/S1/train.csv).

Usage:
    /opt/anaconda3/bin/python3 scripts/train_hemolysis.py \
        --data data/hemolysis_train.csv --out models
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from train_esm import embed_sequences


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path,
                    help="hemolysis CSV with 'text' and 'labels' columns")
    ap.add_argument("--out", default="models", type=Path)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    seqs, labels = [], []
    with open(args.data) as fh:
        for r in csv.DictReader(fh):
            seqs.append("".join(c for c in r["text"] if c.isalpha()).upper())
            labels.append(int(r["labels"]))
    labels = np.array(labels)
    print(f"data: {len(seqs)} peptides, {labels.sum()} hemolytic / "
          f"{len(labels) - labels.sum()} non-hemolytic")

    print("loading ESM-2...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    X = embed_sequences(seqs, tokenizer, model)
    print(f"embeddings: {X.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.2, stratify=labels, random_state=args.seed)
    clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, (prob >= 0.5).astype(int))
    auc_ = roc_auc_score(y_te, prob)
    prec, rec, _ = precision_recall_curve(y_te, prob)
    print(f"hemolysis classifier: acc {acc:.3f} | AUC {auc_:.3f} | "
          f"PR-AUC {auc(rec, prec):.3f}")

    import joblib
    joblib.dump({"clf": clf, "model_name": "facebook/esm2_t6_8M_UR50D"},
                args.out / "hemolysis.joblib")
    print(f"saved -> {args.out / 'hemolysis.joblib'}")


if __name__ == "__main__":
    main()
