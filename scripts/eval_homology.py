#!/usr/bin/env python3
"""Honest evaluation of the AMP classifier: homology-controlled split.

The naive random 80/20 split leaks homology: GRAMPA contains many
near-identical variants of the same peptide family, so the model can
memorize families instead of learning antimicrobiality (the problem
named by GenPept-Curated-2025). This script:

  1. Clusters AMP sequences by k-mer Jaccard similarity (>= 0.8)
  2. Splits by CLUSTER, not by sequence — no family in both train/test
  3. Trains the ESM-2 classifier on the cluster split
  4. Reports overall AUC + length-binned AUC (the length confound)

Usage:
    /opt/anaconda3/bin/python3 scripts/eval_homology.py --data data
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

from train_esm import embed_sequences

K = 5
JACCARD_THRESHOLD = 0.8


def kmer_set(seq: str, k: int = K) -> set[str]:
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def cluster_sequences(seqs: list[str]) -> list[list[int]]:
    """Greedy clustering by k-mer Jaccard >= threshold."""
    sets = [kmer_set(s) for s in seqs]
    clusters: list[list[int]] = []
    assigned = [False] * len(seqs)
    for i in range(len(seqs)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(seqs)):
            if assigned[j]:
                continue
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            if union and inter / union >= JACCARD_THRESHOLD:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    return clusters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    # Load.
    pos_seqs = []
    with open(args.data / "grampa.csv") as fh:
        for r in csv.DictReader(fh):
            pos_seqs.append(r["sequence"])
    neg_seqs = []
    for line in (args.data / "non_amp_short.fasta").read_text().splitlines():
        if line.startswith(">"):
            continue
        neg_seqs.append(line.strip().upper())
    neg_seqs = neg_seqs[: len(pos_seqs)]  # balance
    print(f"data: {len(pos_seqs)} AMP + {len(neg_seqs)} non-AMP (short)")

    # Cluster the AMPs.
    print(f"clustering {len(pos_seqs)} AMPs (k={K}, Jaccard>={JACCARD_THRESHOLD})...")
    clusters = cluster_sequences(pos_seqs)
    n_singletons = sum(1 for c in clusters if len(c) == 1)
    print(f"  {len(clusters)} clusters, {n_singletons} singletons, "
          f"largest {max(len(c) for c in clusters)}")

    # Split by cluster.
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(clusters))
    n_train_clusters = int(0.8 * len(clusters))
    train_clusters = set(perm[:n_train_clusters])
    train_idx = [i for ci, c in enumerate(clusters) if ci in train_clusters for i in c]
    test_idx = [i for ci, c in enumerate(clusters) if ci not in train_clusters for i in c]
    print(f"  train {len(train_idx)} seqs ({len(train_clusters)} clusters), "
          f"test {len(test_idx)} seqs ({len(clusters) - n_train_clusters} clusters)")


    # negatives: random split
    neg_perm = rng.permutation(len(neg_seqs))
    n_tr = int(0.8 * len(neg_seqs))
    neg_tr = [neg_seqs[i] for i in neg_perm[:n_tr]]
    neg_te = [neg_seqs[i] for i in neg_perm[n_tr:]]

    # Embed everything.
    print("loading ESM-2...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    X_tr = np.vstack([embed_sequences([pos_seqs[i] for i in train_idx], tokenizer, model),
                      embed_sequences(neg_tr, tokenizer, model)])
    y_tr = np.array([1] * len(train_idx) + [0] * len(neg_tr))
    X_te = np.vstack([embed_sequences([pos_seqs[i] for i in test_idx], tokenizer, model),
                      embed_sequences(neg_te, tokenizer, model)])
    y_te = np.array([1] * len(test_idx) + [0] * len(neg_te))

    clf = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    print(f"\nhomology-controlled AUC: {roc_auc_score(y_te, prob):.3f}")

    # Length-binned AUC (the length confound).
    lens = np.array([len(pos_seqs[i]) for i in test_idx] + [len(s) for s in neg_te])
    print("\nlength-binned AUC:")
    for lo, hi in [(5, 20), (21, 50), (51, 100)]:
        mask = (lens >= lo) & (lens <= hi)
        if mask.sum() > 10 and y_te[mask].sum() > 0 and (1 - y_te[mask]).sum() > 0:
            print(f"  [{lo:3d},{hi:3d}] n={mask.sum():4d}  AUC={roc_auc_score(y_te[mask], prob[mask]):.3f}")


if __name__ == "__main__":
    main()
