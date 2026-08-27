#!/usr/bin/env python3
"""Score peptides with the ESM-2 AMP classifier.

Loads the ESM-2 embeddings + logistic regression trained by
train_esm.py and scores a peptide or FASTA file.

Usage:
    python3 scripts/score_esm.py --model models/esm_amp.joblib \
        --seq GLPRKILCAIAKKKGKCKGPLKLVCKC
    python3 scripts/score_esm.py --model models/esm_amp.joblib \
        --fasta my_peptides.fasta
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from train_esm import embed_sequences


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/esm_amp.joblib", type=Path)
    ap.add_argument("--seq", type=str, help="single peptide sequence")
    ap.add_argument("--fasta", type=Path, help="FASTA file of peptides")
    args = ap.parse_args()

    import joblib
    model = joblib.load(args.model)
    clf = model["clf"]
    model_name = model["model_name"]

    items = []
    if args.seq:
        items.append(("peptide", args.seq.upper()))
    if args.fasta:
        from amp_scan import read_fasta
        items.extend(read_fasta(args.fasta))
    if not items:
        sys.exit("pass --seq or --fasta")

    print(f"loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm = AutoModel.from_pretrained(model_name)

    seqs = ["".join(c for c in s if c.isalpha()) for _, s in items]
    X = embed_sequences(seqs, tokenizer, esm)
    prob = clf.predict_proba(X)[:, 1]

    print(f"{'id':<24} {'AMP prob':>9}  verdict")
    print("-" * 50)
    for (name, _), p in zip(items, prob):
        verdict = "antimicrobial" if p >= 0.5 else "not antimicrobial"
        print(f"{name:<24} {p:>9.3f}  {verdict}")


if __name__ == "__main__":
    main()
