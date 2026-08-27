#!/usr/bin/env python3
"""Generate novel AMP candidates with the trained GPT, filtered by
ESM-2 (activity) + hemolysis (toxicity).

Pipeline (the AllTheBacteria/APEX workflow, on a laptop):
  1. Sample N sequences from the character-level GPT
  2. Score each with the ESM-2 AMP classifier (activity)
  3. Score each with the ESM-2 hemolysis classifier (toxicity)
  4. Keep: AMP prob >= threshold AND hemolysis prob < threshold
  5. Report the survivors with novelty (not in training data)

Usage:
    /opt/anaconda3/bin/python3 scripts/generate_amps.py \
        --gpt models/amp_gpt.pt --amp models/esm_amp.joblib \
        --hemo models/hemolysis.joblib --n 2000 --top 20
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from train_esm import embed_sequences
from train_gpt import GPT, GPTConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpt", default="models/amp_gpt.pt", type=Path)
    ap.add_argument("--amp", default="models/esm_amp.joblib", type=Path)
    ap.add_argument("--hemo", default="models/hemolysis.joblib", type=Path)
    ap.add_argument("--n", default=2000, type=int, help="sequences to sample")
    ap.add_argument("--top", default=20, type=int, help="top candidates to show")
    ap.add_argument("--amp-threshold", default=0.9, type=float)
    ap.add_argument("--hemo-threshold", default=0.3, type=float)
    ap.add_argument("--temperature", default=1.0, type=float)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load GPT.
    ckpt = torch.load(args.gpt, map_location=device, weights_only=False)
    config = ckpt["config"]
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    model = GPT(config, len(stoi)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"GPT loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)")

    # Load classifiers.
    import joblib
    amp_clf = joblib.load(args.amp)["clf"]
    hemo_clf = joblib.load(args.hemo)["clf"]
    print("ESM-2 classifiers loaded")

    # Sample sequences.
    start = torch.tensor([[stoi["M"]]], dtype=torch.long, device=device)
    seqs = []
    with torch.no_grad():
        for _ in range(args.n):
            out = model.generate(start, max_new_tokens=45,
                                 temperature=args.temperature)
            s = "".join(itos[int(i)] for i in out[0].tolist()[1:])
            if 5 <= len(s) <= 50:
                seqs.append(s)
    print(f"sampled {len(seqs)} valid sequences (5-50 aa)")

    # Score with ESM-2.
    print("loading ESM-2 for scoring...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    esm = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    X = embed_sequences(seqs, tokenizer, esm)
    amp_prob = amp_clf.predict_proba(X)[:, 1]
    hemo_prob = hemo_clf.predict_proba(X)[:, 1]

    # Known sequences (for novelty check).
    known = set()
    with open("data/grampa.csv") as fh:
        for r in csv.DictReader(fh):
            known.add(r["sequence"].upper())

    # Filter.
    survivors = []
    for s, ap, hp in zip(seqs, amp_prob, hemo_prob):
        if ap >= args.amp_threshold and hp < args.hemo_threshold:
            survivors.append((s, ap, hp, s not in known))
    survivors.sort(key=lambda t: -t[1])

    print(f"\n{len(survivors)} candidates pass "
          f"(AMP>={args.amp_threshold}, hemolysis<{args.hemo_threshold})")
    print(f"{'#':<4} {'AMP prob':>8} {'Hemo':>6} {'novel':>6}  sequence")
    print("-" * 70)
    for i, (s, ap, hp, novel) in enumerate(survivors[: args.top], 1):
        print(f"{i:<4} {ap:>8.3f} {hp:>6.3f} {str(novel):>6}  {s}")


if __name__ == "__main__":
    main()
