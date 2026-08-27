#!/usr/bin/env python3
"""Download training data for AMP-Scan.

Positives: GRAMPA (Witten & Witten 2019) — 51,345 peptide entries with
log10 MIC (uM) against E. coli, from APD/DADP/DBAASP/DRAMP/PEP_LIFE/
YADAMP databases.
Negatives: UniProt reviewed proteins (length 40-400 aa) that are NOT
annotated antimicrobial (keyword KW-0929) — sampled to match.

Outputs:
    data/grampa.csv        positives: sequence, mic (log10 uM)
    data/non_amp.fasta     negatives: sampled UniProt sequences

Usage:
    python3 scripts/download_amp.py --out data
"""

import argparse
import csv
import io
import random
import urllib.request
from pathlib import Path

GRAMPA_URL = ("https://raw.githubusercontent.com/zswitten/"
              "Antimicrobial-Peptides/master/data/grampa.csv")
UNIPROT_QUERY = ("https://rest.uniprot.org/uniprotkb/stream?format=fasta&"
                 "query=reviewed:true%20AND%20length:%5B40%20TO%20400%5D%20"
                 "AND%20NOT%20keyword:KW-0929")
N_NEGATIVES = 5000


def fetch(url: str) -> bytes:
    print(f"  fetching {url[:80]}...")
    with urllib.request.urlopen(url) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--n-negatives", default=N_NEGATIVES, type=int)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. GRAMPA positives (with MIC).
    raw = fetch(GRAMPA_URL).decode()
    rows = list(csv.DictReader(io.StringIO(raw)))
    seen = set()
    pos = []
    for r in rows:
        seq = (r.get("sequence") or "").strip().upper()
        if len(seq) < 5 or seq in seen:
            continue
        seen.add(seq)
        try:
            mic = float(r["value"])
        except (ValueError, KeyError):
            continue
        pos.append((seq, mic))
    print(f"  {len(pos)} unique positive sequences with MIC")

    # 2. UniProt negatives (non-AMP proteins).
    fasta = fetch(UNIPROT_QUERY).decode()
    neg = []
    cur = []
    for line in fasta.splitlines():
        if line.startswith(">"):
            if cur:
                neg.append("".join(cur))
            cur = []
        else:
            cur.append(line.strip())
    if cur:
        neg.append("".join(cur))
    neg = [s.upper() for s in neg if 5 <= len(s) <= 400]
    random.seed(42)
    random.shuffle(neg)
    neg = neg[: args.n_negatives]
    print(f"  {len(neg)} negative sequences (non-AMP proteins)")

    # 3. Write outputs.
    with open(args.out / "grampa.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", "mic"])
        w.writerows(pos)
    with open(args.out / "non_amp.fasta", "w") as fh:
        for i, s in enumerate(neg):
            fh.write(f">nonamp_{i}\n{s}\n")

    print(f"done -> {args.out / 'grampa.csv'} ({len(pos)} rows), "
          f"{args.out / 'non_amp.fasta'} ({len(neg)} sequences)")


if __name__ == "__main__":
    main()
