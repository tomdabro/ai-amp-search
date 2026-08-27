#!/usr/bin/env python3
"""AMP-Scan: predict antimicrobial activity of any peptide sequence.

A universal CLI for scientists: given a peptide sequence or FASTA file,
predict (1) whether it is antimicrobial (AMP probability) and (2) its
activity against E. coli (predicted log10 MIC in uM), using a Random
Forest trained on physicochemical features (the approach validated by
Lu et al. 2026: RF on 8 key properties, 82% test accuracy).

Two modes:

    # Train the model from GRAMPA + UniProt negatives
    python3 scripts/amp_scan.py train --data data --out models

    # Score a peptide or FASTA file
    python3 scripts/amp_scan.py score --model models --seq GLPRKILCAIAKKKGKCKGPLKLVCKC
    python3 scripts/amp_scan.py score --model models --fasta my_peptides.fasta

Features (all computed from the sequence, no external libraries):
length, molecular weight, isoelectric point (pI), net charge at pH 7,
mean hydrophobicity (Kyte-Doolittle), hydrophobic moment, fraction of
hydrophobic / charged / aromatic residues, Boman index.

Usage:
    python3 scripts/amp_scan.py train --data data --out models
    python3 scripts/amp_scan.py score --model models --seq <PEPTIDE>
    python3 scripts/amp_scan.py score --model models --fasta <FILE>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             roc_auc_score, r2_score)
from sklearn.model_selection import train_test_split

# Amino acid properties (Kyte-Doolittle hydrophobicity, molecular weight,
# pKa values for charge at pH 7).
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2}
MW = {"A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
      "Q": 146.15, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17,
      "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
      "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15}
PK_N_TERM, PK_C_TERM = 9.69, 2.34
PK_R, PK_H, PK_K, PK_D, PK_E, PK_C = 12.48, 6.00, 10.53, 3.65, 4.25, 8.18
HYDROPHOBIC = {"A", "I", "L", "M", "F", "W", "Y", "V", "C"}
CHARGED = {"R", "K", "D", "E", "H"}
AROMATIC = {"F", "W", "Y"}

FEATURES = ["length", "mw", "pI", "charge_pH7", "hydrophobicity",
            "hydrophobic_moment", "frac_hydrophobic", "frac_charged",
            "frac_aromatic", "boman"]


def _charge_at_pH7(seq: str) -> float:
    """Net charge at pH 7 (Henderson-Hasselbalch, side chains + termini)."""
    q = 0.0
    for aa in seq:
        if aa == "R":
            q += 1 / (1 + 10 ** (7 - PK_R))
        elif aa == "H":
            q += 1 / (1 + 10 ** (7 - PK_H))
        elif aa == "K":
            q += 1 / (1 + 10 ** (7 - PK_K))
        elif aa == "D":
            q -= 1 / (1 + 10 ** (PK_D - 7))
        elif aa == "E":
            q -= 1 / (1 + 10 ** (PK_E - 7))
        elif aa == "C":
            q -= 1 / (1 + 10 ** (PK_C - 7))
    # termini
    q += 1 / (1 + 10 ** (7 - PK_N_TERM))
    q -= 1 / (1 + 10 ** (PK_C_TERM - 7))
    return q


def _pI(seq: str) -> float:
    """Isoelectric point: pH where net charge ~ 0 (bisection)."""
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2
        q = 0.0
        for aa in seq:
            if aa == "R":
                q += 1 / (1 + 10 ** (mid - PK_R))
            elif aa == "H":
                q += 1 / (1 + 10 ** (mid - PK_H))
            elif aa == "K":
                q += 1 / (1 + 10 ** (mid - PK_K))
            elif aa == "D":
                q -= 1 / (1 + 10 ** (PK_D - mid))
            elif aa == "E":
                q -= 1 / (1 + 10 ** (PK_E - mid))
            elif aa == "C":
                q -= 1 / (1 + 10 ** (PK_C - mid))
        q += 1 / (1 + 10 ** (mid - PK_N_TERM))
        q -= 1 / (1 + 10 ** (PK_C_TERM - mid))
        if q > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _hydrophobic_moment(seq: str, angle: float = 100.0) -> float:
    """Eisenberg hydrophobic moment (helical, 100 deg per residue)."""
    rad = np.deg2rad(angle)
    hx = sum(KD.get(a, 0.0) * np.cos(i * rad) for i, a in enumerate(seq))
    hy = sum(KD.get(a, 0.0) * np.sin(i * rad) for i, a in enumerate(seq))
    return float(np.hypot(hx, hy) / len(seq))


def features(seq: str) -> list[float]:
    """Compute the 10 physicochemical features for one sequence."""
    seq = seq.upper()
    n = len(seq)
    kd = [KD.get(a, 0.0) for a in seq]
    return [
        n,
        sum(MW.get(a, 0.0) for a in seq) - 18.02 * (n - 1),  # mw (minus H2O)
        _pI(seq),
        _charge_at_pH7(seq),
        float(np.mean(kd)),
        _hydrophobic_moment(seq),
        sum(1 for a in seq if a in HYDROPHOBIC) / n,
        sum(1 for a in seq if a in CHARGED) / n,
        sum(1 for a in seq if a in AROMATIC) / n,
        float(np.mean([abs(v) for v in kd])),  # Boman index (approx)
    ]


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Parse a FASTA file into (header, sequence) pairs."""
    out = []
    header, cur = None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if header is not None:
                out.append((header, "".join(cur)))
            header, cur = line[1:], []
        elif line:
            cur.append(line)
    if header is not None:
        out.append((header, "".join(cur)))
    return out


def train(data_dir: Path, out_dir: Path, seed: int) -> None:
    """Train classifier (AMP vs non-AMP) + regressor (log10 MIC)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Positives: GRAMPA with MIC.
    pos_seqs, pos_mic = [], []
    with open(data_dir / "grampa.csv") as fh:
        for r in csv.DictReader(fh):
            pos_seqs.append(r["sequence"])
            pos_mic.append(float(r["mic"]))
    # Negatives: UniProt non-AMP proteins.
    neg_seqs = [s for _, s in read_fasta(data_dir / "non_amp.fasta")]

    X_pos = np.array([features(s) for s in pos_seqs])
    X_neg = np.array([features(s) for s in neg_seqs])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))
    print(f"train: {len(pos_seqs)} AMP + {len(neg_seqs)} non-AMP, "
          f"{X.shape[1]} features")

    # --- Classifier ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, (prob >= 0.5).astype(int))
    auc_ = roc_auc_score(y_te, prob)
    prec, rec, _ = precision_recall_curve(y_te, prob)
    print(f"classifier: acc {acc:.3f} | AUC {auc_:.3f} | PR-AUC {auc(rec, prec):.3f}")

    # --- Regressor (MIC, log10 uM) ---
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_pos, pos_mic, test_size=0.2, random_state=seed)
    reg = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    reg.fit(Xr_tr, yr_tr)
    pred = reg.predict(Xr_te)
    r2 = r2_score(yr_te, pred)
    corr = float(np.corrcoef(yr_te, pred)[0, 1])
    print(f"regressor (log10 MIC): R2 {r2:.3f} | Pearson r {corr:.3f}")

    # --- Save ---
    import joblib
    joblib.dump({"clf": clf, "reg": reg, "features": FEATURES},
                out_dir / "amp_scan.joblib")
    print(f"saved model -> {out_dir / 'amp_scan.joblib'}")


def score(model_path: Path, seq: str | None, fasta: Path | None) -> None:
    """Score one peptide or a FASTA file."""
    import joblib
    model = joblib.load(model_path)
    clf, reg = model["clf"], model["reg"]

    items = []
    if seq:
        items.append(("peptide", seq.upper()))
    if fasta:
        items.extend(read_fasta(fasta))

    if not items:
        sys.exit("pass --seq or --fasta")

    print(f"{'id':<24} {'AMP prob':>9} {'log10 MIC':>10} {'MIC (uM)':>10}  verdict")
    print("-" * 70)
    for name, s in items:
        s = "".join(c for c in s if c.isalpha())
        if len(s) < 5:
            print(f"{name:<24} {'<5 aa, skipped':>30}")
            continue
        f = np.array([features(s)]).reshape(1, -1)
        p = float(clf.predict_proba(f)[0, 1])
        mic = float(reg.predict(f)[0])
        # Amphipathicity check: real AMPs are cationic + amphipathic
        # (hydrophobic face + charged face). Pure-hydrophobic decoys
        # (e.g. AAAAAAAALLLLLLLL) fool the hydrophobicity feature, so
        # require net charge >= +2 and a hydrophobic moment >= 0.5
        # (known AMPs: 0.99-1.46; decoys/proteins: 0.24-0.27).
        charge = _charge_at_pH7(s)
        moment = _hydrophobic_moment(s)
        amphipathic = charge >= 2.0 and moment >= 0.5
        if p >= 0.5 and amphipathic:
            verdict = "antimicrobial"
        elif p >= 0.5:
            verdict = "antimicrobial (low confidence: not amphipathic)"
        else:
            verdict = "not antimicrobial"
        print(f"{name:<24} {p:>9.3f} {mic:>10.2f} {10 ** mic:>10.1f}  {verdict}")


# --- Genetic code for genome scanning ---
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def _find_orfs(seq: str, frame: int, strand: str) -> list[dict]:
    """Find ORFs (M...stop) in one frame of a nucleotide sequence."""
    orfs = []
    i = frame
    n = len(seq)
    while i + 3 <= n:
        codon = seq[i:i + 3]
        if codon == "ATG":
            j = i
            aa = []
            while j + 3 <= n:
                c = seq[j:j + 3]
                a = CODON_TABLE.get(c, "X")
                if a == "*":
                    break
                aa.append(a)
                j += 3
            peptide = "".join(aa)
            if 5 <= len(peptide) <= 400:
                orfs.append({"start": i, "end": j, "strand": strand,
                             "frame": frame, "peptide": peptide})
            i = j + 3
        else:
            i += 3
    return orfs


def genome(model_path: Path, fasta: Path, top: int) -> None:
    """Scan a whole-genome FASTA for potential AMPs (6-frame ORF scan)."""
    import joblib
    model = joblib.load(model_path)
    clf, reg = model["clf"], model["reg"]

    hits = []
    feat_cache: dict[str, np.ndarray] = {}
    for header, seq in read_fasta(fasta):
        seq = "".join(c for c in seq.upper() if c in "ACGTN")
        print(f"scanning {header} ({len(seq):,} bp)")
        for strand, s in (("+", seq), ("-", seq.translate(COMPLEMENT)[::-1])):
            for frame in range(3):
                for orf in _find_orfs(s, frame, strand):
                    p = orf["peptide"]
                    if p not in feat_cache:
                        feat_cache[p] = np.array([features(p)]).reshape(1, -1)
                    f = feat_cache[p]
                    prob = float(clf.predict_proba(f)[0, 1])
                    mic = float(reg.predict(f)[0])
                    charge = _charge_at_pH7(p)
                    moment = _hydrophobic_moment(p)
                    amphipathic = charge >= 2.0 and moment >= 0.5
                    if prob >= 0.5 and amphipathic:
                        hits.append({**orf, "prob": prob, "mic": mic,
                                     "charge": charge, "moment": moment})

    hits.sort(key=lambda h: -h["prob"])
    print(f"\n{len(hits)} amphipathic AMP candidates found\n")
    print(f"{'strand':<7} {'frame':<6} {'start':<8} {'end':<8} "
          f"{'AMP prob':>9} {'MIC(uM)':>8}  peptide")
    print("-" * 80)
    for h in hits[:top]:
        print(f"{h['strand']:<7} {h['frame']:<6} {h['start']:<8} {h['end']:<8} "
              f"{h['prob']:>9.3f} {10 ** h['mic']:>8.1f}  {h['peptide'][:40]}")


def plots(data_dir: Path, model_path: Path, out_dir: Path, seed: int) -> None:
    """Generate evaluation plots (ROC, PR, features, MIC, amphipathicity)."""
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    model = joblib.load(model_path)
    clf, reg = model["clf"], model["reg"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild the dataset for evaluation.
    pos_seqs, pos_mic = [], []
    with open(data_dir / "grampa.csv") as fh:
        for r in csv.DictReader(fh):
            pos_seqs.append(r["sequence"])
            pos_mic.append(float(r["mic"]))
    neg_seqs = [s for _, s in read_fasta(data_dir / "non_amp.fasta")]
    X_pos = np.array([features(s) for s in pos_seqs])
    X_neg = np.array([features(s) for s in neg_seqs])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]

    # 1. ROC curve
    fpr, tpr, _ = roc_curve(y_te, prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_te, prob):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="AMP classifier — ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=150)
    plt.close(fig)

    # 2. Precision-recall curve
    prec, rec, _ = precision_recall_curve(y_te, prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=2, label=f"PR-AUC = {auc(rec, prec):.3f}")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="AMP classifier — precision-recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pr.png", dpi=150)
    plt.close(fig)

    # 3. Feature importances
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([FEATURES[i] for i in order], imp[order])
    ax.invert_yaxis()
    ax.set(xlabel="Mean decrease in impurity", title="Feature importance")
    fig.tight_layout()
    fig.savefig(out_dir / "features.png", dpi=150)
    plt.close(fig)

    # 4. MIC regression scatter
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_pos, np.array(pos_mic), test_size=0.2, random_state=seed)
    reg.fit(Xr_tr, yr_tr)
    pred = reg.predict(Xr_te)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(yr_te, pred, s=8, alpha=0.4)
    lims = [min(yr_te.min(), pred.min()), max(yr_te.max(), pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set(xlabel="Observed log10 MIC", ylabel="Predicted log10 MIC",
           title=f"MIC regression (r = {np.corrcoef(yr_te, pred)[0, 1]:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "mic_scatter.png", dpi=150)
    plt.close(fig)

    # 5. Amphipathicity landscape (charge vs hydrophobic moment)
    charges = np.array([_charge_at_pH7(s) for s in pos_seqs[:500]])
    moments = np.array([_hydrophobic_moment(s) for s in pos_seqs[:500]])
    n_charges = np.array([_charge_at_pH7(s) for s in neg_seqs[:500]])
    n_moments = np.array([_hydrophobic_moment(s) for s in neg_seqs[:500]])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(charges, moments, s=6, alpha=0.4, label="AMPs")
    ax.scatter(n_charges, n_moments, s=6, alpha=0.3, label="non-AMPs")
    ax.axhline(0.5, color="r", ls="--", lw=1, label="moment threshold")
    ax.axvline(2.0, color="r", ls="--", lw=1, label="charge threshold")
    ax.set(xlabel="Net charge at pH 7", ylabel="Hydrophobic moment",
           title="Amphipathicity filter")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "amphipathicity.png", dpi=150)
    plt.close(fig)

    print(f"saved plots -> {out_dir}/ (roc.png, pr.png, features.png, "
          f"mic_scatter.png, amphipathicity.png)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="train the model")
    p_train.add_argument("--data", default="data", type=Path)
    p_train.add_argument("--out", default="models", type=Path)
    p_train.add_argument("--seed", default=42, type=int)

    p_score = sub.add_parser("score", help="score a peptide or FASTA")
    p_score.add_argument("--model", default="models/amp_scan.joblib", type=Path)
    p_score.add_argument("--seq", type=str, help="single peptide sequence")
    p_score.add_argument("--fasta", type=Path, help="FASTA file of peptides")

    p_genome = sub.add_parser("genome", help="scan a genome FASTA for AMPs")
    p_genome.add_argument("--model", default="models/amp_scan.joblib", type=Path)
    p_genome.add_argument("--fasta", required=True, type=Path,
                          help="whole-genome FASTA (nucleotide)")
    p_genome.add_argument("--top", default=20, type=int,
                          help="show top N candidates (default 20)")

    p_plots = sub.add_parser("plots", help="generate evaluation plots")
    p_plots.add_argument("--data", default="data", type=Path)
    p_plots.add_argument("--model", default="models/amp_scan.joblib", type=Path)
    p_plots.add_argument("--out", default="plots", type=Path)
    p_plots.add_argument("--seed", default=42, type=int)

    args = ap.parse_args()
    if args.mode == "train":
        train(args.data, args.out, args.seed)
    elif args.mode == "score":
        score(args.model, args.seq, args.fasta)
    elif args.mode == "genome":
        genome(args.model, args.fasta, args.top)
    else:
        plots(args.data, args.model, args.out, args.seed)


if __name__ == "__main__":
    main()
