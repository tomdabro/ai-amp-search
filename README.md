# AMP-Scan — antimicrobial peptide prediction and discovery

A CLI tool for scientists: given any peptide sequence or FASTA file,
predict (1) whether it is antimicrobial (AMP probability) and (2) its
activity against *E. coli* (predicted log10 MIC in uM). Includes a
 generative pipeline that designs novel AMP candidates, filtered for
 activity and toxicity.

> **For educational purposes only.** This project is a learning
> exercise and is not a validated research tool. The models are
> trained on public data and have NOT been experimentally validated;
> predictions are hypotheses for lab testing, not clinical results.
> Subject to change.

**Why this matters:** antimicrobial peptides (AMPs) are the leading
candidate to replace failing antibiotics. Southern and Eastern Europe
carry the EU's highest AMR burden (ECDC EARS-Net): Romania's
carbapenem-resistant *K. pneumoniae* rose from <5% (2010) to >40% in
recent years; Greece has reported ICU resistance rates of 50-80%
(WHONET-Greece); Italy consistently ranks among the highest-burden
countries for this pathogen. Predicting which peptide sequences are
antimicrobial from sequence alone is a real drug-discovery task.

## Papers this is based on

- **Witten J, Witten Z. 2019.** Deep learning regression model for
  antimicrobial peptide design. *bioRxiv*.
  [DOI: 10.1101/692681](https://doi.org/10.1101/692681)
  (82 citations). Built GRAMPA: 51,345 peptide entries with measured
  MICs against *E. coli*, from APD/DADP/DBAASP/DRAMP/PEP_LIFE/YADAMP.
  Our positives + MIC targets.
- **Lu P, Li W, Zubair M, Li L, Han G, Chu Y. 2026.** Machine Learning
  Prediction and Experimental Validation of Antimicrobial Peptide
  Activity Differences against Gram-Positive and Gram-Negative
  Bacteria. *ACS Omega* 11(32):47517-47527.
  [DOI: 10.1021/acsomega.6c01772](https://doi.org/10.1021/acsomega.6c01772)
  Random Forest on 8 physicochemical properties → 82% test accuracy,
  validated on 18 synthesized peptides. Our feature-based RF approach
  follows this.
- **Sevilla-Fortuny J, González-Candelas F, García-González N. 2024.**
  Improved prediction of antimicrobial resistance in *Klebsiella
  pneumoniae* using machine learning. *bioRxiv*.
  [DOI: 10.1101/2024.12.10.627815](https://doi.org/10.1101/2024.12.10.627815).
  Context: ML beats rule-based methods
  for hard antibiotics.
- **Pham HT, Huynh B, Nguyen-Vo TH. 2026.** GenPept-Curated-2025: A
  Benchmark Dataset for Antimicrobial Peptide Prediction with
  Homology-Controlled Partitioning.
  [DOI: 10.64898/2026.04.25.720793](https://doi.org/10.64898/2026.04.25.720793).
  Names homology leakage, negative-set reliability, and length-dependent
  imbalance as the three evaluation pitfalls. Our homology-controlled
  evaluation follows its protocol.
- **Xie P, Yao L, Guan J, Chung CR, Zhao Z, Long F, Sun Z, Lee TY,
  Chiang YC. 2025.** ConsAMPHemo: A computational framework for
  predicting hemolysis of antimicrobial peptides based on machine
  learning approaches. *Protein Science* 34(7).
  [DOI: 10.1002/pro.70087](https://doi.org/10.1002/pro.70087).
  Our toxicity classifier uses its data.

## Supporting database: DBAASP

**DBAASP** (Database of Antimicrobial Activity and Structure of
Peptides, https://dbaasp.org) is a manually curated database of
peptide antimicrobial activity with measured MICs against specific
target species — including clinically relevant resistant strains:

- *K. pneumoniae* ATCC BAA-2146 (NDM-1 carbapenemase producer)
- *K. pneumoniae* NCTC 13443 (KPC carbapenemase producer)
- *E. coli* ESBL, *S. aureus* MR (MRSA)
- CCARM / CUN / XJ clinical collections

Fetch real AMP-MIC data for any target species:

```bash
# 1,060 peptides with K. pneumoniae activity in DBAASP
python3 scripts/fetch_dbaasp.py --species "Klebsiella pneumoniae" \
    --out data/dbaasp_kp.csv

# MRSA specifically
python3 scripts/fetch_dbaasp.py --species "Staphylococcus aureus MR" \
    --out data/dbaasp_mrsa.csv
```

Each row: peptide sequence, target species, MIC, unit (uM or ug/ml),
medium (TSB/MHB/BHIB). Verified: 60 peptides fetched for K.
pneumoniae → 660 activity rows, 41 unique sequences, MIC range
3.5e-05 to 773 uM.

**Why this matters:** the model in this repo is trained on GRAMPA
(reference-strain MICs). DBAASP adds the clinical-isolate dimension —
the same peptide's MIC against resistant strains — which is the
clinically relevant question and the gap in most AMP tools.

## ESM-2: protein language model classifier

The physicochemical RF is the fast, dependency-free baseline. The
state-of-the-art approach uses a **protein language model**: ESM-2
(facebook/esm2_t6_8M_UR50D), a transformer trained on 220M protein
sequences — the same model family used by the AllTheBacteria/APEX
AMP-discovery pipeline (Hunt et al. 2026).

```bash
# Train: embed all peptides with ESM-2, logistic regression on top
/opt/anaconda3/bin/python3 scripts/train_esm.py --data data --out models

# Score with the ESM-2 model
/opt/anaconda3/bin/python3 scripts/score_esm.py --model models/esm_amp.joblib \
    --seq GLPRKILCAIAKKKGKCKGPLKLVCKC
```

Verified comparison (80/20 split, seed 42):

| Model | Accuracy | AUC | PR-AUC |
|-------|----------|-----|--------|
| Physicochemical RF (10 features) | 0.988 | 0.999 | 0.999 |
| ESM-2 embeddings + logistic regression | 0.992 | 1.000 | 1.000 |

**But the naive split is leaky.** GRAMPA contains near-identical
variants of the same peptide family; a random split lets the model
memorize families. With a **homology-controlled split** (k-mer
Jaccard >= 0.8 clustering, split by cluster — the protocol named by
GenPept-Curated-2025), the honest ESM-2 numbers are:

```
homology-controlled AUC: 0.971
length-binned AUC:
  [  5, 20] n=1069  AUC=0.959
  [ 21, 50] n=1543  AUC=0.972
```

The model still works across length bins (not just predicting length),
but 0.971 is the number to quote — not 1.000. Run it yourself:

```bash
/opt/anaconda3/bin/python3 scripts/eval_homology.py --data data
```

And on the known-peptide test:

| Peptide | RF prob | ESM-2 prob |
|---------|---------|------------|
| GRAMPA disulfide AMP | 0.997 | 1.000 |
| UniProt AbTIR (non-AMP) | 0.267 | 0.018 |

The ESM-2 separation is cleaner — the language model learned
evolutionary representations the 10 hand-crafted features miss.

## Hemolysis (toxicity) prediction

An AMP drug candidate needs potency AND low toxicity. The hemolysis
classifier (ConsAMPHemo S1 data, Xie et al. 2025, Protein Science)
uses the same ESM-2 embeddings:

```bash
/opt/anaconda3/bin/python3 scripts/train_hemolysis.py \
    --data data/hemolysis_train.csv --out models
```

Verified: acc 0.994 | AUC 0.998 | PR-AUC 0.998 (884 peptides, balanced).

## Generative design: GPT + ESM-2 + hemolysis filter

The full discovery pipeline (the AllTheBacteria/APEX workflow, on a
laptop):

```bash
# 1. Train a character-level GPT on AMP sequences (2.69M params, MPS)
/opt/anaconda3/bin/python3 scripts/train_gpt.py --data data --out models

# 2. Generate novel candidates, filter by ESM-2 activity + hemolysis
/opt/anaconda3/bin/python3 scripts/generate_amps.py \
    --gpt models/amp_gpt.pt --amp models/esm_amp.joblib \
    --hemo models/hemolysis.joblib --n 2000 --top 20
```

Verified: GPT trained to convergence on MPS (100 epochs, val split,
cosine LR schedule) — best val perplexity 10.13 at epoch 40, best
checkpoint saved. From 2,000 sampled sequences, **80 novel
candidates** pass AMP >= 0.9 AND hemolysis < 0.3 — all absent from the
training data. Top hit: AMP 0.987, hemolysis 0.112.

## Fine-tuned ESM-2 (Strategy 2: fine-tune, not frozen embeddings)

```bash
/opt/anaconda3/bin/python3 scripts/finetune_esm.py --data data --out models
```

Fine-tunes the full 33.5M-param ESM-2 (t12_35M) with a classification
head end to end on MPS (6 epochs): acc 0.956 | AUC 0.990. This is the
"adapt an existing bio-LLM" path — the frozen-embedding logistic
regression is the fast baseline; the fine-tuned model is the real
training.

## Trained models on Hugging Face

All trained models are published on the Hugging Face Hub:
**https://huggingface.co/tomekdab/amp-scan-models**

| File | What | Size | Verified |
|------|------|------|----------|
| `esm_amp.joblib` | AMP classifier (frozen ESM-2 + LR) | 3 KB | acc 0.992, honest AUC 0.971 |
| `hemolysis.joblib` | toxicity classifier | 3 KB | acc 0.994, AUC 0.998 |
| `amp_gpt.pt` | converged GPT (2.69M params) | 10.8 MB | val perplexity 10.13 |
| `esm_finetuned.pt` | 35M ESM-2 fine-tuned end-to-end | 134 MB | acc 0.956, AUC 0.990 |

The GitHub repo ships code + data only (models are regenerable via
`scripts/train_*.py`); the Hub hosts the trained artifacts.

## Install

```bash
pip install -r requirements.txt
# deps: numpy, scikit-learn, joblib, matplotlib, pandas, torch, transformers
```

## Repository layout

```
scripts/
  download_amp.py       GRAMPA positives + UniProt negatives
  fetch_dbaasp.py      DBAASP clinical-isolate AMP-MIC data
  amp_scan.py          train/score/plots/genome (physicochemical RF)
  train_esm.py         ESM-2 embeddings + logistic regression
  score_esm.py         score with the ESM-2 model
  eval_homology.py     homology-controlled evaluation (the honest number)
  train_hemolysis.py   toxicity classifier
  train_gpt.py         character-level GPT on AMP sequences
  generate_amps.py     GPT generation + ESM-2/hemolysis filter
  finetune_esm.py      ESM-2 fine-tuned end-to-end
data/                  training data (regenerable, gitignored)
models/                trained models (regenerable, gitignored)
plots/                 evaluation plots (regenerable, gitignored)
```

## Skills exercised

- Sequence featurization (physicochemical properties, no external libs)
- Protein language models (ESM-2 embeddings, fine-tuning)
- Generative modeling (character-level GPT on biological sequences)
- Classification + regression on the same data (AMP + MIC)
- Honest evaluation: homology-controlled splits, length-binned AUC
- A reusable CLI for other scientists (train/score/plots/genome modes)
- Model publishing on the Hugging Face Hub

## Gotchas

- Use `/opt/anaconda3/bin/python3` (has sklearn, torch, transformers).
- GRAMPA is on the `master` branch of `zswitten/Antimicrobial-Peptides`.
- The MIC values are log10 uM — lower = more active.
- The classifier is trained on GRAMPA AMPs vs UniProt proteins; it
  generalizes to short peptides but not to full-length proteins.
- The amphipathicity filter (charge >= +2, moment >= 0.5) flags
  pure-hydrophobic decoys as "low confidence" instead of antimicrobial.

## License

Code: MIT (see LICENSE). The data is third-party and has its own
terms: GRAMPA (Witten & Witten 2019, GitHub), DBAASP (usage policy at
dbaasp.org), UniProt (CC-BY-4.0). The repo ships code only — data and
models are regenerable via the scripts and gitignored.
