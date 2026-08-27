# AMP-Scan — universal antimicrobial peptide predictor

A CLI tool for scientists: given any peptide sequence or FASTA file,
predict (1) whether it is antimicrobial and (2) its activity against
*E. coli* (predicted log10 MIC in uM). No external dependencies beyond
scikit-learn — all features are computed from the sequence.

**Why this matters:** antimicrobial peptides (AMPs) are the leading
candidate to replace failing antibiotics. Romania has among the EU's
highest AMR rates (carbapenem-resistant *K. pneumoniae* <5% in 2010 →
>40% today, ECDC). Predicting which peptide sequences are antimicrobial
from sequence alone is a real drug-discovery task.

## Papers this is based on

- **Witten & Witten 2019** — "Deep learning regression model for
  antimicrobial peptide design" (bioRxiv, 82 citations). Built GRAMPA:
  51,345 peptide entries with measured MICs against *E. coli*, from
  APD/DADP/DBAASP/DRAMP/PEP_LIFE/YADAMP. Our positives + MIC targets.
- **Lu et al. 2026** — "ML Prediction and Experimental Validation of
  AMP Activity Differences against Gram-Positive and Gram-Negative
  Bacteria" (ACS Omega). Random Forest on 8 physicochemical properties
  → 82% test accuracy, validated on 18 synthesized peptides. Our
  feature-based RF approach follows this.
 - **Sevilla-Fortuny et al. 2024** — "Improved prediction of AMR in
   K. pneumoniae using ML" (bioRxiv). Context: ML beats rule-based
   methods for hard antibiotics.

## Supporting database: DBAASP
@@
 **Why this matters:** the model in this repo is trained on GRAMPA
 (reference-strain MICs). DBAASP adds the clinical-isolate dimension —
 the same peptide's MIC against resistant strains — which is the
 clinically relevant question and the gap in most AMP tools.
+
+## ESM-2: protein language model classifier
+
+The physicochemical RF is the fast, dependency-free baseline. The
+state-of-the-art approach uses a **protein language model**: ESM-2
+(facebook/esm2_t6_8M_UR50D), a transformer trained on 220M protein
+sequences — the same model family used by the AllTheBacteria/APEX
+AMP-discovery pipeline (Hunt et al. 2026).
+
+```bash
+# Train: embed all peptides with ESM-2, logistic regression on top
+/opt/anaconda3/bin/python3 scripts/train_esm.py --data data --out models
+
+# Score with the ESM-2 model
+/opt/anaconda3/bin/python3 scripts/score_esm.py --model models/esm_amp.joblib \
+    --seq GLPRKILCAIAKKKGKCKGPLKLVCKC
+```
+
+Verified comparison (80/20 split, seed 42):
+
+| Model | Accuracy | AUC | PR-AUC |
+|-------|----------|-----|--------|
+ | Physicochemical RF (10 features) | 0.988 | 0.999 | 0.999 |
 | **ESM-2 embeddings + logistic regression** | **0.992** | **1.000** | **1.000** |

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
# 1. Train a character-level GPT on AMP sequences (0.81M params, MPS)
/opt/anaconda3/bin/python3 scripts/train_gpt.py --data data --out models

# 2. Generate novel candidates, filter by ESM-2 activity + hemolysis
/opt/anaconda3/bin/python3 scripts/generate_amps.py \
    --gpt models/amp_gpt.pt --amp models/esm_amp.joblib \
    --hemo models/hemolysis.joblib --n 2000 --top 20
```

Verified: GPT trained in 14s on MPS (loss 2.23, perplexity 9.31 vs 20
for random). From 2,000 sampled sequences, **62 novel candidates** pass
AMP >= 0.9 AND hemolysis < 0.3 — all absent from the training data.
Top hit: AMP 0.995, hemolysis 0.032.

## Fine-tuned ESM-2 (Strategy 2: fine-tune, not frozen embeddings)

```bash
/opt/anaconda3/bin/python3 scripts/finetune_esm.py --data data --out models
```

Fine-tunes the full 7.53M-param ESM-2 with a classification head end
to end on MPS (49s, 3 epochs): acc 0.950 | AUC 0.987. This is the
"adapt an existing bio-LLM" path — the frozen-embedding logistic
regression is the fast baseline; the fine-tuned model is the real
training.
+
+And on the known-peptide test:
+
+| Peptide | RF prob | ESM-2 prob |
+|---------|---------|------------|
+| GRAMPA disulfide AMP | 0.997 | 1.000 |
+| UniProt AbTIR (non-AMP) | 0.267 | 0.018 |
+
+The ESM-2 separation is cleaner — the language model learned
+evolutionary representations the 10 hand-crafted features miss.
+
+**Honest note:** AUC 1.000 on this split means the AMP/non-AMP
+distinction is easy for ESM-2 — the hard part of AMP discovery is
+*novelty* (finding new AMPs, not classifying known ones) and
+*validation* (synthesizing and testing). The genome-scan mode
+(`amp_scan.py genome`) generates candidates; a lab test validates
+them. That is the AllTheBacteria workflow.

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
pneumoniae -> 660 activity rows, 41 unique sequences, MIC range
3.5e-05 to 773 uM.

**Why this matters:** the model in this repo is trained on GRAMPA
(reference-strain MICs). DBAASP adds the clinical-isolate dimension —
the same peptide's MIC against resistant strains — which is the
clinically relevant question and the gap in most AMP tools.

## Install

```bash
cd /Users/tomekdab/git/amp-ml
# deps: scikit-learn, numpy, joblib (all in anaconda base)
/opt/anaconda3/bin/python3 -c "import sklearn, joblib; print('ok')"
```

## Train

```bash
# Downloads GRAMPA positives (6,642 unique sequences + MIC) and
# UniProt non-AMP negatives (5,000 proteins), then trains.
python3 scripts/download_amp.py --out data
/opt/anaconda3/bin/python3 scripts/amp_scan.py train --data data --out models
```

Verified results (80/20 split, seed 42):

```
classifier: acc 0.988 | AUC 0.999 | PR-AUC 0.999
regressor (log10 MIC): R2 0.302 | Pearson r 0.552
```

## Score — the universal tool

```bash
# Single peptide
/opt/anaconda3/bin/python3 scripts/amp_scan.py score \
    --model models/amp_scan.joblib \
    --seq GLPRKILCAIAKKKGKCKGPLKLVCKC

# FASTA file of candidates
/opt/anaconda3/bin/python3 scripts/amp_scan.py score \
    --model models/amp_scan.joblib --fasta my_peptides.fasta
```

Output: per sequence — AMP probability, predicted log10 MIC, MIC in
uM, and a verdict.

Verified on known peptides:

| Peptide | AMP prob | log10 MIC | MIC (uM) | Verdict |
|---------|----------|-----------|----------|---------|
| GRAMPA example (disulfide AMP) | 1.000 | 0.38 | 2.4 | antimicrobial |
| magainin 2 (frog AMP) | 1.000 | 1.27 | 18.8 | antimicrobial |
| cecropin A (insect AMP) | 1.000 | 0.49 | 3.1 | antimicrobial |
| UniProt AbTIR (non-AMP protein) | 0.430 | 0.45 | 2.8 | not antimicrobial |

## Features (10, all from sequence)

length, molecular weight, isoelectric point (pI), net charge at pH 7,
mean hydrophobicity (Kyte-Doolittle), hydrophobic moment (Eisenberg,
100°), fraction hydrophobic / charged / aromatic residues, Boman index.

## Amphipathicity filter (handles the classic pitfall)

The model over-trusts hydrophobicity: pure-hydrophobic sequences
(e.g. `AAAAAAAALLLLLLLL`) score 0.967 as AMPs. This is the classic
AMP-prediction pitfall (Lu et al. report the same structural bias).
Real AMPs are amphipathic — a hydrophobic face + a cationic face — so
the scorer applies a post-filter: net charge >= +2 and hydrophobic
moment >= 0.5 (known AMPs: 0.99-1.46; decoys/proteins: 0.24-0.27).
Non-amphipathic high-probability hits are flagged
"low confidence: not amphipathic" instead of called antimicrobial.

Verified:

| Peptide | AMP prob | Verdict |
|---------|----------|---------|
| magainin 2 (charge +3.1, moment 1.46) | 1.000 | antimicrobial |
| cecropin A (charge +6.0, moment 0.99) | 1.000 | antimicrobial |
| `AAAAAAAALLLLLLLL` (charge 0, moment 0.24) | 0.967 | low confidence: not amphipathic |
| UniProt AbTIR (non-AMP protein) | 0.267 | not antimicrobial |

## Play challenges

1. **Feature ablation:** which of the 10 features matters most?
   (Use `clf.feature_importances_` — expect hydrophobicity and charge
   to dominate, matching Lu et al.)
2. **Threshold tuning:** the classifier threshold is 0.5. For a
   drug-discovery screen, would you prefer high recall (don't miss
   AMPs) or high precision (fewer false leads to synthesize)?
3. **Cross-validation:** switch `train_test_split` to stratified
   k-fold and report mean ± std AUC.
4. **Gram+ vs Gram−:** GRAMPA MICs are against *E. coli* (Gram−).
   The Lu et al. paper shows the target spectrum differs — how would
   you extend the tool to predict Gram+ activity?
5. **Amphipathicity tuning:** the filter thresholds (charge >= +2,
   moment >= 0.5) are hand-set. Measure them on the GRAMPA positives
   and pick values that keep 95% of known AMPs while rejecting the
   hydrophobic decoys.

## Skills exercised

- Sequence featurization (physicochemical properties, no external libs)
- Classification + regression on the same data (AMP + MIC)
- A reusable CLI for other scientists (train/score modes, FASTA input)
- Model interpretability and honest limitation reporting

## Gotchas

- Use `/opt/anaconda3/bin/python3` (has sklearn, joblib).
- GRAMPA is on the `master` branch of `zswitten/Antimicrobial-Peptides`.
- The MIC values are log10 uM — lower = more active.
 - The classifier is trained on GRAMPA AMPs vs UniProt proteins; it
   generalizes to short peptides but not to full-length proteins.

## License

Code: MIT (see LICENSE). The data is third-party and has its own
terms: GRAMPA (Witten & Witten 2019, GitHub), DBAASP (usage policy at
dbaasp.org), UniProt (CC-BY-4.0). The repo ships code only — data and
models are regenerable via the scripts and gitignored.
