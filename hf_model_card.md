---
license: mit
language: en
tags:
  - biology
  - antimicrobial-peptides
  - esm2
  - protein-language-model
  - drug-discovery
library_name: transformers
pipeline_tag: text-classification
---

# AMP-Scan models

Trained models for antimicrobial peptide (AMP) prediction, part of the
[ai-amp-search](https://github.com/tomdabro/ai-amp-search) project.

## Models in this repo

| File | What it is | Size | Verified |
|------|-----------|------|----------|
| `esm_amp.joblib` | ESM-2 embeddings + logistic regression (AMP classifier) | 3 KB | acc 0.992, AUC 1.000 (naive split) / 0.971 (homology-controlled) |
| `hemolysis.joblib` | ESM-2 embeddings + logistic regression (toxicity) | 3 KB | acc 0.994, AUC 0.998 |
| `amp_gpt.pt` | Character-level GPT trained on AMP sequences (2.69M params) | 10 MB | val perplexity 10.13 (best epoch) |
| `esm_finetuned.pt` | ESM-2 35M fine-tuned end-to-end with classification head | 134 MB | acc 0.956, AUC 0.990 |

## Usage

```python
import joblib, torch
from transformers import AutoModel, AutoTokenizer

# AMP classifier (frozen ESM-2 embeddings + logistic regression)
clf = joblib.load("esm_amp.joblib")["clf"]
tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
esm = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
# ... embed with mean pooling, then clf.predict_proba(X)
```

Full pipeline (train, score, generate, evaluate) in the GitHub repo:
`https://github.com/tomdabro/ai-amp-search`

## Data

- **GRAMPA** (Witten & Witten 2019, bioRxiv): 6,642 unique AMP
  sequences with log₁₀ MIC against E. coli.
  [DOI: 10.1101/692681](https://doi.org/10.1101/692681). Source
  databases: APD (Wang 2004,
  [DOI: 10.1093/nar/gkh025](https://doi.org/10.1093/nar/gkh025)),
  DRAMP (Fan et al. 2016,
  [DOI: 10.1038/srep24482](https://doi.org/10.1038/srep24482)),
  DBAASP (Gogoladze et al. 2014,
  [DOI: 10.1111/1574-6968.12489](https://doi.org/10.1111/1574-6968.12489)).
- **ConsAMPHemo** (Xie et al. 2025, Protein Science): 884 peptides,
  hemolytic vs non-hemolytic.
  [DOI: 10.1002/pro.70087](https://doi.org/10.1002/pro.70087).
- **UniProt** (CC-BY-4.0): non-AMP negatives. The UniProt Consortium
  2017, *Nucleic Acids Research* 45(D1):D158-D169.
  [DOI: 10.1093/nar/gkw1099](https://doi.org/10.1093/nar/gkw1099).
- **DBAASP** (Gogoladze et al. 2014, *FEMS Microbiology Letters*
  357(1):63-68): clinical-isolate AMP-MIC validation data.
  [DOI: 10.1111/1574-6968.12489](https://doi.org/10.1111/1574-6968.12489).

## Honest evaluation

The naive random split gives AUC 1.000 — inflated by homology leakage
(near-identical family variants in both train and test). With a
homology-controlled split (k-mer Jaccard >= 0.8 clustering, split by
cluster), the honest AUC is **0.971**. Length-binned AUC: [5,20] 0.959,
[21,50] 0.972 — the model is not just predicting length.

## License

MIT (code). Data is third-party with its own terms.
