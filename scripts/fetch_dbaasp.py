#!/usr/bin/env python3
"""Fetch AMP-MIC data from DBAASP (Database of Antimicrobial Activity
and Structure of Peptides) for a target species.

DBAASP (https://dbaasp.org) is a manually curated database of peptide
antimicrobial activity with measured MICs against specific target
species — including clinical isolates (e.g. "Staphylococcus aureus MR",
"Klebsiella pneumoniae"). This fetcher pulls peptides + their
targetActivities (species, MIC, concentration, unit, medium) into a
clean CSV for training or analysis.

Usage:
    python3 scripts/fetch_dbaasp.py --species "Klebsiella pneumoniae" \
        --out data/dbaasp_kp.csv
    python3 scripts/fetch_dbaasp.py --species "Staphylococcus aureus MR" \
        --out data/dbaasp_mrsa.csv
"""

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://dbaasp.org"


def api_get(path: str, params: dict | None = None) -> dict:
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def fetch_peptides(species: str, limit: int = 500) -> list[dict]:
    """Fetch peptide list for a target species (paginated)."""
    out, offset = [], 0
    while True:
        data = api_get("/peptides", {
            "targetSpecies.value": species,
            "limit": min(100, limit - len(out)),
            "offset": offset,
        })
        batch = data.get("data", [])
        out.extend(batch)
        total = data.get("totalCount", 0)
        offset += len(batch)
        print(f"  fetched {len(out)}/{total} peptides")
        if not batch or len(out) >= total or len(out) >= limit:
            break
        time.sleep(0.3)
    return out


def fetch_detail(peptide_id: int) -> dict:
    time.sleep(0.2)
    return api_get(f"/peptides/{peptide_id}")


def extract_rows(detail: dict) -> list[dict]:
    """Flatten a peptide detail into (sequence, species, MIC) rows."""
    # Sequence: prefer the peptide's own, else first monomer.
    seq = (detail.get("sequence") or "").strip()
    if not seq and detail.get("monomers"):
        seq = detail["monomers"][0].get("sequence") or ""
    seq = seq.upper()
    if not seq:
        return []

    rows = []
    for ta in detail.get("targetActivities") or []:
        species = (ta.get("targetSpecies") or {}).get("name", "")
        conc = ta.get("concentration")
        unit = (ta.get("unit") or {}).get("name", "")
        medium = (ta.get("medium") or {}).get("name", "")
        try:
            mic = float(conc)
        except (TypeError, ValueError):
            continue
        rows.append({
            "peptide_id": detail.get("id"),
            "name": detail.get("name", ""),
            "sequence": seq,
            "target_species": species,
            "mic": mic,
            "unit": unit,
            "medium": medium,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species", required=True,
                    help="target species, e.g. 'Klebsiella pneumoniae'")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", default=500, type=int,
                    help="max peptides to fetch (default 500)")
    args = ap.parse_args()

    print(f"searching DBAASP for peptides active against: {args.species}")
    peptides = fetch_peptides(args.species, args.limit)
    print(f"fetching details for {len(peptides)} peptides...")

    rows = []
    for p in peptides:
        try:
            detail = fetch_detail(p["id"])
        except Exception as e:  # noqa: BLE001 — skip one bad record
            print(f"  skip peptide {p['id']}: {e}")
            continue
        rows.extend(extract_rows(detail))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "peptide_id", "name", "sequence", "target_species",
            "mic", "unit", "medium"])
        w.writeheader()
        w.writerows(rows)

    n_seq = len({r["sequence"] for r in rows})
    print(f"done -> {args.out} ({len(rows)} activity rows, "
          f"{n_seq} unique sequences)")


if __name__ == "__main__":
    main()
