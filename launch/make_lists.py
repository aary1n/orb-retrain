"""Write the row lists used by training and evaluation. Runs in this session's container.

Inputs: data/meta/{train,val,test}.csv.gz (from the Modal job data_metadata) and
removal/test_overlap_parent_ids.txt. Output: lists/*.gz plus lists/MANIFEST.json
with n and sha256 for each. Upload with:  modal volume put -f orb-retrain lists lists

Training rows are identified by sid (unique in AIMD train: 551,578 rows, 551,578
sids). Eval rows are identified by split/subset/key, because sids repeat in val
(500 pairs) and test (250 groups); every official row is kept.
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "meta"
OUT = ROOT / "lists"


def write(name, lines):
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    with gzip.open(p, "wt") as f:
        for s in lines:
            f.write(s + "\n")
    h = hashlib.sha256()
    for s in lines:
        h.update((s + "\n").encode())
    return {"n": len(lines), "sha256_of_lines": h.hexdigest(), "sha256_of_gz_file": hashlib.sha256(p.read_bytes()).hexdigest()}


def rows(d):
    d = d.assign(k=d["key"].astype(int)).sort_values(["split", "subset", "k"])
    return [f"{a}\t{b}\t{c}\t{e}\t{n}" for a, b, c, e, n in zip(d.split, d.subset, d.key, d.sid, d.natoms)]


def main():
    tr = pd.read_csv(META / "train.csv.gz", dtype=str, keep_default_na=False, usecols=["split", "subset", "key", "sid", "natoms"])
    va = pd.read_csv(META / "val.csv.gz", dtype=str, keep_default_na=False, usecols=["split", "subset", "key", "sid", "natoms"])
    te = pd.read_csv(META / "test.csv.gz", dtype=str, keep_default_na=False, usecols=["split", "subset", "key", "sid", "natoms", "parent_id"])
    overlap = set((ROOT / "removal" / "test_overlap_parent_ids.txt").read_text().split())
    te_clean = te[~te.parent_id.isin(overlap)]
    aimd = tr[tr.subset.str.startswith("aimd")]
    assert aimd.sid.nunique() == len(aimd) == 551_578
    m = {}
    m["full_train_sids.txt.gz"] = write("full_train_sids.txt.gz", sorted(aimd.sid))
    m["val_all_rows.tsv.gz"] = write("val_all_rows.tsv.gz", rows(va))
    m["test_clean_rows.tsv.gz"] = write("test_clean_rows.tsv.gz", rows(te_clean))
    rng = np.random.default_rng(0)
    m["val_subset5000_rows.tsv.gz"] = write("val_subset5000_rows.tsv.gz", rows(va.iloc[np.sort(rng.choice(len(va), 5000, replace=False))]))
    # small sets for the smoke run only
    rng = np.random.default_rng(1)
    m["smoke_val_rows.tsv.gz"] = write("smoke_val_rows.tsv.gz", rows(va.iloc[np.sort(rng.choice(len(va), 2000, replace=False))]))
    m["smoke_test_rows.tsv.gz"] = write("smoke_test_rows.tsv.gz", rows(te_clean.iloc[np.sort(rng.choice(len(te_clean), 1000, replace=False))]))
    m["smoke_val_subset_rows.tsv.gz"] = write("smoke_val_subset_rows.tsv.gz", rows(va.iloc[np.sort(rng.choice(len(va), 500, replace=False))]))
    m["_notes"] = {"test_clean": f"test minus {len(overlap)} starting crystals also in train: {len(te)} -> {len(te_clean)} rows",
                   "val_duplicate_sid_rows": int(va.sid.duplicated(keep=False).sum()),
                   "test_clean_duplicate_sid_rows": int(te_clean.sid.duplicated(keep=False).sum()),
                   "row format": "split<TAB>subset<TAB>key<TAB>sid<TAB>natoms, sorted by split, subset, int(key)"}
    (OUT / "MANIFEST.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    sys.exit(main())
