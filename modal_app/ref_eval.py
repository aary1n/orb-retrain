"""Reference evaluation: the released orb-v3-conservative-inf-omat checkpoint on the full val and clean test sets.

Runs inside Modal (Aaryan's credits) on one H100, once. It writes its own files under
runs/<run_id>/ (never val_metrics.csv, which the training container is appending to):
  reference_metrics.csv, reference_model.json, eval/reference_orbv3_inf_omat_{val,test_clean}.csv.gz
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import time
from pathlib import Path

WEIGHTS_URL = ("https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/"
               "orb-v3-conservative-inf-omat-20250404.ckpt")
REF_NAME = "orbv3_inf_omat"


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run_reference(vol_root: str, run_id: str, commit, local_dir: str = "/tmp/data", limit: int | None = None,
                  device=None) -> dict:
    import torch
    from cached_path import cached_path

    import train_job
    from dataset_lmdb import AseLmdbDataset
    from orb_models import utils
    from orb_models.forcefield import pretrained, property_definitions

    rd = Path(vol_root) / "runs" / run_id
    cfg = json.loads((rd / "config.json").read_text())
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()

    # data files to local disk (val and test only)
    os.makedirs(local_dir, exist_ok=True)
    local = {}
    for split, subset in cfg["files_needed"]:
        if split == "train":
            continue
        src = f"{vol_root}/omat24_1M/{split}/{subset}/{split}.aselmdb"
        dst = f"{local_dir}/{split}__{subset}.aselmdb"
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        local[(split, subset)] = dst
    copy_s = time.time() - t0
    _log(f"copied {len(local)} files in {copy_s:.1f} s")

    # the released checkpoint
    t1 = time.time()
    ck = Path(cached_path(WEIGHTS_URL))
    ck_sha = train_job.sha256_file(ck)
    utils.set_torch_precision("float32-high")
    model = pretrained.orb_v3_conservative_inf_omat(device=device, precision="float32-high", compile=False)
    sc = model.system_config
    info = {"weights_url": WEIGHTS_URL, "weights_sha256": ck_sha, "weights_bytes": ck.stat().st_size,
            "orb_models": __import__("orb_models").__version__ if hasattr(__import__("orb_models"), "__version__") else "0.5.5",
            "system_config": {"radius": sc.radius, "max_num_neighbors": sc.max_num_neighbors},
            "params": sum(p.numel() for p in model.parameters()), "load_s": time.time() - t1,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "compile": False}
    _log("loaded reference model", info)
    assert (sc.radius, sc.max_num_neighbors) == tuple(cfg["system_config"].values()), (sc, cfg["system_config"])

    targets = property_definitions.instantiate_property_config({"graph": ["energy", "stress"], "node": ["forces"]})
    eval_files = sorted(local)
    efi = {k: i for i, k in enumerate(eval_files)}
    results = {}
    (rd / "eval").mkdir(parents=True, exist_ok=True)
    rows_out = []
    for name, rel in cfg["eval_sets"].items():
        rows = train_job.read_rows(vol_root, rel)
        if limit:
            rows = rows[:limit]
        eds = AseLmdbDataset("omat", [local[k] for k in eval_files], [(efi[(r[0], r[1])], r[2].encode()) for r in rows],
                             system_config=sc, target_config=targets, augmentations=[])
        res, per = train_job.evaluate_rows(model, eds, rows, device, True, cfg, per_structure=True)
        results[name] = res
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["split", "subset", "key", "sid", "natoms", "energy_err_eV", "forces_mae_eV_per_A",
                    "forces_rmse_eV_per_A", "stress_mae_eV_per_A3"])
        w.writerows(per)
        pp = rd / "eval" / f"reference_{REF_NAME}_{name}.csv.gz"
        with gzip.open(str(pp) + ".tmp", "wt") as g:
            g.write(buf.getvalue())
        os.replace(str(pp) + ".tmp", pp)
        for grp in ("all", "aimd", "rattled"):
            if res[grp]:
                rows_out.append({"step": -1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "set": f"ref_{REF_NAME}_{name}",
                                 "group": grp, **res[grp], "eval_seconds": res["seconds"]})
        _log(f"reference {name}: {res['all']} ({res['seconds']:.0f} s)")
    with open(rd / "reference_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    info.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"), elapsed_s=time.time() - t0, copy_s=copy_s,
                results=results, limit=limit)
    (rd / "reference_model.json").write_text(json.dumps(info, indent=1))
    commit()
    return json.loads(json.dumps(info, default=float))
