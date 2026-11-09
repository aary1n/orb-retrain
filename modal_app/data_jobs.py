"""CPU data jobs. They run inside Modal (Aaryan's credits), CPU only.

download_and_unpack: download OMat24-1M, check sha256, unpack to the data
Volume with a per-file sha256 manifest. Skipped if its done-marker exists and
records the expected sha256.

extract_metadata: read every .aselmdb row once and write one CSV.gz per split
with the fields needed by the removal rules, plus small sample LMDBs for
local tests.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import tarfile
import time
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/251210/omat24_1M_251210.tar.gz"
TAR_BYTES = 2_274_568_000
TAR_SHA256 = "789ecc1306575a1652806b339265d7659c57282eeb1b3e3b54ba0e1154938b81"
NVT_TRAIN_SHA256 = "acac5dae2b147f1193a0cfe0e32510272ade131c48ce7b9e45103babbaaa7bae"
NVT_TRAIN_COUNT = 201_512


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def download_and_unpack(vol_root: str, commit) -> dict:
    root = Path(vol_root)
    out_dir = root / "omat24_1M"
    marker = root / "markers" / "DONE_unpack.json"
    if marker.exists():
        m = json.loads(marker.read_text())
        if m.get("tar_sha256") == TAR_SHA256:
            _log("skip: unpack marker exists with the expected sha256")
            return {"skipped": True, **{k: m[k] for k in ("tar_sha256", "n_files", "nvt_train_sha256")}}

    # 1. download to the container's local disk, hashing as we go
    local = Path("/tmp/omat24_1M_251210.tar.gz")
    h = hashlib.sha256()
    t0 = time.time()
    n = 0
    with urllib.request.urlopen(URL, timeout=120) as r, open(local, "wb") as f:
        while True:
            b = r.read(16 << 20)
            if not b:
                break
            f.write(b)
            h.update(b)
            n += len(b)
    dl_s = time.time() - t0
    sha = h.hexdigest()
    _log(f"downloaded {n} bytes in {dl_s:.1f} s, sha256 {sha}")
    if n != TAR_BYTES or sha != TAR_SHA256:
        raise RuntimeError(f"tarball mismatch: bytes {n} (want {TAR_BYTES}), sha256 {sha} (want {TAR_SHA256})")

    # 2. unpack member by member onto the Volume, refusing unsafe paths
    t1 = time.time()
    manifest = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local, "r:gz") as tf:
        for m in tf:
            name = m.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"unsafe tar member {name!r}")
            if m.isdir():
                continue
            if not m.isfile():
                manifest.append({"name": name, "type": "skipped-nonfile"})
                continue
            dest = out_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            fh = hashlib.sha256()
            src = tf.extractfile(m)
            with open(dest, "wb") as g:
                while True:
                    b = src.read(16 << 20)
                    if not b:
                        break
                    g.write(b)
                    fh.update(b)
            manifest.append({"name": name, "bytes": m.size, "sha256": fh.hexdigest()})
            _log("unpacked", name, m.size)
    unpack_s = time.time() - t1

    nvt = [e for e in manifest if e["name"].endswith("train/aimd-from-PBE-1000-nvt/train.aselmdb")]
    nvt_sha = nvt[0]["sha256"] if nvt else None
    res = {
        "url": URL,
        "tar_bytes": n,
        "tar_sha256": sha,
        "download_s": dl_s,
        "unpack_s": unpack_s,
        "n_files": len(manifest),
        "nvt_train_sha256": nvt_sha,
        "nvt_train_sha256_ok": nvt_sha == NVT_TRAIN_SHA256,
        "manifest": manifest,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (root / "markers").mkdir(parents=True, exist_ok=True)
    (root / "markers" / "unpack_manifest.json").write_text(json.dumps(res, indent=1))
    if not res["nvt_train_sha256_ok"]:
        commit()
        raise RuntimeError(f"nvt train.aselmdb sha256 {nvt_sha} != {NVT_TRAIN_SHA256}")
    marker.write_text(json.dumps({k: v for k, v in res.items() if k != "manifest"}, indent=1))
    commit()
    local.unlink()
    return {k: v for k, v in res.items() if k != "manifest"}


# ---------------------------------------------------------------------------
META_FIELDS = [
    "split", "subset", "key", "sid", "parent_id", "composition_reduced", "prototype_label",
    "parent_prototype_label", "task_type", "natoms", "elements_from_numbers",
]


def _one_file(args):
    import zlib
    from collections import Counter

    import lmdb
    import numpy as np
    from ase.data import chemical_symbols
    from ase.io.jsonio import decode

    path, split, subset = args
    env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=True)
    rows = []
    data_keys = None
    other_keys = []
    first = None
    with env.begin() as txn:
        for k, v in txn.cursor():
            if not k.isdigit():
                other_keys.append(k.decode(errors="replace"))
                continue
            d = decode(zlib.decompress(v).decode())
            dd = d.get("data", {})
            if data_keys is None:
                data_keys = sorted(dd.keys())
                first = {kk: (str(dd[kk])[:200]) for kk in data_keys}
                first["_row_keys"] = sorted(d.keys())
            nums = np.asarray(d["numbers"])
            cnt = Counter(chemical_symbols[z] for z in nums.tolist())
            rows.append([
                split, subset, k.decode(), dd.get("sid"), dd.get("parent_id"),
                dd.get("composition_reduced") if isinstance(dd.get("composition_reduced"), str) else json.dumps(dd.get("composition_reduced")),
                dd.get("prototype_label"), dd.get("parent_prototype_label"), dd.get("task_type"), len(nums),
                ",".join(f"{s}:{c}" for s, c in sorted(cnt.items())),
            ])
    env.close()
    npz = {}
    mp = Path(path).parent / "metadata.npz"
    if mp.exists():
        z = np.load(mp, allow_pickle=False)
        npz = {kk: [list(z[kk].shape), str(z[kk].dtype)] for kk in z.files}
    return {"path": path, "split": split, "subset": subset, "n": len(rows), "rows": rows,
            "data_keys": data_keys, "first_data": first, "other_keys": other_keys, "metadata_npz": npz}


def extract_metadata(vol_root: str, commit, n_proc: int = 8) -> dict:
    from multiprocessing import Pool

    import lmdb

    root = Path(vol_root)
    marker = root / "markers" / "DONE_meta.json"
    if marker.exists():
        _log("skip: metadata marker exists")
        return json.loads(marker.read_text())
    base = root / "omat24_1M"
    files = sorted(base.rglob("*.aselmdb"))
    jobs = []
    for p in files:
        rel = p.relative_to(base).parts  # [..., split, subset, file]
        split, subset = rel[-3], rel[-2]
        jobs.append((str(p), split, subset))
    jobs.sort(key=lambda j: -os.path.getsize(j[0]))  # biggest first
    t0 = time.time()
    results = []
    with Pool(n_proc) as pool:
        for r in pool.imap_unordered(_one_file, jobs):
            _log("read", r["split"], r["subset"], r["n"])
            results.append(r)
    read_s = time.time() - t0

    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_files = {}
    for split in sorted({r["split"] for r in results}):
        p = meta_dir / f"{split}.csv.gz"
        with gzip.open(p, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(META_FIELDS)
            for r in sorted([r for r in results if r["split"] == split], key=lambda r: r["subset"]):
                w.writerows(r["rows"])
        out_files[split] = {"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    # small sample LMDBs for local tests (raw values copied unchanged)
    samp = root / "samples"
    samp.mkdir(parents=True, exist_ok=True)
    for name, pick in [("train_aimd_sample", lambda j: j[1] == "train" and j[2].startswith("aimd")),
                       ("val_sample", lambda j: j[1] == "val" and j[2] in ("aimd-from-PBE-1000-nvt", "rattled-500"))]:
        dst = samp / f"{name}.aselmdb"
        if dst.exists():
            dst.unlink()
        out = lmdb.open(str(dst), subdir=False, map_size=2 << 30, lock=False)
        n = 0
        with out.begin(write=True) as wt:
            for j in sorted(filter(pick, jobs)):
                env = lmdb.open(j[0], subdir=False, readonly=True, lock=False)
                with env.begin() as txn:
                    c = 0
                    for k, v in txn.cursor():
                        if not k.isdigit():
                            continue
                        wt.put(str(n).encode(), v)
                        n += 1
                        c += 1
                        if c >= 500:
                            break
                env.close()
        out.sync()
        out.close()
        lock = Path(str(dst) + "-lock")
        if lock.exists():
            lock.unlink()

    summary = {
        "read_s": read_s,
        "files": [{k: r[k] for k in ("path", "split", "subset", "n", "other_keys", "metadata_npz")} for r in results],
        "counts_by_split": {s: sum(r["n"] for r in results if r["split"] == s) for s in {r["split"] for r in results}},
        "aimd_train": sum(r["n"] for r in results if r["split"] == "train" and r["subset"].startswith("aimd")),
        "data_keys": results[0]["data_keys"],
        "first_data_example": results[0]["first_data"],
        "meta_files": out_files,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (meta_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    marker.write_text(json.dumps(summary, indent=1))
    commit()
    return summary
