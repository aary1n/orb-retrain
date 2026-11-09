"""Phase A: time the three models of brief section 2 on one GPU type.

Runs inside Modal (Aaryan's credits). One container, the three models one after
another, 200 timed steps each (after 10 warm-up steps), batch 100, the
finetune.py v0.5.5 recipe: Adam + OneCycle via utils.get_optim(3e-4, ...),
utils.gradient_clipping(model, 0.5), random rotation augmentation,
precision "float32-high", loss = model.loss(batch).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import time
from pathlib import Path

MODELS = [
    ("conservative_full", "orb_v3_conservative_architecture", dict(latent_dim=256, base_mlp_hidden_dim=1024)),
    ("conservative_narrow", "orb_v3_conservative_architecture", dict(latent_dim=128, base_mlp_hidden_dim=512)),
    ("direct_full", "orb_v3_direct_architecture", dict(latent_dim=256, base_mlp_hidden_dim=1024)),
]


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def aimd_train_files(vol_root: str) -> list[str]:
    fs = sorted(glob.glob(f"{vol_root}/omat24_1M/**/train/aimd-from-PBE-*/train.aselmdb", recursive=True))
    assert len(fs) == 4, fs
    return fs


def copy_local(files: list[str], dst_dir: str = "/tmp/data") -> tuple[list[str], float, int]:
    os.makedirs(dst_dir, exist_ok=True)
    t = time.time()
    out, nbytes = [], 0
    for f in files:
        d = os.path.join(dst_dir, Path(f).parent.name + ".aselmdb")
        shutil.copyfile(f, d)
        nbytes += os.path.getsize(d)
        out.append(d)
    return out, time.time() - t, nbytes


def build_loader(paths, index, batch_size, num_workers, seed, sc):
    import torch
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler

    from dataset_lmdb import AseLmdbDataset
    from orb_models import utils
    from orb_models.dataset import augmentations
    from orb_models.forcefield import base, property_definitions

    targets = property_definitions.instantiate_property_config({"graph": ["energy", "stress"], "node": ["forces"]})
    ds = AseLmdbDataset("omat", paths, index, system_config=sc, target_config=targets,
                        augmentations=[augmentations.rotate_randomly])
    g = torch.Generator()
    g.manual_seed(seed)
    bs = BatchSampler(RandomSampler(ds, generator=g), batch_size=batch_size, drop_last=False)
    kw = dict(persistent_workers=True, prefetch_factor=4, timeout=600) if num_workers > 0 else {}
    return ds, DataLoader(ds, num_workers=num_workers, worker_init_fn=utils.worker_init_fn,
                          collate_fn=base.batch_graphs, batch_sampler=bs, **kw)


def run_timing(vol_root, gpu, tag, commit, steps=200, warmup=10, batch_size=100, num_workers=7,
               deadline_s=810, t0=None, files=None, device=None):
    import numpy as np
    import torch

    from dataset_lmdb import decimal_keys
    from orb_models import utils
    from orb_models.forcefield import base, pretrained
    from orb_models.forcefield.atomic_system import SystemConfig

    t0 = t0 or time.time()
    left = lambda: deadline_s - (time.time() - t0)  # noqa: E731
    res = {"gpu_requested": gpu, "tag": tag, "steps": steps, "warmup": warmup, "batch_size": batch_size,
           "num_workers": num_workers, "torch": torch.__version__, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        res["gpu_name"] = torch.cuda.get_device_name(0)
        res["gpu_mem_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30

    src = files or aimd_train_files(vol_root)
    if files is None:
        paths, copy_s, nbytes = copy_local(src)
        res["copy_to_local_disk"] = {"seconds": copy_s, "bytes": nbytes, "files": src}
        _log(f"copied {nbytes/1e9:.2f} GB to local disk in {copy_s:.1f} s")
    else:
        paths = src
    t = time.time()
    index = [(fi, k) for fi, p in enumerate(paths) for k in decimal_keys(p)]
    res["n_structures_indexed"] = len(index)
    res["index_s"] = time.time() - t
    _log(f"indexed {len(index)} structures in {res['index_s']:.1f} s")

    utils.set_torch_precision("float32-high")
    sc = SystemConfig(radius=6.0, max_num_neighbors=120)
    ds, loader = build_loader(paths, index, batch_size, num_workers, seed=0, sc=sc)

    # loader-only throughput (also warms the workers)
    it = iter(loader)
    t = time.time()
    nb, ns = 0, 0
    while nb < 60:
        b = next(it)
        nb += 1
        ns += len(b.n_node)
    res["loader_only"] = {"batches": nb, "structures": ns, "seconds": time.time() - t,
                          "structures_per_s": ns / (time.time() - t)}
    _log("loader only:", res["loader_only"])

    res["models"] = {}
    for name, fn, kw in MODELS:
        if left() < 120:
            res["models"][name] = {"skipped": "deadline"}
            continue
        try:
            res["models"][name] = _one_model(name, fn, kw, sc, device, it, ds, steps, warmup, left)
        except Exception as exc:  # noqa: BLE001  (e.g. CUDA out of memory) - record and go on
            import traceback
            res["models"][name] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
            _log(name, "ERROR", repr(exc))
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _partial_save(vol_root, tag, res, commit, files)

    # GPU-only throughput of the full conservative model by batch size (fresh model, preloaded batches)
    if device.type == "cuda":
        res["conservative_full_by_batch"] = _sweep(sc, device, ds, left)
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    res["elapsed_s"] = time.time() - t0
    _partial_save(vol_root, tag, res, commit, files)
    return json.loads(json.dumps(res, default=float))


def _sweep(sc, device, ds, left, sizes=(25, 50, 100, 150, 200)):
    import numpy as np
    import torch

    from orb_models import utils
    from orb_models.forcefield import base, pretrained

    out = {}
    utils.seed_everything(0)
    model = pretrained.orb_v3_conservative_architecture(latent_dim=256, base_mlp_hidden_dim=1024,
                                                        system_config=sc, device=device)
    model.train()
    opt, _ = utils.get_optim(3e-4, 10_000, model)
    rng = np.random.default_rng(1)
    for bsz in sizes:
        if left() < 100:
            out[bsz] = {"skipped": "deadline"}
            continue
        try:
            out[bsz] = _by_batch(model, opt, ds, rng, bsz, device, base)
        except Exception as exc:  # noqa: BLE001  (CUDA OOM is expected at some size)
            out[bsz] = {"error": repr(exc)[:300]}
            torch.cuda.empty_cache()
            _log("sweep", bsz, out[bsz])
            break
        _log("sweep", bsz, out[bsz])
        torch.cuda.empty_cache()
    return out


def _partial_save(vol_root, tag, res, commit, files):
    if vol_root and files is None:
        os.makedirs(f"{vol_root}/timing", exist_ok=True)
        with open(f"{vol_root}/timing/{tag}.json", "w") as f:
            json.dump(res, f, indent=1, default=float)
        commit()


def _one_model(name, fn, kw, sc, device, it, ds, steps, warmup, left):
    import numpy as np
    import torch

    from orb_models import utils
    from orb_models.forcefield import base, pretrained

    utils.seed_everything(0)
    model = getattr(pretrained, fn)(system_config=sc, device=device, **kw)
    model.train()
    nparam = sum(p.numel() for p in model.parameters())
    ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt, sched = utils.get_optim(3e-4, warmup + steps, model)
    hooks = utils.gradient_clipping(model, 0.5)
    rec = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    nan = False
    for _ in range(warmup + steps):
        if left() < 60:
            break
        tw = time.perf_counter()
        batch = next(it)
        tg = time.perf_counter()
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        out = model.loss(batch)
        loss = out.loss
        if torch.isnan(loss):  # same check as finetune.py; forces a sync each step
            nan = True
        loss.backward()
        opt.step()
        sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        te = time.perf_counter()
        rec.append((tg - tw, te - tg, int(len(batch.n_node)), int(batch.n_node.sum()), int(batch.n_edge.sum()),
                    float(loss.detach())))
    for h in hooks:
        h.remove()
    timed = rec[warmup:]
    a = np.array([r[:5] for r in timed], dtype=float) if timed else np.zeros((0, 5))
    tot = a[:, 0].sum() + a[:, 1].sum() if len(a) else float("nan")
    m = {
        "params_all": nparam,
        "params_trainable": ntrain,
        "timed_steps": len(timed),
        "wall_s": tot,
        "s_per_step": tot / max(len(timed), 1),
        "data_wait_s_per_step": a[:, 0].mean() if len(a) else None,
        "compute_s_per_step": a[:, 1].mean() if len(a) else None,
        "structures": a[:, 2].sum(),
        "atoms": a[:, 3].sum(),
        "edges": a[:, 4].sum(),
        "structures_per_s": a[:, 2].sum() / tot if len(a) else None,
        "atoms_per_structure": a[:, 3].sum() / max(a[:, 2].sum(), 1),
        "edges_per_structure": a[:, 4].sum() / max(a[:, 2].sum(), 1),
        "peak_gpu_mem_gib": (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else None,
        "loss_first": rec[0][5] if rec else None,
        "loss_last": rec[-1][5] if rec else None,
        "nan_seen": nan,
    }
    _log(name, json.dumps(m, default=float))

    # validation-style evaluation speed (forces need autograd for the conservative model)
    if left() > 90:
        model.eval()
        te0 = time.time()
        ne = 0
        for _ in range(20):
            b = next(it).to(device)
            if "conservative" in name:
                model.predict(b)
            else:
                with torch.no_grad():
                    model.predict(b)
            ne += len(b.n_node)
        if device.type == "cuda":
            torch.cuda.synchronize()
        m["eval_structures_per_s"] = ne / (time.time() - te0)
        model.train()

    return m


def _by_batch(model, opt, ds, rng, bsz, device, base):
    """30 steps on 3 preloaded batches of bsz structures: GPU time only, no data loading."""
    import torch

    torch.cuda.reset_peak_memory_stats()
    bl = [base.batch_graphs([ds[int(j)] for j in rng.integers(0, len(ds), bsz)]).to(device) for _ in range(3)]
    for k in range(3):  # warm-up
        opt.zero_grad(set_to_none=True)
        model.loss(bl[k % 3]).loss.backward()
        opt.step()
    torch.cuda.synchronize()
    tt = time.perf_counter()
    nst = 0
    for k in range(30):
        opt.zero_grad(set_to_none=True)
        model.loss(bl[k % 3]).loss.backward()
        opt.step()
        nst += int(len(bl[k % 3].n_node))
    torch.cuda.synchronize()
    dt = time.perf_counter() - tt
    return {"structures_per_s": nst / dt, "s_per_step": dt / 30, "atoms_per_step": sum(int(b.n_node.sum()) for b in bl) / 3,
            "edges_per_step": sum(int(b.n_edge.sum()) for b in bl) / 3,
            "peak_gpu_mem_gib": torch.cuda.max_memory_allocated() / 2**30}
