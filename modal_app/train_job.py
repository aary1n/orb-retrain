"""One training chunk. Runs inside Modal (Aaryan's credits) on one H100.

Recipe (orb-models v0.5.5 finetune.py): Adam + OneCycle via
utils.get_optim(lr, total_steps, model) (peak 10x lr, pct_start 0.05),
utils.gradient_clipping(model, 0.5) hooks, random rotation augmentation,
precision "float32-high", loss = model.loss(batch), batch 100. Targets are
normalised inside orb-models exactly as in its own training (PLAN.md 2).

What a chunk does:
- copies the training and evaluation files from the Volume to local disk;
- on the very first attempt of a run, saves the step-0 state_dict (before any
  optimiser step) and writes its sha256 to run_record.json as the start
  fingerprint;
- runs any checkpoint evaluation that is still missing (full val + clean test);
- trains, logging loss and orb-models' raw energy/force/stress errors every
  log_every steps; evaluates a fixed val subset every val_every steps;
- at 25/50/75/100 % of steps saves a state_dict checkpoint and runs the full
  val + clean-test evaluation, with per-structure errors;
- skips a batch that runs out of GPU memory, logs it, and writes STOP at
  max_oom_skips skips;
- stops at cfg["stop_at_step"] if set (state "paused"), or at the 6 h wall
  clock (state "chunk_end"), or when all steps are done (state "done").
"""

from __future__ import annotations

import csv
import faulthandler
import gzip
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

GPA_PER_EV_A3 = 160.21766208
# (orb-models' metric name in model.loss(batch).log, column in train_log.csv). Raw = physical units:
# energy_mae_raw is |E_pred - E_true| per structure (total energy), forces per component, stress per Voigt component.
TRAIN_COLS = [
    ("loss", "loss"), ("energy_loss", "energy_loss"), ("grad-forces_loss", "forces_loss"),
    ("grad-stress_loss", "stress_loss"), ("confidence_loss", "confidence_loss"),
    ("energy_mae_raw", "energy_mae_raw_eV_per_structure"), ("grad-forces_mae_raw", "forces_mae_raw_eV_per_A"),
    ("grad-stress_mae_raw", "stress_mae_raw_eV_per_A3"),
]


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tensor_fingerprint(sd) -> str:
    """sha256 over (name, dtype, shape, bytes) of every tensor, sorted by name. Device independent."""
    import numpy as np

    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(np.ascontiguousarray(t.numpy()).tobytes())
    return h.hexdigest()


def read_rows(vol_root: str, rel: str):
    """Row list file: lines 'split<TAB>subset<TAB>key<TAB>sid<TAB>natoms'."""
    with gzip.open(f"{vol_root}/{rel}", "rt") as f:
        return [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]


class ResumableBatches:
    """Endless batches of indices: epoch e is permutation(seed, e); state = (epoch, batch)."""

    def __init__(self, n: int, batch_size: int, seed: int, epoch: int = 0, batch: int = 0):
        self.n, self.bs, self.seed = n, batch_size, seed
        self.epoch, self.batch = epoch, batch

    def __iter__(self):
        import numpy as np

        e, b = self.epoch, self.batch
        while True:
            perm = np.random.default_rng([self.seed, e]).permutation(self.n)
            nb = math.ceil(self.n / self.bs)
            for j in range(b, nb):
                yield perm[j * self.bs:(j + 1) * self.bs].tolist()
            e, b = e + 1, 0

    @staticmethod
    def advance(state, n, bs):
        e, b = state
        b += 1
        if b >= math.ceil(n / bs):
            e, b = e + 1, 0
        return e, b


def atom_budget_batches(natoms, max_structs: int, max_atoms: int):
    """Consecutive batches of at most max_structs structures and max_atoms atoms (a bigger one goes alone)."""
    out, cur, ca = [], [], 0
    for i, n in enumerate(natoms):
        if cur and (len(cur) >= max_structs or ca + n > max_atoms):
            out.append(cur)
            cur, ca = [], 0
        cur.append(i)
        ca += n
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------------------- evaluation
def evaluate_rows(model, ds, rows, device, conservative, cfg, rearm=None, per_structure=False):
    """Energy/force/stress errors on `rows` (ds index i <-> rows[i]). Returns (summaries, per-structure rows)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from orb_models.forcefield import base

    nat = [int(r[4]) for r in rows]
    batches = atom_budget_batches(nat, cfg["eval_max_structs"], cfg["eval_max_atoms"])
    loader = DataLoader(ds, batch_sampler=batches, num_workers=cfg["eval_workers"], collate_fn=base.batch_graphs)
    fk, sk = ("grad_forces", "grad_stress") if conservative else ("forces", "stress")
    n = len(rows)
    e_err = np.zeros(n)
    f_abs = np.zeros(n)
    f_sq = np.zeros(n)
    s_abs = np.zeros(n)
    t0 = time.time()
    model.eval()
    for idx, b in zip(batches, loader):
        b = b.to(device)
        if conservative:
            p = model.predict(b)
        else:
            with torch.no_grad():
                p = model.predict(b)
        nb = len(b.n_node)
        seg = torch.repeat_interleave(torch.arange(nb, device=device), b.n_node)
        de = (p["energy"].detach().reshape(-1) - b.system_targets["energy"].reshape(-1))
        df = (p[fk].detach() - b.node_targets["forces"])
        fa = torch.zeros(nb, device=device, dtype=df.dtype).index_add_(0, seg, df.abs().sum(dim=1))
        fs = torch.zeros(nb, device=device, dtype=df.dtype).index_add_(0, seg, (df ** 2).sum(dim=1))
        ds_ = (p[sk].detach() - b.system_targets["stress"]).abs().mean(dim=1)
        e_err[idx] = de.cpu().numpy()
        f_abs[idx] = fa.cpu().numpy()
        f_sq[idx] = fs.cpu().numpy()
        s_abs[idx] = ds_.cpu().numpy()
        if rearm:
            rearm()
    model.train()
    secs = time.time() - t0
    natoms = np.array(nat, dtype=float)
    aimd = np.array([r[1].startswith("aimd") for r in rows])

    def summ(mask):
        m = int(mask.sum())
        if m == 0:
            return None
        nc = 3 * natoms[mask].sum()
        return {"n": m,
                "energy_mae_meV_per_atom": 1000 * float(np.mean(np.abs(e_err[mask]) / natoms[mask])),
                "forces_mae_meV_per_A": 1000 * float(f_abs[mask].sum() / nc),
                "forces_rmse_meV_per_A": 1000 * float(math.sqrt(f_sq[mask].sum() / nc)),
                "stress_mae_GPa": GPA_PER_EV_A3 * float(np.mean(s_abs[mask]))}

    out = {"all": summ(np.ones(n, bool)), "aimd": summ(aimd), "rattled": summ(~aimd), "seconds": secs}
    per = None
    if per_structure:
        per = [(r[0], r[1], r[2], r[3], int(r[4]), float(e_err[i]), float(f_abs[i] / (3 * nat[i])),
                float(math.sqrt(f_sq[i] / (3 * nat[i]))), float(s_abs[i])) for i, r in enumerate(rows)]
    return out, per


# --------------------------------------------------------------------------- main
def run_chunk(vol_root: str, run_id: str, chunk: int, t0: float, commit) -> dict:
    import numpy as np
    import torch

    from dataset_lmdb import AseLmdbDataset
    from orb_models import utils
    from orb_models.dataset import augmentations
    from orb_models.forcefield import base, pretrained, property_definitions
    from orb_models.forcefield.atomic_system import SystemConfig
    from torch.utils.data import DataLoader

    rd = Path(vol_root) / "runs" / run_id
    cfg = json.loads((rd / "config.json").read_text())
    wall = cfg["chunk_wall_s"]
    left = lambda: wall - (time.time() - t0)  # noqa: E731
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conservative = "conservative" in cfg["arch_fn"]

    def rearm():
        faulthandler.dump_traceback_later(cfg.get("watchdog_s", 900), exit=True)

    rearm()

    def append_csv(name, row):
        p = rd / name
        new = not p.exists()
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    rec_p = rd / "run_record.json"
    rec = json.loads(rec_p.read_text()) if rec_p.exists() else {"run_id": run_id, "chunks": []}

    def save_rec():
        tmp = rec_p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, rec_p)

    # ---- environment record for this chunk ----
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    code = {}
    for mname in ("common", "dataset_lmdb", "train_job", "plan_constants", "train_app"):
        m = sys.modules.get(mname)
        if m is not None and getattr(m, "__file__", None):
            code[mname] = sha256_file(m.__file__)
    ch = {"chunk": chunk, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
          "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
          "torch": str(torch.__version__), "code_sha256": code,
          "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest()}
    if not (rd / "pip_freeze.txt").exists():
        (rd / "pip_freeze.txt").write_text(freeze)

    # ---- copy data to local disk ----
    tc = time.time()
    local = {}
    ldir = cfg.get("local_data_dir", "/tmp/data")
    os.makedirs(ldir, exist_ok=True)
    nbytes = 0
    for split, subset in cfg["files_needed"]:
        src = f"{vol_root}/omat24_1M/{split}/{subset}/{split}.aselmdb"
        dst = f"{ldir}/{split}__{subset}.aselmdb"
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        nbytes += os.path.getsize(dst)
        local[(split, subset)] = dst
        rearm()
    ch["copy_s"], ch["copy_bytes"] = time.time() - tc, nbytes
    _log(f"copied {nbytes/1e9:.2f} GB to local disk in {ch['copy_s']:.1f} s")

    # ---- training rows ----
    train_sids = set(gzip.open(f"{vol_root}/{cfg['train_sids']}", "rt").read().split())
    paths, fidx, index = [], {}, []
    with gzip.open(f"{vol_root}/meta/train.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["subset"].startswith("aimd") and r["sid"] in train_sids:
                k = ("train", r["subset"])
                if k not in fidx:
                    fidx[k] = len(paths)
                    paths.append(local[k])
                index.append((fidx[k], r["key"].encode()))
    if len(index) != cfg["n_train"]:
        raise RuntimeError(f"training rows {len(index)} != n_train {cfg['n_train']}")

    utils.set_torch_precision("float32-high")
    sc = SystemConfig(**cfg["system_config"])
    targets = property_definitions.instantiate_property_config({"graph": ["energy", "stress"], "node": ["forces"]})
    ds = AseLmdbDataset("omat", paths, index, system_config=sc, target_config=targets,
                        augmentations=[augmentations.rotate_randomly])

    # ---- evaluation sets (rows identified by split/subset/key; sids are not unique in val/test) ----
    eval_files = sorted(local)
    efi = {k: i for i, k in enumerate(eval_files)}
    eval_sets = {}
    for name, rel in cfg["eval_sets"].items():
        rows = read_rows(vol_root, rel)
        eds = AseLmdbDataset("omat", [local[k] for k in eval_files], [(efi[(r[0], r[1])], r[2].encode()) for r in rows],
                             system_config=sc, target_config=targets, augmentations=[])
        eval_sets[name] = (rows, eds)
    vrows = read_rows(vol_root, cfg["val_subset"])
    vds = AseLmdbDataset("omat", [local[k] for k in eval_files], [(efi[(r[0], r[1])], r[2].encode()) for r in vrows],
                         system_config=sc, target_config=targets, augmentations=[])

    # ---- model, optimiser, schedule ----
    utils.seed_everything(cfg["seed"])
    model = getattr(pretrained, cfg["arch_fn"])(system_config=sc, device=device, **cfg["arch_kwargs"])
    model.train()
    total = cfg["total_steps"]
    opt, sched = utils.get_optim(cfg["lr"], total, model)
    hooks = utils.gradient_clipping(model, cfg["clip"])
    ckdir = rd / "ckpt"
    ckdir.mkdir(parents=True, exist_ok=True)
    (rd / "eval").mkdir(exist_ok=True)
    latest = ckdir / "latest.pt"

    def cpu_sd():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def save_state_dict(name):
        tmp = ckdir / (name + ".tmp")
        torch.save(cpu_sd(), tmp)
        os.replace(tmp, ckdir / name)
        return sha256_file(ckdir / name)

    step, sstate, oom_skips = 0, (0, 0), 0
    if latest.exists():
        st = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, sstate, oom_skips = st["step"], tuple(st["sampler"]), st.get("oom_skips", 0)
        random.setstate(st["py_rng"])
        np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"].cpu())  # map_location moved these to the GPU; set_rng_state wants CPU
        if device.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        _log(f"resumed at step {step}, sampler {sstate}, oom_skips {oom_skips}")
    else:
        # step-0 checkpoint: the state_dict before the first optimiser step = the start fingerprint
        name = "ckpt_step000000.pt"
        fp = {"file": f"ckpt/{name}", "tensor_sha256": tensor_fingerprint(model.state_dict())}
        fp["file_sha256"] = save_state_dict(name)
        old = rec.get("start_fingerprint")
        if old and old["tensor_sha256"] != fp["tensor_sha256"]:  # a restart rebuilt a different model
            (rd / "STOP").write_text(f"start fingerprint mismatch: {old} vs {fp}\n")
            commit()
            return {"state": "stopped", "reason": "start fingerprint mismatch", "step": 0}
        rec["start_fingerprint"] = fp
        save_rec()
        commit()
        _log("step-0 fingerprint", fp)
    ch["start_step"] = step

    milestones = {int(round(total * q)): q for q in (0.25, 0.5, 0.75, 1.0)}
    ckpt_steps = [0] + sorted(milestones)

    def ckpt_name(s):
        return "ckpt_step000000.pt" if s == 0 else f"ckpt_{int(milestones[s]*100):03d}pct_step{s:06d}.pt"

    def full_eval(s, m=None):
        """Full val + clean-test evaluation of the checkpoint at step s (m = model holding it)."""
        m = m or model
        for name, (rows, eds) in eval_sets.items():
            res, per = evaluate_rows(m, eds, rows, device, conservative, cfg, rearm=rearm, per_structure=True)
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["split", "subset", "key", "sid", "natoms", "energy_err_eV", "forces_mae_eV_per_A",
                        "forces_rmse_eV_per_A", "stress_mae_eV_per_A3"])
            w.writerows(per)
            pp = rd / "eval" / f"step{s:06d}_{name}.csv.gz"
            with gzip.open(str(pp) + ".tmp", "wt") as g:
                g.write(buf.getvalue())
            os.replace(str(pp) + ".tmp", pp)
            for grp in ("all", "aimd", "rattled"):
                if res[grp]:
                    append_csv("val_metrics.csv", {"step": s, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                   "set": name, "group": grp, **res[grp],
                                                   "eval_seconds": res["seconds"]})
            _log(f"eval step {s} {name}: {res['all']} ({res['seconds']:.0f} s)")
        (rd / "eval" / f"step{s:06d}.done").write_text("ok\n")
        commit()

    def missing_evals(upto):
        return [s for s in ckpt_steps if s <= upto and not (rd / "eval" / f"step{s:06d}.done").exists()
                and (ckdir / ckpt_name(s)).exists()]

    for s in missing_evals(step):  # evaluations a previous chunk did not finish
        if left() < 1200:
            break
        if s == step:
            full_eval(s)
        else:
            em = getattr(pretrained, cfg["arch_fn"])(system_config=sc, device=device, **cfg["arch_kwargs"])
            em.load_state_dict(torch.load(ckdir / ckpt_name(s), map_location=device, weights_only=False))
            full_eval(s, em)
            del em

    bs = cfg["batch_size"]
    sampler = ResumableBatches(len(ds), bs, cfg["seed"], *sstate)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=cfg["num_workers"], collate_fn=base.batch_graphs,
                        worker_init_fn=utils.worker_init_fn, persistent_workers=True, prefetch_factor=4, timeout=600)

    def save_latest():
        tmp = ckdir / "latest.pt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "sampler": list(sstate), "oom_skips": oom_skips, "py_rng": random.getstate(),
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}, tmp)
        os.replace(tmp, latest)

    def write_status(extra):
        st = {"run": run_id, "chunk": chunk, "step": step, "total_steps": total, "oom_skips": oom_skips,
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"), **extra}
        (rd / "status.json").write_text(json.dumps(st, indent=1))

    def val_subset(s):
        res, _ = evaluate_rows(model, vds, vrows, device, conservative, cfg, rearm=rearm)
        for grp in ("all", "aimd", "rattled"):
            if res[grp]:
                append_csv("val_metrics.csv", {"step": s, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                               "set": "val_subset", "group": grp, **res[grp],
                                               "eval_seconds": res["seconds"]})
        _log(f"val subset step {s}: {res['all']}")

    it = iter(loader)
    last_ckpt = time.time()
    acc, acc_n, acc_struct, acc_atoms, t_log = None, 0, 0, 0, time.time()
    per_step = []
    fake_oom = set(cfg.get("debug_fake_oom_steps", []))
    stop_at = cfg.get("stop_at_step")
    state, reason = "chunk_end", "chunk wall time reached"
    write_status({"state": "running"})
    commit()
    while True:
        if step >= total:
            state, reason = "done", "all steps done"
            break
        if stop_at is not None and step >= stop_at:
            state, reason = "paused", f"stop_at_step {stop_at} reached"
            break
        if left() < cfg.get("save_margin_s", 600):
            break
        batch = next(it)
        oom = None
        try:
            if step in fake_oom:  # test hook, only set in the smoke run's config
                fake_oom.discard(step)
                raise torch.cuda.OutOfMemoryError("fake out-of-memory for testing the guard")
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model.loss(batch)
            loss = out.loss
            if step == 0 and chunk == 0 and not (rd / "loss_keys.json").exists():
                (rd / "loss_keys.json").write_text(json.dumps(sorted(out.log.keys())))
            finite = bool(torch.isfinite(loss))
            if finite:
                loss.backward()
        except torch.cuda.OutOfMemoryError as exc:
            oom = repr(exc)[:300]
        if oom is not None:
            oom_skips += 1
            append_csv("oom_skips.csv", {"step": step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                         "n_structures": int(len(batch.n_node)), "n_atoms": int(batch.n_node.sum()),
                                         "n_edges": int(batch.n_edge.sum()), "skips_total": oom_skips, "error": oom})
            batch = out = loss = None
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            sstate = ResumableBatches.advance(sstate, len(ds), bs)
            _log(f"OOM at step {step}: skipped batch ({oom_skips} so far)")
            if oom_skips >= cfg["max_oom_skips"]:
                state, reason = "stopped", f"{oom_skips} out-of-memory skips"
                (rd / "STOP").write_text(reason + "\n")
                break
            rearm()
            continue
        if not finite:
            state, reason = "stopped", f"non-finite loss at step {step}"
            (rd / "STOP").write_text(reason + "\n")
            break
        opt.step()
        sched.step()
        step += 1
        sstate = ResumableBatches.advance(sstate, len(ds), bs)
        vals = torch.stack([torch.as_tensor(out.log[k], device=loss.device).detach().reshape(()).float()
                            if k in out.log else torch.tensor(float("nan"), device=loss.device) for k, _ in TRAIN_COLS])
        acc = vals if acc is None else acc + vals
        acc_n += 1
        acc_struct += int(len(batch.n_node))
        acc_atoms += int(batch.n_node.sum())
        per_step.append((step, float(loss.detach())))
        rearm()
        if step % cfg["log_every"] == 0:
            dt = time.time() - t_log
            means = (acc / acc_n).tolist()
            row = {"step": step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "lr": sched.get_last_lr()[0],
                   "steps_per_s": acc_n / dt, "structures_per_s": acc_struct / dt, "atoms_per_step": acc_atoms / acc_n}
            row.update({col: v for (_, col), v in zip(TRAIN_COLS, means)})
            row["oom_skips_total"] = oom_skips
            append_csv("train_log.csv", row)
            with open(rd / "loss_per_step.csv", "a") as f:
                if f.tell() == 0:
                    f.write("step,loss\n")
                f.writelines(f"{s},{v}\n" for s, v in per_step)
            per_step = []
            acc, acc_n, acc_struct, acc_atoms, t_log = None, 0, 0, 0, time.time()
        if step in milestones:
            sha = save_state_dict(ckpt_name(step))
            rec.setdefault("checkpoints", {})[str(step)] = {"file": f"ckpt/{ckpt_name(step)}", "file_sha256": sha}
            save_rec()
            save_latest()
            write_status({"state": "running"})
            commit()
            last_ckpt = time.time()
            if left() > 1200:
                full_eval(step)
            continue
        if step % cfg["val_every"] == 0:
            val_subset(step)
        if time.time() - last_ckpt > cfg["ckpt_every_s"]:
            save_latest()
            write_status({"state": "running"})
            commit()
            last_ckpt = time.time()

    faulthandler.cancel_dump_traceback_later()
    for h in hooks:
        h.remove()
    if per_step:
        with open(rd / "loss_per_step.csv", "a") as f:
            if f.tell() == 0:
                f.write("step,loss\n")
            f.writelines(f"{s},{v}\n" for s, v in per_step)
    if state == "paused" and step % cfg["val_every"] != 0 and step not in milestones:
        val_subset(step)
    save_latest()  # always: also after a STOP, so the run can be inspected or resumed
    ch["latest_sha256"] = sha256_file(latest)
    ch.update(end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ"), end_step=step, state=state, reason=reason,
              oom_skips=oom_skips)
    rec["chunks"].append(ch)
    save_rec()
    write_status({"state": state, "reason": reason})
    commit()
    _log(f"chunk {chunk} ends: {state} ({reason}) at step {step}")
    return {"state": state, "reason": reason, "step": step, "copy_s": ch["copy_s"], "oom_skips": oom_skips}
