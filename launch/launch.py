"""Start, resume or inspect a training run. It only spawns into the deployed app `orb-retrain`.

Runs wherever the Modal client is logged in (this session's container, or Aaryan's
laptop with `python -m modal`). It never trains anything itself; every GPU attempt
goes through the same spend ledger inside the deployed function.

  python launch.py start full <seed> [--stop-at N]   first chunk of a new run (stops at step N if given)
  python launch.py start smoke 0                      the smoke test (small eval sets, fake OOM at 2 steps)
  python launch.py resume <run_id> [--stop-at N]      continue a paused run from its latest checkpoint
  python launch.py status <run_id>                    print status.json and the ledger total

Only condition "full" is built; the A/B removal arms are not built yet (by decision of 2 Nov 2026).
"""

import argparse
import inspect
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VOL = "orb-retrain"
APP = "orb-retrain"

AIMD = ["aimd-from-PBE-1000-npt", "aimd-from-PBE-1000-nvt", "aimd-from-PBE-3000-npt", "aimd-from-PBE-3000-nvt"]
RATTLED = ["rattled-300", "rattled-300-subsampled", "rattled-500", "rattled-500-subsampled", "rattled-1000",
           "rattled-1000-subsampled", "rattled-relax"]


def vol():
    return modal.Volume.from_name(VOL, version=2)


def read_json(v, path):
    try:
        return json.loads(b"".join(v.read_file(path)))
    except Exception:  # noqa: BLE001  (missing file)
        return None


def put_json(v, path, obj):
    with v.batch_upload(force=True) as b:
        b.put_file(io.BytesIO(json.dumps(obj, indent=1).encode()), path)


def git_commit():
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "modal_app"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        return {"commit": h, "modal_app_uncommitted_changes": bool(dirty)}
    except Exception:  # noqa: BLE001
        return {"commit": None}


def arch_record(arch_kwargs, system_config):
    """Every argument of orb_v3_conservative_architecture (defaults filled in) plus the SystemConfig."""
    try:
        from orb_models.forcefield import pretrained

        sig = inspect.signature(pretrained.orb_v3_conservative_architecture)
        args = {k: (p.default if p.default is not inspect.Parameter.empty else None) for k, p in sig.parameters.items()}
    except Exception:  # noqa: BLE001  (orb-models not installed here: record the 0.5.5 defaults read from source)
        args = {"latent_dim": 256, "base_mlp_hidden_dim": 1024, "base_mlp_depth": 2, "head_mlp_hidden_dim": 256,
                "head_mlp_depth": 1, "num_message_passing_steps": 5, "activation": "silu",
                "has_charge_spin_cond": False, "has_stress": True, "device": None, "system_config": None}
    args.update(arch_kwargs)
    args.pop("device", None)
    args["system_config"] = system_config
    return {"function": "orb_models.forcefield.pretrained.orb_v3_conservative_architecture",
            "orb_models_version": "0.5.5", "arguments": args}


def make_config(condition, seed, stop_at):
    smoke = condition == "smoke"
    run_id = f"smoke_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}" if smoke else f"{condition}_seed{seed}"
    manifest = json.loads((ROOT / "lists" / "MANIFEST.json").read_text())
    files = [["train", s] for s in AIMD] + [[sp, s] for sp in ("val", "test") for s in AIMD + RATTLED]
    sc = {"radius": 6.0, "max_num_neighbors": 120}
    cfg = {
        "run_id": run_id, "condition": condition, "seed": seed,
        "arch_fn": "orb_v3_conservative_architecture",
        "arch_kwargs": {"latent_dim": 256, "base_mlp_hidden_dim": 1024},
        "system_config": sc,
        "total_steps": 400 if smoke else 62_000, "batch_size": 100, "lr": 3e-4, "clip": 0.5,
        "train_sids": "lists/full_train_sids.txt.gz",
        "train_sids_sha256_of_lines": manifest["full_train_sids.txt.gz"]["sha256_of_lines"],
        "n_train": 551_578,
        "eval_sets": ({"val": "lists/smoke_val_rows.tsv.gz", "test_clean": "lists/smoke_test_rows.tsv.gz"} if smoke
                      else {"val": "lists/val_all_rows.tsv.gz", "test_clean": "lists/test_clean_rows.tsv.gz"}),
        "val_subset": "lists/smoke_val_subset_rows.tsv.gz" if smoke else "lists/val_subset5000_rows.tsv.gz",
        "lists_manifest": {k: v for k, v in manifest.items() if not k.startswith("_")},
        "files_needed": files,
        "log_every": 50 if smoke else 100, "val_every": 100 if smoke else 2_500,
        "ckpt_every_s": 1800, "chunk_wall_s": 6 * 3600, "save_margin_s": 600,
        "num_workers": 7, "eval_workers": 4, "eval_max_structs": 100, "eval_max_atoms": 2500,
        "max_oom_skips": 20, "watchdog_s": 900,
        "stop_at_step": stop_at, "auto_spawn_next": stop_at is None,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": git_commit(),
    }
    if smoke:
        cfg["debug_fake_oom_steps"] = [150, 151]
    return cfg, arch_record(cfg["arch_kwargs"], sc)


def spawn(v, run_id, chunk):
    locks = modal.Dict.from_name("orb-retrain-locks", create_if_missing=True)
    if not locks.put(f"{run_id}/next/{chunk}", True, skip_if_exists=True):
        sys.exit(f"refused: chunk {chunk} of {run_id} was already spawned once (lock key exists)")
    fn = modal.Function.from_name(APP, "train_chunk")
    try:
        fn.hydrate()
    except Exception as exc:  # noqa: BLE001
        locks.pop(f"{run_id}/next/{chunk}", None)
        sys.exit(f"cannot reach {APP}.train_chunk ({exc!r}). The app stops itself when the last approved run ends; "
                 f"redeploy with:  cd retrain/modal_app && modal deploy train_app.py")
    call = fn.spawn(run_id, chunk)
    rec = {"run_id": run_id, "chunk": chunk, "function_call_id": call.object_id,
           "spawned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    local = ROOT / "runs_local" / run_id
    local.mkdir(parents=True, exist_ok=True)
    with open(local / "launches.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return rec


def cmd_start(a):
    if a.condition not in ("full", "smoke"):
        sys.exit(f"condition {a.condition!r} is not built yet")
    cfg, arch = make_config(a.condition, a.seed, a.stop_at)
    v = vol()
    rid = cfg["run_id"]
    if read_json(v, f"runs/{rid}/config.json") is not None:
        sys.exit(f"refused: runs/{rid}/config.json already exists on the Volume; use resume")
    local = ROOT / "runs_local" / rid
    local.mkdir(parents=True, exist_ok=True)
    (local / "config.json").write_text(json.dumps(cfg, indent=1))
    (local / "arch.json").write_text(json.dumps(arch, indent=1))
    put_json(v, f"runs/{rid}/config.json", cfg)
    put_json(v, f"runs/{rid}/arch.json", arch)
    if a.condition != "smoke":  # the smoke run is not an approved run, so it never triggers the app stop
        ap = read_json(v, "approved_runs.json") or {"runs": []}
        if rid not in ap["runs"]:
            ap["runs"].append(rid)
        ap["updated_utc"] = cfg["created_utc"]
        put_json(v, "approved_runs.json", ap)
    spawn(v, rid, 0)


def cmd_resume(a):
    v = vol()
    st = read_json(v, f"runs/{a.run_id}/status.json")
    cfg = read_json(v, f"runs/{a.run_id}/config.json")
    if st is None or cfg is None:
        sys.exit("no status/config for that run on the Volume")
    if st.get("state") != "paused":
        sys.exit(f"refused: run state is {st.get('state')!r}, not 'paused'")
    cfg["stop_at_step"] = a.stop_at
    cfg["auto_spawn_next"] = a.stop_at is None
    cfg.setdefault("resumes", []).append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          "from_step": st["step"], "stop_at_step": a.stop_at})
    put_json(v, f"runs/{a.run_id}/config.json", cfg)
    (ROOT / "runs_local" / a.run_id).mkdir(parents=True, exist_ok=True)
    (ROOT / "runs_local" / a.run_id / "config.json").write_text(json.dumps(cfg, indent=1))
    att = read_json(v, f"runs/{a.run_id}/attempts.json") or {"by_chunk": {}}
    tried = [int(k) for k in att["by_chunk"]] + [int(st["chunk"])]
    spawn(v, a.run_id, max(tried) + 1)  # the next chunk number no attempt has used


def cmd_status(a):
    v = vol()
    print(json.dumps(read_json(v, f"runs/{a.run_id}/status.json"), indent=1))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("start")
    s.add_argument("condition")
    s.add_argument("seed", type=int)
    s.add_argument("--stop-at", type=int, default=None)
    s.set_defaults(f=cmd_start)
    r = sp.add_parser("resume")
    r.add_argument("run_id")
    r.add_argument("--stop-at", type=int, default=None)
    r.set_defaults(f=cmd_resume)
    t = sp.add_parser("status")
    t.add_argument("run_id")
    t.set_defaults(f=cmd_status)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
