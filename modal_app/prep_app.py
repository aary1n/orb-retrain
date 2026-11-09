"""Modal app for everything before training: image test, data jobs, Phase A timing.

Runs as an ephemeral app via `modal run prep_app.py::<function>` from this
directory. Every function goes through the spend ledger before any work.
Global scope is kept tiny: only modal, time and the image/volume handles.
"""

import time

_T0 = time.time()  # first thing at import inside the container; ledger start time

import modal  # noqa: E402

PY = "3.11"
PIP = [
    "torch==2.6.0",  # PyPI wheel, bundles CUDA 12.4 runtime
    "orb-models==0.5.5",
    "ase==3.29.0",
    "numpy==2.4.6",
    "scipy==1.17.1",
    "dm-tree==0.1.8",
    "cached-path==1.8.10",
    "lmdb==3.0.0",
    "tqdm==4.70.1",
]
image = (
    modal.Image.debian_slim(python_version=PY)
    .pip_install(*PIP)
    .add_local_python_source("common", "dataset_lmdb", "data_jobs", "timing_job")
)

app = modal.App("orb-retrain-prep")
data_vol = modal.Volume.from_name("orb-retrain", version=2)
ledger_vol = modal.Volume.from_name("orb-retrain-ledger", version=2)
VOLS = {"/vol": data_vol, "/ledger": ledger_vol}


def _guarded(spec_kwargs: dict, body, *args, **kwargs):
    """Reserve in the ledger, heartbeat, run body, book. Never raises (so no Modal retry)."""
    import traceback
    from pathlib import Path

    import common as C

    spec = C.Spec(**spec_kwargs)
    extra = {
        "modal_task_id": __import__("os").environ.get("MODAL_TASK_ID"),
        "modal_function_call_id": modal.current_function_call_id(),
        "modal_input_id": modal.current_input_id(),
    }
    att = C.Attempt(Path(C.LEDGER_MOUNT), spec, reload=ledger_vol.reload, commit=ledger_vol.commit,
                    t_start=_T0, extra=extra)
    try:
        seen = att.check_and_reserve()
    except C.LedgerRefused as exc:
        print(f"[ledger] REFUSED: {exc}", flush=True)
        return {"refused": str(exc)}
    print(f"[ledger] reserved {spec.reservation_usd:.4f} USD for {spec.run}; total before {seen['total']:.4f}", flush=True)
    att.start_heartbeat()
    note, out = "ok", None
    try:
        out = body(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001
        note = f"error: {exc!r}"
        out = {"error": repr(exc), "traceback": traceback.format_exc()}
        print(out["traceback"], flush=True)
    finally:
        booked = att.book(note)
        print(f"[ledger] booked {booked:.4f} USD for {spec.run} ({note})", flush=True)
    if isinstance(out, dict):
        out["_ledger"] = {"attempt_id": att.attempt_id, "booked_usd": booked, "reserved_usd": spec.reservation_usd}
    return out


# --------------------------------------------------------------------------- CPU
TEST_RES = dict(cpu=(1.0, 1.0), memory=(4096, 4096))


@app.function(image=image, volumes=VOLS, **TEST_RES, timeout=600, startup_timeout=600, retries=0,
              max_containers=1, scaledown_window=2)
def image_test():
    def body():
        import subprocess

        import torch
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.atomic_system import SystemConfig

        sc = SystemConfig(radius=6.0, max_num_neighbors=120)
        n = {}
        for name, fn, kw in [
            ("conservative_full", pretrained.orb_v3_conservative_architecture, dict(latent_dim=256, base_mlp_hidden_dim=1024)),
            ("conservative_narrow", pretrained.orb_v3_conservative_architecture, dict(latent_dim=128, base_mlp_hidden_dim=512)),
            ("direct_full", pretrained.orb_v3_direct_architecture, dict(latent_dim=256, base_mlp_hidden_dim=1024)),
        ]:
            m = fn(system_config=sc, device="cpu", **kw)
            n[name] = sum(p.numel() for p in m.parameters())
        freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
        out = {"torch": str(torch.__version__), "cuda_build": str(torch.version.cuda), "params": n,
               "nproc": __import__("os").cpu_count(), "pip_freeze": freeze}
        print("IMAGE_TEST", {k: v for k, v in out.items() if k != "pip_freeze"}, flush=True)
        return out

    return _guarded(dict(run="image_test", phase="test", gpu=None, cpu=1.0, mem_gib=4.0,
                         startup_timeout=600, timeout=600), body)


DATA_RES = dict(cpu=(2.0, 2.0), memory=(8192, 8192))


@app.function(image=image, volumes=VOLS, **DATA_RES, timeout=10800, startup_timeout=600, retries=0,
              max_containers=1, scaledown_window=2)
def data_download():
    import data_jobs

    return _guarded(dict(run="data_download", phase="data", gpu=None, cpu=2.0, mem_gib=8.0,
                         startup_timeout=600, timeout=10800),
                    data_jobs.download_and_unpack, "/vol", data_vol.commit)


META_RES = dict(cpu=(8.0, 8.0), memory=(16384, 16384))


@app.function(image=image, volumes=VOLS, **META_RES, timeout=10800, startup_timeout=600, retries=0,
              max_containers=1, scaledown_window=2)
def data_metadata():
    import data_jobs

    def body():
        data_vol.reload()
        return data_jobs.extract_metadata("/vol", data_vol.commit, n_proc=8)

    return _guarded(dict(run="data_metadata", phase="data", gpu=None, cpu=8.0, mem_gib=16.0,
                         startup_timeout=600, timeout=10800), body)


# --------------------------------------------------------------------------- Phase A (GPU)
GPU_RES = dict(cpu=(8.0, 8.0), memory=(32768, 32768))
GPU_KW = dict(image=image, volumes=VOLS, **GPU_RES, timeout=900, startup_timeout=600, retries=0,
              max_containers=1, scaledown_window=2)


def _timing(gpu: str, tag: str):
    import timing_job

    def body():
        data_vol.reload()
        return timing_job.run_timing("/vol", gpu, tag, data_vol.commit, steps=200, warmup=10,
                                     batch_size=100, num_workers=7, deadline_s=900 - 90, t0=_T0)

    return _guarded(dict(run=f"phaseA_{tag}", phase="phaseA", gpu=gpu, cpu=8.0, mem_gib=32.0,
                         startup_timeout=600, timeout=900), body)


@app.function(gpu="H100!", **GPU_KW)
def time_h100():
    return _timing("H100!", "h100")


@app.function(gpu="A100-40GB", **GPU_KW)
def time_a100_40gb():
    return _timing("A100-40GB", "a100_40gb")


@app.function(gpu="L40S", **GPU_KW)
def time_l40s():
    return _timing("L40S", "l40s")
