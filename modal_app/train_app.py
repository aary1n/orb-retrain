"""Deployed Modal app `orb-retrain`: one GPU training function, chunks spawn the next.

Deploy:  cd retrain/modal_app && modal deploy train_app.py
Start a run:  python launch.py start <condition> <seed> [--stop-at N]   (or ../launch_run.sh)
Resume a paused run:  python launch.py resume <run_id>

GPU, CPU, memory and the number of runs at once come from plan_constants.py,
which must match PLAN.md.
"""

import time

import modal

import plan_constants as P  # noqa: E402  (plain constants, no heavy imports)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*P.PIP)
    .add_local_python_source("common", "dataset_lmdb", "train_job", "plan_constants", "ref_eval")
)
app = modal.App("orb-retrain")
data_vol = modal.Volume.from_name("orb-retrain", version=2)
ledger_vol = modal.Volume.from_name("orb-retrain-ledger", version=2)


@app.function(
    image=image,
    gpu=P.GPU,  # exact type, no fallback list
    cpu=(P.CPU, P.CPU),
    memory=(P.MEM_MIB, P.MEM_MIB),
    timeout=P.TIMEOUT_S,  # 6 h 15 min backstop; the code ends each chunk itself at 6 h 00
    startup_timeout=600,
    retries=modal.Retries(max_retries=1, initial_delay=60.0),
    max_containers=P.MAX_AT_ONCE,
    scaledown_window=2,
    volumes={"/vol": data_vol, "/ledger": ledger_vol},
)
def train_chunk(run_id: str, chunk: int):
    t0 = time.time()  # this attempt's start: ledger start time and the 6 h chunk clock (containers may be reused)
    # ---- first lines: restart counter, STOP, end date. No torch, no GPU yet. ----
    import calendar
    import json
    import os

    rd = f"/vol/runs/{run_id}"
    data_vol.reload()
    cpath = f"{rd}/attempts.json"
    c = json.load(open(cpath)) if os.path.exists(cpath) else {"by_chunk": {}}
    c["by_chunk"][str(chunk)] = c["by_chunk"].get(str(chunk), 0) + 1
    restarts = sum(max(0, n - 1) for n in c["by_chunk"].values())
    c["restarts"] = restarts
    os.makedirs(rd, exist_ok=True)
    with open(cpath + ".tmp", "w") as f:
        json.dump(c, f)
    os.replace(cpath + ".tmp", cpath)
    data_vol.commit()
    end_ts = calendar.timegm(time.strptime(P.END_DATE_UTC, "%Y-%m-%dT%H:%M:%SZ"))
    if restarts > P.MAX_RESTARTS:
        return {"stopped": f"restart counter {restarts} > {P.MAX_RESTARTS}"}
    if os.path.exists(f"{rd}/STOP") or os.path.exists("/vol/STOP"):
        return {"stopped": "STOP file"}
    if time.time() > end_ts:
        return {"stopped": f"end date {P.END_DATE_UTC} passed"}

    # ---- ledger ----
    from pathlib import Path

    import common as C

    spec = C.Spec(run=run_id, phase="train", gpu=P.GPU, cpu=P.CPU, mem_gib=P.MEM_MIB / 1024,
                  startup_timeout=600, timeout=P.TIMEOUT_S, others_at_once=P.MAX_AT_ONCE - 1)
    att = C.Attempt(Path("/ledger"), spec, reload=ledger_vol.reload, commit=ledger_vol.commit, t_start=t0,
                    end_date_utc=P.END_DATE_UTC,
                    extra={"chunk": chunk, "modal_input_id": modal.current_input_id(),
                           "modal_function_call_id": modal.current_function_call_id()})
    try:
        att.check_and_reserve()
    except C.LedgerRefused as exc:
        _maybe_stop_app(f"ledger refused: {exc}")
        return {"refused": str(exc)}

    # ---- one trainer per run ----
    locks = modal.Dict.from_name("orb-retrain-locks", create_if_missing=True)
    me = {"input_id": modal.current_input_id(), "attempt": att.attempt_id, "chunk": chunk, "last_alive": time.time()}
    if not locks.put(run_id, me, skip_if_exists=True):
        holder = locks.get(run_id) or {}
        # A Modal retry or pre-emption restart of this same chunk keeps the input id before the colon and
        # changes the suffix. Modal never runs one input twice at once, so such a holder is a dead attempt.
        same_input = str(holder.get("input_id")).split(":")[0] == str(me["input_id"]).split(":")[0]
        if not same_input and time.time() - holder.get("last_alive", 0) < 20 * 60:
            att.book("exit: another trainer holds the lock")
            return {"exit": "lock held", "holder": holder}
        locks.put(run_id, me)  # stale lock or our own earlier attempt: take it over

    def beat():
        me["last_alive"] = time.time()
        locks.put(run_id, me)

    att.start_heartbeat()
    import threading

    stop_beat = threading.Event()

    def beat_loop():
        while not stop_beat.wait(300):
            try:
                beat()
            except Exception as exc:  # noqa: BLE001
                print(f"[lock] heartbeat failed: {exc!r}", flush=True)

    threading.Thread(target=beat_loop, daemon=True).start()

    import train_job

    try:
        out = train_job.run_chunk("/vol", run_id, chunk, t0, data_vol.commit)
    except BaseException as exc:  # book the real time, then let Modal retry once (max_retries=1)
        stop_beat.set()
        att.book(f"error: {exc!r}"[:300])
        raise
    stop_beat.set()
    booked = att.book(out.get("state", ""))
    locks.pop(run_id, None)

    # ---- last step: spawn the next chunk only if everything allows it ----
    if out.get("state") == "chunk_end":
        ok, why = _may_continue(run_id, att)
        if ok and locks.put(f"{run_id}/next/{chunk + 1}", True, skip_if_exists=True):
            modal.Function.from_name("orb-retrain", "train_chunk").spawn(run_id, chunk + 1)
            out["spawned_next"] = chunk + 1
        else:
            out["not_spawned"] = why
    _maybe_stop_app("after chunk")
    out["booked_usd"] = booked
    return out


def _may_continue(run_id, att):
    import json
    import os

    import common as C

    rd = f"/vol/runs/{run_id}"
    data_vol.reload()
    if os.path.exists(f"{rd}/STOP") or os.path.exists("/vol/STOP"):
        return False, "STOP file"
    if time.time() > __import__("calendar").timegm(time.strptime(P.END_DATE_UTC, "%Y-%m-%dT%H:%M:%SZ")):
        return False, "end date passed"
    c = json.load(open(f"{rd}/attempts.json"))
    if c.get("restarts", 0) > P.MAX_RESTARTS:
        return False, "restart counter"
    cfg = json.load(open(f"{rd}/config.json"))
    if not cfg.get("auto_spawn_next", False):
        return False, "auto_spawn_next is off for this run"
    try:
        ledger_vol.reload()
        t = C.totals(C.read_ledger(att.root))["total"]
    except Exception as exc:  # noqa: BLE001
        return False, f"ledger unreadable: {exc!r}"
    if t + att.spec.reservation_usd * P.MAX_AT_ONCE > C.GLOBAL_CAP_USD:
        return False, f"ledger total {t:.2f} too close to cap"
    return True, ""


def _maybe_stop_app(why: str):
    """Stop the whole app when every approved run is done, or the cap / end date is hit."""
    import json
    import os

    import common as C

    try:
        data_vol.reload()
        ap = json.load(open("/vol/approved_runs.json"))["runs"] if os.path.exists("/vol/approved_runs.json") else []
        states = []
        for r in ap:
            p = f"/vol/runs/{r}/status.json"
            states.append(json.load(open(p)).get("state") if os.path.exists(p) else None)
        all_done = bool(ap) and all(s in ("done", "stopped") for s in states)
        ledger_vol.reload()
        t = C.totals(C.read_ledger(__import__("pathlib").Path("/ledger")))["total"]
        end_ts = __import__("calendar").timegm(time.strptime(P.END_DATE_UTC, "%Y-%m-%dT%H:%M:%SZ"))
        if all_done or t >= C.GLOBAL_CAP_USD - 1.0 or time.time() > end_ts:
            print(f"[stop] stopping app orb-retrain ({why}; all_done={all_done}, total={t:.2f})", flush=True)
            modal.App.lookup("orb-retrain").stop()
    except Exception as exc:  # noqa: BLE001
        print(f"[stop] check failed: {exc!r}", flush=True)


@app.function(image=image, cpu=(1.0, 1.0), memory=(4096, 4096), timeout=600, startup_timeout=600, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/vol": data_vol, "/ledger": ledger_vol})
def image_check(seed: int = 0):
    """CPU-only test of the training image, through the ledger. Builds the model for `seed` on CPU and
    returns its step-0 tensor fingerprint (initialisation happens on CPU, so a GPU run must match it)."""
    import os
    from pathlib import Path

    import common as C

    t0 = time.time()
    spec = C.Spec(run="train_image_check", phase="test", gpu=None, cpu=1.0, mem_gib=4.0, startup_timeout=600,
                  timeout=600)
    att = C.Attempt(Path("/ledger"), spec, reload=ledger_vol.reload, commit=ledger_vol.commit, t_start=t0)
    try:
        att.check_and_reserve()
    except C.LedgerRefused as exc:
        return {"refused": str(exc)}
    try:
        import torch

        import dataset_lmdb  # noqa: F401
        import train_job
        from orb_models import utils
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.atomic_system import SystemConfig

        utils.set_torch_precision("float32-high")
        utils.seed_everything(seed)
        m = pretrained.orb_v3_conservative_architecture(latent_dim=256, base_mlp_hidden_dim=1024, device="cpu",
                                                        system_config=SystemConfig(radius=6.0, max_num_neighbors=120))
        data_vol.reload()
        out = {"torch": str(torch.__version__), "cuda_build": str(torch.version.cuda),
               "params_trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
               "step0_tensor_sha256": train_job.tensor_fingerprint(m.state_dict()),
               "volume_has": {p: os.path.exists(f"/vol/{p}") for p in
                              ["meta/train.csv.gz", "lists", "omat24_1M/train/aimd-from-PBE-1000-nvt/train.aselmdb"]}}
        note = "ok"
    except BaseException as exc:  # noqa: BLE001
        out, note = {"error": repr(exc)}, f"error: {exc!r}"[:300]
    out["booked_usd"] = att.book(note)
    return out


@app.function(image=image, gpu=P.GPU, cpu=(P.CPU, P.CPU), memory=(P.MEM_MIB, P.MEM_MIB), timeout=1200,
              startup_timeout=600, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/vol": data_vol, "/ledger": ledger_vol})
def eval_reference(run_id: str = "full_seed0"):
    """One-off: the released orb-v3-conservative-inf-omat checkpoint on the run's full val and clean test sets.
    Through the ledger (phase "eval"). Writes only reference_* files under runs/<run_id>/."""
    import traceback
    from pathlib import Path

    import common as C

    t0 = time.time()
    spec = C.Spec(run=f"ref_{run_id}", phase="eval", gpu=P.GPU, cpu=P.CPU, mem_gib=P.MEM_MIB / 1024,
                  startup_timeout=600, timeout=1200)
    att = C.Attempt(Path("/ledger"), spec, reload=ledger_vol.reload, commit=ledger_vol.commit, t_start=t0,
                    end_date_utc=P.END_DATE_UTC)
    try:
        att.check_and_reserve()
    except C.LedgerRefused as exc:
        return {"refused": str(exc)}
    att.start_heartbeat()
    try:
        import ref_eval

        data_vol.reload()
        out = ref_eval.run_reference("/vol", run_id, data_vol.commit)
        note = "ok"
    except BaseException as exc:  # noqa: BLE001
        out, note = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}, f"error: {exc!r}"[:300]
    out["booked_usd"] = att.book(note)
    return out
