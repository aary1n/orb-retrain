"""Size the training plan against the ledger's own refusal rule (common.py).

Runs in this session's container; costs nothing. Inputs are measured Phase A
numbers (timing/h100.json) and the prices in common.py. Estimates are labelled.

Model: N runs, k at once, same steps per run. Each run is split into chunks of
at most 6 h of wall time. Every chunk start is checked exactly like
Attempt.check_and_reserve():
  total(ledger) + R + (k-1)*R <= cap,
where a finished chunk counts its booked time (+120 s margin), a running chunk
counts (last heartbeat - start + 20 min) x price, or the full reservation R if
it has not yet written a heartbeat (first 5 minutes).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))
import common as C  # noqa: E402

T = json.loads((Path(__file__).resolve().parents[1] / "timing" / "h100.json").read_text())
import os
MODEL = os.environ.get("MODEL", "conservative_full")
M = T["models"][MODEL]
S_PER_STEP = M["s_per_step"]  # measured, batch 100, includes data wait
EVAL_SPS = M["eval_structures_per_s"]  # measured
PRICE = C.price_per_hour("H100!", 8.0, 32.0)
TIMEOUT = 6 * 3600 + 15 * 60
R = (600 + TIMEOUT) / 3600 * PRICE
CHUNK = 6 * 3600
# ESTIMATES (not measured): container start to first step, per chunk; and end-of-chunk save + val.
START_OVERHEAD_S = 180
END_OVERHEAD_S = 60
N_VAL = 107_732
N_TEST = 53_687
VAL_SUBSET = 5_000
VAL_SUBSET_EVERY = 2_500


def run_seconds(steps: int) -> tuple[float, int]:
    """Wall seconds of training + evaluation for one run, and its number of chunks."""
    train = steps * S_PER_STEP
    evals = 4 * N_VAL / EVAL_SPS + N_TEST / EVAL_SPS + (steps / VAL_SUBSET_EVERY) * VAL_SUBSET / EVAL_SPS
    work = train + evals
    usable = CHUNK - START_OVERHEAD_S - END_OVERHEAD_S - 600  # 600 s save margin inside the chunk
    n_chunks = math.ceil(work / usable)
    return work + n_chunks * (START_OVERHEAD_S + END_OVERHEAD_S), n_chunks


def simulate(n_runs: int, k: int, steps: int, pre_spend: float, cap: float = C.GLOBAL_CAP_USD, stagger_s: float = 330):
    """Event simulation. Returns (ok, final_total, wall_hours, refusals, chunk_starts)."""
    total_s, n_chunks = run_seconds(steps)
    per_chunk = total_s / n_chunks
    booked = pre_spend
    running = []  # (start_time, end_time, run)
    queue = list(range(n_runs))
    next_chunk = {r: 0 for r in range(n_runs)}
    t = 0.0
    refusals = 0
    starts = []
    last_start = -1e9

    def ledger_total(now):
        s = booked
        for st, en, _ in running:
            el = now - st
            if el < 300:
                s += R
            else:
                s += (math.floor(el / 300) * 300 + 1200) / 3600 * PRICE
        return s

    active_runs = []
    while queue or active_runs or running:
        busy = {x[2] for x in running}
        pending = [r for r in active_runs if r not in busy] + queue
        free = len(running) < k
        if free and pending and (not running or t - last_start >= stagger_s - 1e-6):
            r = pending[0]
            tot = ledger_total(t)
            if tot + R + (k - 1) * R > cap:
                refusals += 1
                return False, booked, t / 3600, refusals, starts
            if r in queue:
                queue.remove(r)
                active_runs.append(r)
            running.append((t, t + per_chunk, r))
            starts.append((round(t / 3600, 2), r, next_chunk[r], round(tot, 2)))
            next_chunk[r] += 1
            last_start = t
            continue
        events = [en for _, en, _ in running]
        if free and pending and running:
            events.append(last_start + stagger_s)
        t = max(t, min(events))
        for x in [x for x in running if x[1] <= t]:
            running.remove(x)
            booked += (x[1] - x[0] + C.BOOK_MARGIN_S) / 3600 * PRICE
            if next_chunk[x[2]] >= n_chunks:
                active_runs.remove(x[2])
    return True, booked, t / 3600, refusals, starts


def best_steps(n_runs, k, pre_spend, keep_free=15.0):
    """Largest steps (multiple of 1,000) for which every chunk start passes and the end total <= cap - keep_free."""
    lo, hi = 1_000, 400_000
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2 // 1000 * 1000
        ok, tot, wall, _, _ = simulate(n_runs, k, mid, pre_spend)
        if ok and tot <= C.GLOBAL_CAP_USD - keep_free:
            best = (mid, tot, wall)
            lo = mid + 1000
        else:
            hi = mid - 1000
    return best


if __name__ == "__main__":
    pre = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    print(f"model {MODEL}  price/h {PRICE:.4f}  reservation per chunk R {R:.2f}  s/step {S_PER_STEP:.4f}  eval {EVAL_SPS:.0f}/s  pre-spend {pre}")
    out = []
    for n_runs, label in [(9, "3 conditions x 3 seeds"), (6, "3 conditions x 2 seeds")]:
        for k in (1, 2, 3):
            b = best_steps(n_runs, k, pre)
            if b is None:
                continue
            steps, tot, wall = b
            secs, nch = run_seconds(steps)
            row = {"runs": n_runs, "label": label, "at_once": k, "steps_per_run": steps,
                   "epochs_full_set": steps * 100 / 551_578, "structures_seen_M": steps * 100 / 1e6,
                   "hours_per_run": secs / 3600, "chunks_per_run": nch,
                   "usd_per_run": secs / 3600 * PRICE + nch * C.BOOK_MARGIN_S / 3600 * PRICE,
                   "end_total_usd": tot, "wall_hours_all_runs": wall}
            out.append(row)
            print(json.dumps({k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in row.items()}))
    (Path(__file__).parent / f"sizing_{MODEL}.json").write_text(json.dumps(out, indent=1))
