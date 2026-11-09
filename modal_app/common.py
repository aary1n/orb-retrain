"""Shared constants, prices and the spend ledger.

Everything here is plain Python (no torch, no modal at import time) so that it
can be imported at the very top of every Modal function and unit-tested in
this session's container.

Prices were read from https://modal.com/pricing on 2026-11-02 (per second,
copied in retrain/timing/prices_2026-11-02.json).
"""

from __future__ import annotations

import calendar
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
GLOBAL_CAP_USD = 200.0  # SET BY AARYAN in the brief: whole-brief Modal cap.
# Sub-caps per phase. A phase may never pass its own cap, and the total may
# never pass GLOBAL_CAP_USD. "train" has no sub-cap beyond the global cap.
PHASE_CAPS_USD = {
    "test": 0.50,  # CPU image tests
    "data": 3.00,  # CPU download / unpack / metadata jobs
    "phaseA": 5.00,  # GPU timing (brief: cap $5, enforced by the ledger)
}
# Every Modal function, its end date. After this, attempts refuse to start.
END_DATE_UTC = "2026-11-30T00:00:00Z"  # placeholder; PLAN.md sets the real one

# ---------------------------------------------------------------------------
# Prices (USD per second), modal.com/pricing, read 2026-11-02
# ---------------------------------------------------------------------------
GPU_USD_PER_S = {
    "H100!": 0.001097,  # "Nvidia H100 SXM5"
    "A100-40GB": 0.000583,  # "Nvidia A100, 40 GB"
    "L40S": 0.000542,  # "Nvidia L40S"
    None: 0.0,
}
CPU_USD_PER_CORE_S = 0.0000131  # "Physical core (2 vCPU equivalent)"
MEM_USD_PER_GIB_S = 0.00000222


def price_per_hour(gpu: str | None, cpu_cores: float, mem_gib: float) -> float:
    """Full price per hour = GPU + cpu_request x CPU rate + memory_request x memory rate."""
    per_s = GPU_USD_PER_S[gpu] + cpu_cores * CPU_USD_PER_CORE_S + mem_gib * MEM_USD_PER_GIB_S
    return per_s * 3600.0


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
LEDGER_VOLUME = "orb-retrain-ledger"  # its own small Volume, version 2
LEDGER_MOUNT = "/ledger"
DATA_VOLUME = "orb-retrain"
DATA_MOUNT = "/vol"
DEAD_GRACE_S = 20 * 60  # a file with no clean end counts (last_alive - start + 20 min)
HEARTBEAT_S = 5 * 60
BOOK_MARGIN_S = 120  # added to every clean booking to cover container boot + scaledown


class LedgerRefused(Exception):
    pass


@dataclass
class Spec:
    """What an attempt is allowed to cost."""

    run: str  # e.g. "phaseA_h100", "data_download", "full_seed0"
    phase: str  # key into PHASE_CAPS_USD, or "train"
    gpu: str | None
    cpu: float
    mem_gib: float
    startup_timeout: int
    timeout: int
    others_at_once: int = 0  # other attempts that may run at the same time

    @property
    def price_per_hour(self) -> float:
        return price_per_hour(self.gpu, self.cpu, self.mem_gib)

    @property
    def reservation_usd(self) -> float:
        return (self.startup_timeout + self.timeout) / 3600.0 * self.price_per_hour


def entry_cost(e: dict, now: float | None = None) -> float:
    """Cost the ledger counts for one attempt file."""
    if e.get("clean_end"):
        return float(e["booked_usd"])
    if e.get("last_alive"):
        secs = float(e["last_alive"]) - float(e["start"]) + DEAD_GRACE_S
        return secs / 3600.0 * float(e["price_per_hour"])
    return float(e["reservation_usd"])


def read_ledger(root: Path) -> list[dict]:
    """Read every ledger file. Raises on any unreadable file."""
    entries = []
    base = root / "ledger"
    if not base.exists():
        return entries
    for p in sorted(base.rglob("*.json")):
        with open(p) as f:
            e = json.load(f)
        for k in ("run", "phase", "start", "price_per_hour", "reservation_usd"):
            if k not in e:
                raise ValueError(f"ledger file {p} lacks {k}")
        e["_path"] = str(p)
        entries.append(e)
    return entries


def totals(entries: list[dict]) -> dict:
    out = {"total": 0.0}
    for e in entries:
        c = entry_cost(e)
        out["total"] += c
        out[e["phase"]] = out.get(e["phase"], 0.0) + c
    return out


def _write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Attempt:
    """One attempt's ledger file. Only this container ever writes it.

    reload/commit are the Volume's methods (or no-ops in local tests).
    """

    def __init__(self, root: Path, spec: Spec, reload, commit, t_start: float | None = None,
                 global_cap: float = GLOBAL_CAP_USD, phase_caps: dict | None = None,
                 end_date_utc: str = END_DATE_UTC, extra: dict | None = None):
        self.root = Path(root)
        self.spec = spec
        self._reload = reload
        self._commit = commit
        self.t_start = t_start if t_start is not None else time.time()
        self.global_cap = global_cap
        self.phase_caps = PHASE_CAPS_USD if phase_caps is None else phase_caps
        self.end_date_utc = end_date_utc
        self.attempt_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.t_start)) + "-" + uuid.uuid4().hex[:8]
        self.path = self.root / "ledger" / spec.run / f"{self.attempt_id}.json"
        self.extra = extra or {}
        self.entry: dict | None = None
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- checks -----------------------------------------------------------
    def check_and_reserve(self) -> dict:
        """Reload, add up, refuse or reserve. Returns the totals seen."""
        if (self.root / "STOP").exists():
            raise LedgerRefused("global STOP file exists on the ledger volume")
        end_ts = calendar.timegm(time.strptime(self.end_date_utc, "%Y-%m-%dT%H:%M:%SZ"))
        if time.time() > end_ts:
            raise LedgerRefused(f"end date {self.end_date_utc} has passed")
        try:
            self._reload()
            entries = read_ledger(self.root)
        except Exception as exc:  # unreadable ledger => refuse
            raise LedgerRefused(f"ledger cannot be read: {exc!r}")
        t = totals(entries)
        mine = self.spec.reservation_usd
        others = self.spec.others_at_once * mine
        if t["total"] + mine + others > self.global_cap:
            raise LedgerRefused(
                f"global cap: total {t['total']:.4f} + mine {mine:.4f} + others {others:.4f} > {self.global_cap}")
        pcap = self.phase_caps.get(self.spec.phase)
        if pcap is not None and t.get(self.spec.phase, 0.0) + mine + others > pcap:
            raise LedgerRefused(
                f"phase cap {self.spec.phase}: {t.get(self.spec.phase, 0.0):.4f} + {mine:.4f} + {others:.4f} > {pcap}")
        self.entry = {
            "run": self.spec.run,
            "phase": self.spec.phase,
            "attempt_id": self.attempt_id,
            "gpu": self.spec.gpu,
            "cpu": self.spec.cpu,
            "mem_gib": self.spec.mem_gib,
            "startup_timeout": self.spec.startup_timeout,
            "timeout": self.spec.timeout,
            "price_per_hour": self.spec.price_per_hour,
            "reservation_usd": mine,
            "start": self.t_start,
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.t_start)),
            "last_alive": None,
            "end": None,
            "clean_end": False,
            "booked_usd": None,
            "ledger_total_before": t["total"],
            **self.extra,
        }
        try:
            _write_atomic(self.path, self.entry)
            self._commit()
        except Exception as exc:
            raise LedgerRefused(f"could not write/commit ledger file: {exc!r}")
        return t

    # -- heartbeat ----------------------------------------------------------
    def _beat(self) -> None:
        with self._lock:
            assert self.entry is not None
            self.entry["last_alive"] = time.time()
            _write_atomic(self.path, self.entry)
            self._commit()

    def start_heartbeat(self, every_s: float = HEARTBEAT_S) -> None:
        def loop():
            while not self._hb_stop.wait(every_s):
                try:
                    self._beat()
                except Exception as exc:  # keep beating; log only
                    print(f"[ledger] heartbeat failed: {exc!r}", flush=True)

        self._hb_thread = threading.Thread(target=loop, daemon=True, name="ledger-heartbeat")
        self._hb_thread.start()

    # -- clean end ----------------------------------------------------------
    def book(self, note: str = "") -> float:
        self._hb_stop.set()
        with self._lock:
            assert self.entry is not None
            now = time.time()
            secs = now - self.t_start + BOOK_MARGIN_S
            self.entry.update(
                last_alive=now,
                end=now,
                end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                clean_end=True,
                booked_seconds=secs,
                booked_usd=secs / 3600.0 * self.spec.price_per_hour,
                note=note,
            )
            _write_atomic(self.path, self.entry)
            self._commit()
            return self.entry["booked_usd"]
