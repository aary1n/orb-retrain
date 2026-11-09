"""Local tests of the spend ledger (run in this session's container; no Modal)."""
import json, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_app"))
import common as C

def mk(root, run="r", phase="phaseA", gpu="H100!", others=0, **kw):
    spec = C.Spec(run=run, phase=phase, gpu=gpu, cpu=8, mem_gib=32, startup_timeout=600, timeout=900, others_at_once=others)
    return C.Attempt(Path(root), spec, reload=lambda: None, commit=lambda: None, **kw)

def test_price():
    assert abs(C.price_per_hour("H100!", 0, 0) - 3.9492) < 1e-3
    assert abs(C.price_per_hour("A100-40GB", 0, 0) - 2.0988) < 1e-3
    assert abs(C.price_per_hour("L40S", 0, 0) - 1.9512) < 1e-3

def test_reserve_and_caps():
    with tempfile.TemporaryDirectory() as d:
        a = mk(d); t = a.check_and_reserve()
        assert t["total"] == 0
        e = json.load(open(a.path)); assert e["clean_end"] is False and e["last_alive"] is None
        # unfinished, no heartbeat => full reservation counts
        tot = C.totals(C.read_ledger(Path(d)))
        assert abs(tot["total"] - a.spec.reservation_usd) < 1e-9
        a.book("ok")
        tot = C.totals(C.read_ledger(Path(d)))
        assert tot["total"] < a.spec.reservation_usd  # booked real time (+120 s margin)
        # phase cap: reservation ~ 1.91 USD each, cap 5 => after 2 reserved & unfinished, third refused
        b = mk(d, run="b"); b.check_and_reserve()
        c = mk(d, run="c"); c.check_and_reserve()
        try:
            mk(d, run="x").check_and_reserve(); raise AssertionError("should refuse")
        except C.LedgerRefused as ex:
            assert "phase cap" in str(ex)

def test_dead_with_heartbeat():
    e = dict(start=1000.0, last_alive=1000.0 + 3600, price_per_hour=4.0, reservation_usd=99.0, clean_end=False)
    assert abs(C.entry_cost(e) - 4.0 * (1 + 20/60)) < 1e-9

def test_others_at_once_and_global():
    with tempfile.TemporaryDirectory() as d:
        a = mk(d, phase="train", others=2, global_cap=5.0)
        try:
            a.check_and_reserve(); raise AssertionError("should refuse: 3 x 1.91 > 5")
        except C.LedgerRefused as ex:
            assert "global cap" in str(ex)

def test_unreadable_refuses():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ledger" / "bad"; p.mkdir(parents=True); (p / "x.json").write_text("{not json")
        try:
            mk(d).check_and_reserve(); raise AssertionError("should refuse")
        except C.LedgerRefused as ex:
            assert "cannot be read" in str(ex)

def test_stop_and_end_date():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "STOP").write_text("x")
        try:
            mk(d).check_and_reserve(); raise AssertionError
        except C.LedgerRefused as ex:
            assert "STOP" in str(ex)
    with tempfile.TemporaryDirectory() as d:
        try:
            mk(d, end_date_utc="2020-01-01T00:00:00Z").check_and_reserve(); raise AssertionError
        except C.LedgerRefused as ex:
            assert "end date" in str(ex)

def test_heartbeat():
    with tempfile.TemporaryDirectory() as d:
        a = mk(d); a.check_and_reserve(); a.start_heartbeat(every_s=0.05); time.sleep(0.2)
        e = json.load(open(a.path)); assert e["last_alive"] is not None
        a.book()

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
