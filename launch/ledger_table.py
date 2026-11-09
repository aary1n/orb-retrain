"""Pull the ledger Volume and print the STATUS.md ledger table (runs in this session's container)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modal_app"))
import common as C  # noqa: E402


def main():
    d = ROOT / "data" / "ledger_copy"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    subprocess.run(["modal", "volume", "get", "orb-retrain-ledger", "ledger", str(d)], check=True, capture_output=True)
    E = sorted(C.read_ledger(d), key=lambda e: e["start"])
    lines = ["| Start (UTC) | Run | Phase | GPU | Chunk | End | Booked $ | Counted $ | Reserved $ |", "|---|---|---|---|---|---|---|---|---|"]
    for e in E:
        end = "clean" if e["clean_end"] else ("running (heartbeat)" if e.get("last_alive") else "no heartbeat yet")
        booked = f"{e['booked_usd']:.4f}" if e["clean_end"] else "–"
        lines.append(f"| {e['start_utc'][11:19]} | {e['run']} | {e['phase']} | {e['gpu'] or 'none'} | {e.get('chunk', '')} | {end} | {booked} | {C.entry_cost(e):.4f} | {e['reservation_usd']:.4f} |")
    t = C.totals(E)
    lines.append(f"| | **Total counted** | | | | | | **{t['total']:.4f}** of ${C.GLOBAL_CAP_USD:.0f} | |")
    by = ", ".join(f"{k} ${v:.4f}" for k, v in sorted(t.items()) if k != "total")
    out = "\n".join(lines) + f"\n\nBy phase: {by}.\n"
    print(out)
    (ROOT / "data" / "ledger_table.md").write_text(out)
    (ROOT / "data" / "ledger_totals.json").write_text(json.dumps(t, indent=1))


if __name__ == "__main__":
    main()
