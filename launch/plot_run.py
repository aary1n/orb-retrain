"""Download a run's small files from the Volume and plot its loss curve (runs in this session's container).

  python plot_run.py <run_id>  ->  runs_local/<run_id>/{loss_per_step.csv,train_log.csv,val_metrics.csv,...}
                                   runs_local/<run_id>/loss_curve.png and a printed summary
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SMALL = ["loss_per_step.csv", "train_log.csv", "val_metrics.csv", "status.json", "run_record.json", "config.json",
         "arch.json", "oom_skips.csv", "attempts.json", "loss_keys.json", "STOP"]


def pull(run_id):
    d = ROOT / "runs_local" / run_id
    d.mkdir(parents=True, exist_ok=True)
    for f in SMALL:
        subprocess.run(["modal", "volume", "get", "--force", "orb-retrain", f"runs/{run_id}/{f}", str(d / f)],
                       capture_output=True, text=True)
    return d


def main(run_id):
    d = pull(run_id)
    # After a pre-emption the restarted attempt re-logs the steps since the last checkpoint: keep the later rows.
    lp = pd.read_csv(d / "loss_per_step.csv").drop_duplicates("step", keep="last").sort_values("step")
    tl = pd.read_csv(d / "train_log.csv").drop_duplicates("step", keep="last").sort_values("step")
    st = json.loads((d / "status.json").read_text())
    rec = json.loads((d / "run_record.json").read_text())
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a = ax[0, 0]
    a.plot(lp.step, lp.loss, lw=0.4, alpha=0.5, label="per step")
    w = max(1, len(lp) // 40)
    a.plot(lp.step, lp.loss.rolling(w, center=True).mean(), lw=1.5, label=f"rolling mean ({w} steps)")
    a.set_xlabel("step")
    a.set_ylabel("training loss (orb-models total loss)")
    a.set_yscale("log")
    a.legend()
    a.set_title(f"{run_id}: loss, steps {int(lp.step.min())}–{int(lp.step.max())}")
    a = ax[0, 1]
    a.plot(tl.step, tl.lr)
    a.set_xlabel("step")
    a.set_ylabel("learning rate")
    a.set_title("OneCycle learning rate")
    a = ax[1, 0]
    a.plot(tl.step, tl["forces_mae_raw_eV_per_A"] * 1000, label="forces MAE (train batches)")
    a.set_ylabel("meV/Å")
    a.set_yscale("log")
    a.set_xlabel("step")
    a2 = a.twinx()
    a2.plot(tl.step, tl["energy_mae_raw_eV_per_structure"] / tl["atoms_per_step"] * 100 * 1000, color="C1",
            label="energy MAE per atom, approx. (train batches)")
    a2.set_ylabel("meV/atom (approx.: batch MAE / mean atoms)")
    a2.set_yscale("log")
    a.legend(loc="upper left")
    a2.legend(loc="upper right")
    a.set_title("raw training errors, 100-step means")
    a = ax[1, 1]
    vm = pd.read_csv(d / "val_metrics.csv") if (d / "val_metrics.csv").exists() else None
    if vm is not None:
        vm = vm.drop_duplicates(["step", "set", "group"], keep="last").sort_values("step")
    if vm is not None:
        for s, g in vm.groupby(["set", "group"]):
            if s[1] == "all":
                a.plot(g.step, g.forces_mae_meV_per_A, "o-", label=f"{s[0]} forces MAE")
        a.set_yscale("log")
        a.set_ylabel("meV/Å")
        a.set_xlabel("step")
        a.legend()
        a.set_title("validation forces MAE")
    fig.tight_layout()
    fig.savefig(d / "loss_curve.png", dpi=130)
    n = len(lp)
    q = [lp.loss.iloc[int(n * i / 10):int(n * (i + 1) / 10)].mean() for i in range(10)]
    summary = {
        "run": run_id, "state": st.get("state"), "step": st.get("step"), "oom_skips": st.get("oom_skips"),
        "start_fingerprint": rec.get("start_fingerprint", {}).get("tensor_sha256"),
        "loss_mean_by_tenth": [round(x, 4) for x in q],
        "loss_first100_mean": round(lp.loss.iloc[:100].mean(), 4), "loss_last100_mean": round(lp.loss.iloc[-100:].mean(), 4),
        "lr_first_logged": float(tl.lr.iloc[0]), "lr_last_logged": float(tl.lr.iloc[-1]),
        "steps_per_s_mean": float(tl.steps_per_s.mean()), "structures_per_s_mean": float(tl.structures_per_s.mean()),
        "train_forces_mae_meV_per_A_first_last": [round(1000 * tl["forces_mae_raw_eV_per_A"].iloc[0], 1),
                                                   round(1000 * tl["forces_mae_raw_eV_per_A"].iloc[-1], 1)],
        "val_rows": vm[vm.group == "all"][["step", "set", "n", "energy_mae_meV_per_atom", "forces_mae_meV_per_A",
                                           "stress_mae_GPa"]].round(2).to_dict(orient="records") if vm is not None else None,
    }
    print(json.dumps(summary, indent=1))
    (d / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
