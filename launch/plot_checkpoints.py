"""Figure: full val (solid) and clean test (dashed) errors at the 25/50/75/100 % checkpoints, with the released Orb-v3 omat model as thin lines.
Usage: python plot_checkpoints.py <run_id>   (reads runs_local/<run_id>/val_metrics.csv, which must hold the ref_ rows)"""
import sys
from pathlib import Path
import matplotlib, pandas as pd
matplotlib.use("Agg"); import matplotlib.pyplot as plt
run = sys.argv[1]; d = Path(__file__).resolve().parents[1] / "runs_local" / run
v = pd.read_csv(d / "val_metrics.csv").drop_duplicates(["step", "set", "group"], keep="last")
ref = {(r.set.replace("ref_orbv3_inf_omat_", ""), r.group): r for r in v[v.set.str.startswith("ref_")].itertuples()}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for k, (col, lab) in enumerate([("forces_mae_meV_per_A", "forces MAE (meV/Å)"), ("energy_mae_meV_per_atom", "energy MAE (meV/atom)"), ("stress_mae_GPa", "stress MAE (GPa)")]):
    a = ax[k]
    for s, ls in (("val", "-"), ("test_clean", "--")):
        for g, c in (("aimd", "C0"), ("rattled", "C1"), ("all", "C2")):
            q = v[(v.set == s) & (v.group == g) & (v.step > 0)]
            a.plot(q.step, q[col], ls, marker="o", ms=4, color=c, label=f"{s} {g}")
            r = ref.get((s, g))
            if r is not None:
                a.axhline(getattr(r, col), ls=ls, color=c, lw=0.8, alpha=0.6)
    a.set_yscale("log"); a.set_xlabel("step"); a.set_title(lab + "; thin lines = released Orb-v3 omat"); a.set_xticks([15500, 31000, 46500, 62000])
ax[0].legend(fontsize=7, ncol=2)
fig.suptitle(f"{run}: full val (solid) and clean test (dashed) at the 25/50/75/100 % checkpoints", y=1.0)
fig.tight_layout(); fig.savefig(d / "checkpoints_vs_reference.png", dpi=130); print("saved", d / "checkpoints_vs_reference.png")
