"""Call one function of prep_app in an ephemeral app and save its JSON result.

Usage (from retrain/modal_app): python run_fn.py <function_name> <out.json>
Runs in this session's container; the function itself runs in Modal.
"""
import json, sys
import modal
import prep_app

fn_name, out = sys.argv[1], sys.argv[2]
with modal.enable_output():
    with prep_app.app.run():
        res = getattr(prep_app, fn_name).remote()
with open(out, "w") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k not in ("pip_freeze", "manifest")}, indent=1, default=str)[:6000])
