"""Wait for a spawned train_chunk call and print its result as JSON (runs wherever the Modal client is logged in)."""
import json, sys, modal
call = modal.FunctionCall.from_id(sys.argv[1])
try:
    res = call.get(timeout=float(sys.argv[2]) if len(sys.argv) > 2 else None)
    print("CALL_RESULT", json.dumps(res, default=str))
except Exception as exc:  # noqa: BLE001
    print("CALL_ERROR", repr(exc)[:2000])
    sys.exit(1)
