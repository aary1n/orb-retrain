"""Constants for the deployed training app. They must match PLAN.md.

STATUS: plan A approved by Aaryan on 2026-11-02 (H100, batch 100, 62,000 steps,
one run at a time, cap $200, end date 2026-11-09 00:00 UTC).
"""

PIP = [
    "torch==2.6.0",
    "orb-models==0.5.5",
    "ase==3.29.0",
    "numpy==2.4.6",
    "scipy==1.17.1",
    "dm-tree==0.1.8",
    "cached-path==1.8.10",
    "lmdb==3.0.0",
    "tqdm==4.70.1",
]
GPU = "H100!"  # exact type, no fallback list
CPU = 8.0  # physical cores (request = limit)
MEM_MIB = 32768  # request = limit
TIMEOUT_S = 6 * 3600 + 15 * 60  # backstop; the code ends each chunk at 6 h 00
CHUNK_WALL_S = 6 * 3600
MAX_AT_ONCE = 1  # runs at once (max_containers)
MAX_RESTARTS = 5
END_DATE_UTC = "2026-11-09T00:00:00Z"  # no chunk may start after this
