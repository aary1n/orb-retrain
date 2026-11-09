# Retraining Orb-v3 from scratch on Modal

This repository trains copies of the Orb-v3 machine-learned force field from random weights on [Modal](https://modal.com) GPUs, with a spend ledger that refuses any job that would break a dollar cap.

It is the training pipeline of a research project on what universal force fields learn inside. Training copies from scratch lets controlled changes to the training data be tested against the released model. The analysis code, the data lists and the results are not here; they will be released with the paper.

## What it has run

In November 2026 it trained two full-data copies (seeds 0 and 1) on the 551,578 AIMD frames of the public OMat24 1M subset:

| Setting | Value |
| --- | --- |
| Architecture | Orb-v3 conservative (`orb_v3_conservative_architecture`, latent 256, MLP 1024, radius 6.0 Å, 120 neighbours), random start |
| Recipe | orb-models 0.5.5 `finetune.py` recipe: Adam, OneCycle learning rate, batch 100, random rotations |
| Length | 62,000 steps (about 11 passes over the frames) |
| Hardware | one H100 80 GB, 8 CPU cores, 32 GiB memory |
| Time and cost | about 4.4 h and $20-25 per copy |
| Checkpoints | state dicts at 0, 25, 50, 75 and 100% of training, each followed by a full validation and clean-test evaluation |

## What is here

| Path | What it does |
| --- | --- |
| `modal_app/common.py` | Prices and the spend ledger. Every GPU attempt reserves its worst-case cost on a Modal Volume before it starts, sends heartbeats, and books its real cost when it ends. An attempt is refused if the cap (`GLOBAL_CAP_USD`), a phase cap or the end date would be passed. |
| `modal_app/prep_app.py`, `data_jobs.py` | Data jobs on CPU: download the OMat24 1M subset from Meta's public URL, check its sha256, unpack it, extract per-row metadata. Also the image test and the GPU timing runs. |
| `modal_app/train_app.py`, `train_job.py` | The deployed training app. Each chunk trains for up to 6 h, saves its state, and spawns the next chunk, so pre-emptions resume from the last checkpoint. |
| `modal_app/dataset_lmdb.py` | An LMDB dataset that mirrors orb-models 0.5.5 `ase_sqlite_dataset.py`. |
| `modal_app/ref_eval.py` | Evaluates the released `orb-v3-conservative-inf-omat` on the same validation and test rows, as the yardstick. |
| `modal_app/plan_constants.py` | GPU type, cores, memory, chunk length, restarts and pinned package versions. |
| `launch/` | Start, resume and check runs (`launch.py`), a smoke test (`smoke.sh`), the row lists (`make_lists.py`), the ledger table and training-curve plots. |
| `removal/test_overlap_parent_ids.txt` | The 34 starting crystals that appear in both the training and the test split; their rows are dropped from the clean test set. |
| `plan/`, `timing/` | The ledger simulator, run sizing, and measured throughput and memory on H100, A100 40 GB and L40S. |
| `tests/test_ledger.py` | Unit tests of the ledger; they need no Modal account. |

## How to run it

1. Install `modal` and log in with `modal setup`. Create two Volumes named `orb-retrain` and `orb-retrain-ledger`; the code opens them as version-2 Volumes.
2. Prepare the data, from `modal_app/`:
   ```
   python run_fn.py data_download download.json
   python run_fn.py data_metadata metadata.json
   ```
3. Build the row lists, from the repository root:
   ```
   modal volume get orb-retrain meta data/
   python launch/make_lists.py
   modal volume put -f orb-retrain lists lists
   ```
4. Deploy the training app: `cd modal_app && modal deploy train_app.py`.
5. Run the smoke test (`launch/smoke.sh`), then start a copy: `./launch_run.sh full <seed>`. Check it with `python launch/launch.py status <run_id>`.

Before any spend, set `GLOBAL_CAP_USD` in `modal_app/common.py` and `END_DATE_UTC` in `modal_app/plan_constants.py` to your own limits. Some docstrings still give paths relative to a parent `retrain/` folder and refer to an internal plan document that is not included.

## Tests

The ledger tests run without Modal:

```
cd tests && python -m pytest test_ledger.py
```

## Credits and licence

- Model code and training recipe: [orb-models](https://github.com/orbital-materials/orb-models) by Orbital Materials, Apache License 2.0. `modal_app/dataset_lmdb.py` follows its `ase_sqlite_dataset.py`.
- Data: [OMat24](https://huggingface.co/datasets/facebook/OMAT24) by Meta FAIR, CC BY 4.0. It is downloaded at run time and is not in this repository.
- This repository is released under the Apache License 2.0 (see `LICENSE` and `NOTICE`).
