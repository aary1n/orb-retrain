"""A torch Dataset over OMat24 .aselmdb files.

It mirrors orb-models 0.5.5 `orb_models/dataset/ase_sqlite_dataset.py`
(AseSqliteDataset.__getitem__) line for line. The only difference is where the
row comes from: an LMDB value (zlib-compressed ASE JSON) instead of an ASE
sqlite row. Targets are read by orb-models' own `target_config.extract(row,...)`
(row.energy, row.forces, row.stress), and the graph is built by
`atomic_system.ase_atoms_to_atom_graphs(..., edge_method="knn_scipy", wrap=True)`,
exactly as in AseSqliteDataset.
"""

from __future__ import annotations

import os
import zlib
from typing import Callable, List, Optional, Sequence, Tuple

import ase
import ase.db.row
import lmdb
import numpy as np
import torch
from ase.io.jsonio import decode

from orb_models.dataset.base_datasets import AtomsDataset
from orb_models.forcefield import atomic_system, property_definitions
from orb_models.forcefield.base import AtomGraphs


_ENVS: dict = {}  # (pid, path) -> env. LMDB forbids opening one file twice in a process, so all datasets share.


def open_env(path: str):
    k = (os.getpid(), os.path.abspath(path))
    env = _ENVS.get(k)
    if env is None:
        env = _ENVS[k] = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
                                   max_readers=256)
    return env


def decimal_keys(path: str) -> List[bytes]:
    """All decimal keys of one .aselmdb, sorted numerically (other keys are metadata)."""
    env = open_env(path)
    with env.begin() as txn:
        keys = [k for k in txn.cursor().iternext(keys=True, values=False) if k.isdigit()]
    keys.sort(key=int)
    return keys


def decode_value(raw: bytes) -> dict:
    return decode(zlib.decompress(raw).decode())


class AseLmdbDataset(AtomsDataset):
    """Index = list of (file_number, key) pairs, so any subset can be expressed."""

    def __init__(
        self,
        name: str,
        paths: Sequence[str],
        index: Sequence[Tuple[int, bytes]],
        system_config: atomic_system.SystemConfig,
        target_config: Optional[property_definitions.PropertyConfig] = None,
        augmentations: Optional[List[Callable[[ase.Atoms], None]]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(name=name, system_config=system_config, augmentations=augmentations)
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.paths = list(paths)
        self.index = list(index)
        self.target_config = target_config if target_config is not None else property_definitions.PropertyConfig()
        self.constraints = []  # type: ignore[var-annotated]

    def _txn_get(self, fi: int, key: bytes) -> bytes:
        env = open_env(self.paths[fi])  # cached per process; a forked worker opens its own
        with env.begin() as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(f"{self.paths[fi]} has no key {key!r}")
        return raw

    def get_row(self, idx: int) -> ase.db.row.AtomsRow:
        fi, key = self.index[idx]
        return ase.db.row.AtomsRow(decode_value(self._txn_get(fi, key)))

    def __getitem__(self, idx: int) -> AtomGraphs:
        row = self.get_row(idx)
        atoms = row.toatoms()

        # --- from here on identical to AseSqliteDataset.__getitem__ (orb-models 0.5.5) ---
        atoms.info = {}
        atoms.info.update(self.target_config.extract(row, self.name, "targets"))

        for augmentation in self.augmentations:
            augmentation(atoms)

        atoms.set_constraint(None)
        for constraint_fn in self.constraints:
            constraint_fn(atoms, row.data, self.name)

        return atomic_system.ase_atoms_to_atom_graphs(
            atoms=atoms,
            system_config=self.system_config,
            edge_method="knn_scipy",
            wrap=True,
            system_id=idx,
            output_dtype=self.dtype,
            graph_construction_dtype=self.dtype,
        )

    def get_atom(self, idx: int) -> ase.Atoms:
        return self.get_row(idx).toatoms()

    def get_atom_and_metadata(self, idx: int):
        row = self.get_row(idx)
        return row.toatoms(), row.data

    def get_idx_to_natoms(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return f"AseLmdbDataset({self.name=}, n_files={len(self.paths)}, n={len(self.index)})"
