"""mdCATH state-pair dataset: sample (X_t, X_{t+delta}) for DeepJump-lite.

mdCATH HDF5 layout (confirmed against real files):
    <domain>/                       attrs: numProteinAtoms, numResidues
        element, resid, resname, z, chain   [numProteinAtoms]  (aligned to coords)
        pdb, psf                            scalar strings (FULL solvated system)
        <temperature in {320,348,379,413,450}>/
            <replica in 0..4>/      attrs: numFrames
                coords  [numFrames, numProteinAtoms, 3]  (Angstrom, 1 ns / frame)
                forces, dssp, rmsd, ...

Atom NAMES are not stored as a per-atom array; they live in the PSF string, which
describes the full 8266-atom system. We recover protein atom names by filtering the
PSF NATOM section to protein residues (segid P0 / standard AA resnames) -- this
yields exactly numProteinAtoms names in coords order (verified: resid/resname match).
"""

from __future__ import annotations

import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from ..atom_constants import RESIDUE_ALIASES, RESIDUE_TO_INDEX
from ..representation import (
    apply_layout,
    build_layout,
    canonicalize_symmetric,
    kabsch_align_futures,
    kabsch_align_target,
)

TEMPERATURES = (320, 348, 379, 413, 450)
_PROTEIN_RESNAMES = set(RESIDUE_TO_INDEX) | set(RESIDUE_ALIASES)


def _dec(x):
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def discover_domains(root: str | Path) -> list[Path]:
    root = Path(root).expanduser()
    files = sorted(glob.glob(str(root / "**" / "mdcath_dataset_*.h5"), recursive=True))
    return [Path(f) for f in files]


def parse_protein_atom_names(psf_str: str, expected_n: int) -> list[str]:
    """Extract protein atom names (in coords order) from a CHARMM PSF string.

    PSF NATOM columns: index segid resid resname name type charge mass ...
    Protein atoms are selected by standard-AA resname; the count must equal
    numProteinAtoms and the order matches the coords/element arrays.
    """
    lines = psf_str.splitlines()
    ni = next(i for i, l in enumerate(lines) if "!NATOM" in l)
    n_total = int(lines[ni].split()[0])
    names: list[str] = []
    for k in range(n_total):
        cols = lines[ni + 1 + k].split()
        if len(cols) < 5:
            continue
        resname = cols[3]
        if resname in _PROTEIN_RESNAMES:
            names.append(cols[4])
    if len(names) != expected_n:
        raise ValueError(
            f"PSF protein-atom count {len(names)} != numProteinAtoms {expected_n}"
        )
    return names


class _DomainHandle:
    """Lazily-opened h5 file + cached topology layout for one domain."""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem.replace("mdcath_dataset_", "")
        self._f: h5py.File | None = None
        self._layout = None

    @property
    def f(self) -> h5py.File:
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    @property
    def dom(self):
        return self.f[self.name]

    @property
    def layout(self):
        if self._layout is None:
            dom = self.dom
            n_atoms = int(dom.attrs["numProteinAtoms"])
            names = parse_protein_atom_names(_dec(dom["psf"][()]), n_atoms)
            resid = np.asarray(dom["resid"])
            resname = [_dec(v) for v in np.asarray(dom["resname"])]
            self._layout = build_layout(names, resid, resname)
        return self._layout

    def replicas(self, temperature: int, replicas: list[int]) -> list[tuple[int, int, int]]:
        """Return (temperature, replica, num_frames) for available replicas."""
        out = []
        tg = self.dom.get(str(temperature))
        if tg is None:
            return out
        for r in replicas:
            rg = tg.get(str(r))
            if rg is not None:
                out.append((temperature, r, int(rg.attrs["numFrames"])))
        return out

    def coords(self, temperature: int, replica: int, frame: int) -> np.ndarray:
        return self.dom[str(temperature)][str(replica)]["coords"][frame]

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


class MdcathPairDataset(Dataset):
    """Yields (X_t, X_{t+delta}) residue-level state pairs from mdCATH.

    Scale-safe (up to the full 5398-domain x 5-temp x 5-replica dataset):
      * files are NOT opened at __init__; a per-worker LRU cache opens h5 lazily
        inside __getitem__ (so DataLoader num_workers>0 is fork-safe);
      * a `manifest` (list of {"file", "trajectories":[{temp,replica,num_frames}]})
        supplies frame counts WITHOUT opening every file at startup -- build it once
        with scripts/build_manifest.py, then training init is instant;
      * the sample index is COMPACT (one entry per trajectory + cumulative counts),
        not one entry per frame, so memory stays ~MB even at ~10^8 samples.
    """

    def __init__(
        self,
        files: list[str | Path],
        temperatures: list[int] = (320,),
        replicas: list[int] = (0,),
        delta_frames=1,
        crop_length: int = 128,
        align: bool = True,
        unroll: int = 1,
        canon_symmetric: bool = False,
        manifest: list | None = None,
        max_open_files: int = 64,
        seed: int = 0,
    ):
        self.files = [Path(f) for f in files]
        self.temperatures = list(temperatures)
        self.replicas = list(replicas)
        # multi-scale delta (DeepJump trains one model over 1/10/100 ns jointly)
        self.deltas = [int(delta_frames)] if isinstance(delta_frames, int) else [int(d) for d in delta_frames]
        self.crop_length = int(crop_length)
        self.align = align
        self.unroll = int(unroll)  # number of future steps to return (>=1)
        self.canon_symmetric = canon_symmetric
        self.max_open_files = int(max_open_files)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._rng_worker_id: int | None = None
        self._cache: dict[int, _DomainHandle] = {}  # per-worker lazy handle cache
        self._order: list[int] = []  # LRU order of file indices

        traj_frames = self._trajectory_frames(manifest)  # {(file_idx,temp,rep): num_frames}
        self._traj: list[tuple[int, int, int, int]] = []  # (file_idx, temp, rep, delta)
        counts: list[int] = []
        for (fi, temp, rep), nf in traj_frames.items():
            for d in self.deltas:
                n_valid = nf - d * self.unroll  # number of valid start frames t
                if n_valid > 0:
                    self._traj.append((fi, temp, rep, d))
                    counts.append(n_valid)
        if not counts:
            raise RuntimeError("no (X_t, X_{t+delta}) pairs found; check temps/replicas/deltas")
        self._cum = np.cumsum(counts)
        self._total = int(self._cum[-1])

    def _trajectory_frames(self, manifest) -> dict:
        """Per-trajectory frame counts. From a manifest (no file opens) if given,
        else by scanning files once (fine for small/local N)."""
        out: dict[tuple[int, int, int], int] = {}
        if manifest is not None:
            by_name = {f.name: i for i, f in enumerate(self.files)}
            for entry in manifest:
                fi = by_name.get(Path(entry["file"]).name)
                if fi is None:
                    continue
                for tr in entry["trajectories"]:
                    if int(tr["temp"]) in self.temperatures and int(tr["replica"]) in self.replicas:
                        out[(fi, int(tr["temp"]), int(tr["replica"]))] = int(tr["num_frames"])
        else:
            for fi, f in enumerate(self.files):
                h = _DomainHandle(f)
                for temp in self.temperatures:
                    for (t_, r_, nf) in h.replicas(temp, self.replicas):
                        out[(fi, temp, r_)] = nf
                h.close()
        return out

    def __getstate__(self):
        # Never pickle open h5py handles (they are unpicklable and unsafe across
        # fork/spawn) -- workers reopen lazily. This makes num_workers>0 safe.
        state = self.__dict__.copy()
        state["_cache"] = {}
        state["_order"] = []
        state["_rng_worker_id"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = {}
        self._order = []

    def _get_handle(self, file_idx: int) -> _DomainHandle:
        h = self._cache.get(file_idx)
        if h is None:
            h = _DomainHandle(self.files[file_idx])
            self._cache[file_idx] = h
            self._order.append(file_idx)
            while len(self._order) > self.max_open_files:  # LRU eviction (ulimit-safe)
                old = self._order.pop(0)
                if old in self._cache:
                    self._cache.pop(old).close()
        return h

    def __len__(self) -> int:
        return self._total

    def stratified_indices(self, samples_per_trajectory: int = 1, seed: int = 0) -> list[int]:
        """Return fixed, trajectory-balanced sample indices.

        Sequential dataset indices exhaust one trajectory before moving to the
        next.  Taking the first N validation batches therefore evaluates almost
        exclusively on the first domain/temperature/replica.  This helper draws
        the same number of frames from every available trajectory instead.
        """
        if samples_per_trajectory < 1:
            raise ValueError("samples_per_trajectory must be >= 1")
        rng = np.random.default_rng(seed)
        out: list[int] = []
        start = 0
        for stop in self._cum:
            count = int(stop) - start
            take = min(samples_per_trajectory, count)
            offsets = rng.choice(count, size=take, replace=False)
            out.extend(start + int(offset) for offset in sorted(offsets.tolist()))
            start = int(stop)
        return out

    def _crop(self, n_res: int) -> slice:
        if n_res <= self.crop_length:
            return slice(0, n_res)
        worker = get_worker_info()
        if worker is not None and self._rng_worker_id != worker.id:
            # Dataset objects are copied into workers after construction. Without
            # a worker-specific seed, every worker inherits the same NumPy RNG
            # state and emits pairwise-identical crop starts.
            self.rng = np.random.default_rng(worker.seed)
            self._rng_worker_id = worker.id
        start = int(self.rng.integers(0, n_res - self.crop_length + 1))
        return slice(start, start + self.crop_length)

    def __getitem__(self, i: int) -> dict:
        j = int(np.searchsorted(self._cum, i, side="right"))
        base = int(self._cum[j - 1]) if j > 0 else 0
        t = int(i) - base
        fi, temp, rep, delta = self._traj[j]
        h = self._get_handle(fi)
        layout = h.layout

        c_t = torch.from_numpy(np.asarray(h.coords(temp, rep, t)))
        P_t, V_t = apply_layout(c_t, layout)
        futures = []
        for k in range(1, self.unroll + 1):
            c_k = torch.from_numpy(np.asarray(h.coords(temp, rep, t + k * delta)))
            futures.append(apply_layout(c_k, layout))

        if self.canon_symmetric:  # fix arbitrary symmetric-sidechain labelling
            V_t = canonicalize_symmetric(V_t, layout.res_index)
            futures = [(P, canonicalize_symmetric(V, layout.res_index)) for P, V in futures]

        if self.align:
            # Remove rigid-body tumbling: align every future onto X_t, center at origin.
            P_t, V_t, futures = kabsch_align_futures(P_t, V_t, futures)
        else:
            centroid = P_t.mean(dim=0, keepdim=True)
            P_t = P_t - centroid
            futures = [(P - centroid, V) for P, V in futures]

        res_index = torch.as_tensor(layout.res_index, dtype=torch.long)
        atom_mask = torch.as_tensor(layout.atom_mask, dtype=torch.bool)
        sl = self._crop(layout.num_residues)

        item = {
            "P_t": P_t[sl], "V_t": V_t[sl],
            "P_1": futures[0][0][sl], "V_1": futures[0][1][sl],
            "res_index": res_index[sl], "atom_mask": atom_mask[sl],
            "bond_mask": torch.as_tensor(layout.bond_mask[sl.start:sl.stop - 1], dtype=torch.bool),
            "delta_ns": torch.tensor(float(delta), dtype=torch.float32),  # 1 frame == 1 ns
            "temperature": torch.tensor(temp, dtype=torch.long),
            "replica": torch.tensor(rep, dtype=torch.long),
            "start_frame": torch.tensor(t, dtype=torch.long),
            "residue_start": torch.tensor(sl.start, dtype=torch.long),
            "n_res": (sl.stop - sl.start), "domain": h.name,
        }
        for k in range(2, self.unroll + 1):
            item[f"P_{k}"] = futures[k - 1][0][sl]
            item[f"V_{k}"] = futures[k - 1][1][sl]
        return item

    def close(self):
        for h in self._cache.values():
            h.close()
        self._cache.clear()
        self._order.clear()


def collate_pairs(batch: list[dict]) -> dict:
    """Pad variable-length residue dim to the batch max; build residue_mask."""
    B = len(batch)
    Nmax = max(b["n_res"] for b in batch)
    from ..atom_constants import MAX_HEAVY

    def zeros(*shape):
        return torch.zeros(*shape, dtype=torch.float32)

    # frame keys present in every item: P_t/V_t/P_1/V_1 plus any P_k/V_k (unroll)
    frame_keys = [k for k in batch[0] if k.startswith(("P_", "V_"))]
    out = {k: (zeros(B, Nmax, MAX_HEAVY, 3) if k.startswith("V_") else zeros(B, Nmax, 3))
           for k in frame_keys}
    res_index = torch.zeros(B, Nmax, dtype=torch.long)
    atom_mask = torch.zeros(B, Nmax, MAX_HEAVY, dtype=torch.bool)
    residue_mask = torch.zeros(B, Nmax, dtype=torch.bool)
    bond_mask = torch.zeros(B, max(Nmax - 1, 0), dtype=torch.bool)
    delta_ns = torch.zeros(B, dtype=torch.float32)
    temperature = torch.zeros(B, dtype=torch.long)
    replica = torch.zeros(B, dtype=torch.long)
    start_frame = torch.zeros(B, dtype=torch.long)
    residue_start = torch.zeros(B, dtype=torch.long)

    for b, item in enumerate(batch):
        n = item["n_res"]
        for k in frame_keys:
            out[k][b, :n] = item[k]
        res_index[b, :n] = item["res_index"]
        atom_mask[b, :n] = item["atom_mask"]
        residue_mask[b, :n] = True
        if n > 1:
            bond_mask[b, :n - 1] = item["bond_mask"]
        delta_ns[b] = item["delta_ns"]
        temperature[b] = item["temperature"]
        replica[b] = item["replica"]
        start_frame[b] = item["start_frame"]
        residue_start[b] = item["residue_start"]

    out.update({
        "res_index": res_index, "atom_mask": atom_mask,
        "residue_mask": residue_mask, "bond_mask": bond_mask, "delta_ns": delta_ns,
        "temperature": temperature, "replica": replica, "start_frame": start_frame,
        "residue_start": residue_start,
        "domains": [item["domain"] for item in batch],
    })
    return out
