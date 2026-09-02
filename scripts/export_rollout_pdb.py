#!/usr/bin/env python
"""Export a DeepJump rollout from an mdCATH frame as a multi-state PDB."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from deepjump.atom_constants import HEAVY_ATOM_ORDER, canonical_resname
from deepjump.config import ModelConfig
from deepjump.data.mdcath import _DomainHandle
from deepjump.model import DeepJumpLite
from deepjump.representation import apply_model_layout, build_layout
from deepjump.sampling import rollout
from deepjump.utils import resolve_device


@dataclass(frozen=True)
class ResidueRecord:
    resid: int
    resname: str
    chain: str


def _decode(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode(errors="replace").strip()
    return str(value).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _residue_records(handle: _DomainHandle) -> list[ResidueRecord]:
    dom = handle.dom
    resids = np.asarray(dom["resid"]).reshape(-1)
    resnames = np.asarray(dom["resname"]).reshape(-1)
    chains = np.asarray(dom["chain"]).reshape(-1)
    _, first_indices = np.unique(resids, return_index=True)
    records = []
    for index in np.sort(first_indices):
        chain = _decode(chains[index])[:1] or "A"
        records.append(
            ResidueRecord(
                resid=int(resids[index]),
                resname=canonical_resname(_decode(resnames[index])),
                chain=chain,
            )
        )
    return records


def read_pdb_input(path: Path):
    """Read the first PDB model and return coordinates, layout, and residues."""
    atom_names: list[str] = []
    resids: list[int] = []
    resnames: list[str] = []
    coordinates: list[list[float]] = []
    residues: list[ResidueRecord] = []
    residue_ordinals: dict[tuple[str, int, str], int] = {}
    seen_atoms: set[tuple[int, str]] = set()
    saw_model = False

    for line in path.read_text().splitlines():
        record = line[:6].strip()
        if record == "MODEL":
            if saw_model:
                break
            saw_model = True
            continue
        if record == "ENDMDL" and saw_model:
            break
        if record not in {"ATOM", "HETATM"}:
            continue
        altloc = line[16:17]
        if altloc not in {"", " ", "A"}:
            continue
        atom_name = line[12:16].strip()
        resname = canonical_resname(line[17:20])
        if resname == "UNK" or atom_name.startswith("H"):
            continue
        chain = line[21:22].strip() or "A"
        resid = int(line[22:26])
        insertion = line[26:27].strip()
        key = (chain, resid, insertion)
        if key not in residue_ordinals:
            residue_ordinals[key] = len(residues)
            residues.append(ResidueRecord(resid=resid, resname=resname, chain=chain))
        ordinal = residue_ordinals[key]
        atom_key = (ordinal, atom_name)
        if atom_key in seen_atoms:
            continue
        seen_atoms.add(atom_key)
        atom_names.append(atom_name)
        resids.append(ordinal)
        resnames.append(resname)
        coordinates.append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )

    if not coordinates:
        raise ValueError(f"no supported protein atoms found in {path}")
    coords = np.asarray(coordinates, dtype=np.float32)
    layout = build_layout(atom_names, np.asarray(resids), resnames)
    return coords, layout, residues


def _element(atom_name: str) -> str:
    return next((char for char in atom_name if char.isalpha()), "C").upper()


def write_multistate_pdb(
    path: Path,
    trajectory: list[tuple[torch.Tensor, torch.Tensor]],
    residues: list[ResidueRecord],
    atom_mask: np.ndarray,
) -> None:
    """Write [N,3]/[N,13,3] DeepJump states as one MODEL per PDB state."""
    lines: list[str] = []
    for state, (positions, offsets) in enumerate(trajectory, start=1):
        p = positions.detach().cpu().numpy()
        v = offsets.detach().cpu().numpy()
        lines.append(f"MODEL     {state:4d}")
        serial = 1
        for index, residue in enumerate(residues):
            atoms = [("CA", p[index])]
            for slot, atom_name in enumerate(
                HEAVY_ATOM_ORDER.get(residue.resname, ())
            ):
                if atom_mask[index, slot]:
                    atoms.append((atom_name, p[index] + v[index, slot]))
            for atom_name, xyz in atoms:
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:>4s} {residue.resname:>3s} "
                    f"{residue.chain:1s}{residue.resid:4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"  1.00  0.00          {_element(atom_name):>2s}"
                )
                serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _write_pymol_scripts(pdb_path: Path, n_states: int, fps: int) -> tuple[Path, Path]:
    interactive = pdb_path.with_suffix(".pml")
    render = pdb_path.with_name(f"{pdb_path.stem}_render.pml")
    frame_dir = pdb_path.with_name(f"{pdb_path.stem}_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)

    common = f"""reinitialize
load {pdb_path.resolve()}, rollout
hide everything, all
# Remove rigid-body drift for display only; the exported PDB remains unchanged.
intra_fit rollout and name CA, 1
dss rollout, state=1
show cartoon, rollout
color skyblue, rollout
show sticks, rollout and (name CA or sidechain) and not hydro
color gray70, rollout and elem C
color marine, rollout and elem N
color red, rollout and elem O
color yelloworange, rollout and elem S
color skyblue, rollout and name CA
set stick_radius, 0.13
set cartoon_fancy_helices, on
set cartoon_smooth_loops, on
set cartoon_side_chain_helper, on
set cartoon_transparency, 0.08
set orthoscopic, on
set antialias, 2
bg_color white
set all_states, off
set movie_fps, {fps}
mset 1 -{n_states}
frame 1
orient rollout, state=1
"""
    interactive.write_text(common + "set movie_loop, on\nmplay\n")
    render.write_text(
        common
        + "set ray_trace_frames, off\n"
        + "set cache_frames, off\n"
        + "viewport 1280, 720\n"
        + f"mpng {frame_dir.resolve()}/frame_, 1, {n_states}\n"
        + "quit\n"
    )
    return interactive, render


def _load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_cfg = payload["cfg"]["model"]
    data_cfg = payload["cfg"]["data"]
    model = DeepJumpLite(
        ModelConfig(**model_cfg),
        noise_sigma=data_cfg["noise_sigma"],
        predict_heavy=model_cfg["predict_heavy"],
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


def _safe_prefix(
    trajectory: list[tuple[torch.Tensor, torch.Tensor]], max_abs_coordinate: float
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], str | None]:
    safe = []
    for state, (positions, offsets) in enumerate(trajectory):
        atom_positions = positions.unsqueeze(-2) + offsets
        tensors = (positions, offsets, atom_positions)
        if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
            return safe, f"state {state} was non-finite"
        maximum = max(float(tensor.abs().max().item()) for tensor in tensors)
        if maximum > max_abs_coordinate:
            return safe, (
                f"state {state} exceeded --max-abs-coordinate "
                f"({maximum:.3f} > {max_abs_coordinate:.3f} A)"
            )
        safe.append((positions, offsets))
    return safe, None


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument(
        "--initial-pdb",
        type=Path,
        help="optional first-model PDB coordinates; input H5 still supplies training topology",
    )
    parser.add_argument("--temperature", type=int, default=320)
    parser.add_argument("--replica", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--delta-ns", type=float, default=1.0)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--mode", choices=("ode", "mean"), default="ode")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop-length", type=int, default=0)
    parser.add_argument("--integrator", choices=("euler", "heun"), default="euler")
    parser.add_argument("--tau-max", type=float, default=1.0)
    parser.add_argument("--terminal-denoise", action="store_true")
    parser.add_argument("--drift-anchor", choices=("state", "conditioner"), default="state")
    parser.add_argument(
        "--project-v-atom-mask", action=argparse.BooleanOptionalAction, default=True,
        help="re-mask V onto atom_mask at every sampler transition (default on; --no-project-v-atom-mask reproduces the pre-fix sampler, which writes into V's structural zero padding and compounds into broken geometry)",
    )
    parser.add_argument("--max-abs-coordinate", type=float, default=10000.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.steps < 0 or args.ode_steps < 1:
        parser.error("--steps must be >= 0 and --ode-steps must be >= 1")
    if args.delta_ns <= 0 or args.max_abs_coordinate <= 0 or args.fps < 1:
        parser.error("--delta-ns, --max-abs-coordinate, and --fps must be positive")

    checkpoint = args.ckpt.expanduser().resolve()
    input_h5 = args.input_h5.expanduser().resolve()
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, payload = _load_model(checkpoint, device)
    data_cfg = payload["cfg"]["data"]

    handle = _DomainHandle(input_h5)
    try:
        layout = handle.layout
        available = handle.replicas(args.temperature, [args.replica])
        if not available:
            raise ValueError(
                f"temperature={args.temperature}, replica={args.replica} is unavailable"
            )
        n_frames = available[0][2]
        if not 0 <= args.frame < n_frames:
            raise ValueError(f"--frame must be in [0, {n_frames - 1}]")

        records = _residue_records(handle)
        initial_pdb = None
        initialization = "mdcath_frame"
        if args.initial_pdb is not None:
            initial_pdb = args.initial_pdb.expanduser().resolve()
            coordinates, pdb_layout, records = read_pdb_input(initial_pdb)
            if pdb_layout.num_residues != layout.num_residues:
                raise ValueError(
                    "initial PDB residue count does not match input H5 topology"
                )
            if not np.array_equal(pdb_layout.res_index, layout.res_index):
                raise ValueError(
                    "initial PDB sequence does not match input H5 topology"
                )
            missing = layout.atom_mask & ~pdb_layout.atom_mask
            if missing.any():
                raise ValueError(
                    "initial PDB lacks heavy atoms required by input H5 topology"
                )
            coordinate_layout = pdb_layout
            initialization = "pdb_first_model"
        else:
            coordinates = np.asarray(
                handle.coords(args.temperature, args.replica, args.frame)
            )
            coordinate_layout = layout
        positions, offsets = apply_model_layout(
            coordinates,
            coordinate_layout,
            canon_symmetric=bool(data_cfg.get("canon_symmetric", False)),
        )
        offsets = offsets * torch.as_tensor(layout.atom_mask).unsqueeze(-1)
        positions = positions - positions.mean(0, keepdim=True)

        crop_length = args.crop_length or int(data_cfg["crop_length"])
        crop_length = min(crop_length, layout.num_residues)
        start = max(0, (layout.num_residues - crop_length) // 2)
        stop = start + crop_length
        residue_slice = slice(start, stop)
        positions = positions[residue_slice]
        offsets = offsets[residue_slice]
        atom_mask = np.asarray(layout.atom_mask[residue_slice])
        bond_mask = np.asarray(layout.bond_mask[start : max(start, stop - 1)])
        residues = records[residue_slice]

        init = {
            "P_t": positions[None].to(device),
            "V_t": offsets[None].to(device),
            "res_index": torch.as_tensor(
                layout.res_index[residue_slice], device=device
            )[None],
            "delta_ns": torch.tensor([args.delta_ns], device=device),
            "residue_mask": torch.ones(
                1, crop_length, dtype=torch.bool, device=device
            ),
            "atom_mask": torch.as_tensor(atom_mask, device=device)[None],
            "bond_mask": torch.as_tensor(bond_mask, device=device)[None],
        }
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        trajectory, accepts = rollout(
            model,
            init,
            n_steps=args.steps,
            ode_steps=args.ode_steps,
            mode=args.mode,
            gate=args.gate,
            generator=generator,
            sample_kwargs={
                "integrator": args.integrator,
                "tau_max": args.tau_max,
                "terminal_denoise": args.terminal_denoise,
                "drift_anchor": args.drift_anchor,
                "project_v_atom_mask": args.project_v_atom_mask,
            },
        )
        unbatched = [(p[0].cpu(), v[0].cpu()) for p, v in trajectory]
        safe, stop_reason = _safe_prefix(unbatched, args.max_abs_coordinate)
        if not safe:
            raise RuntimeError(f"initial state could not be exported: {stop_reason}")
        write_multistate_pdb(output, safe, residues, atom_mask)
        interactive_pml, render_pml = _write_pymol_scripts(
            output, len(safe), args.fps
        )

        acceptance_rate = None
        if accepts:
            acceptance_rate = float(torch.stack(accepts).float().mean().item())
        metadata = {
            "classification": "DeepJump model rollout; not a validated physical MD trajectory",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_step": int(payload.get("step", -1)),
            "input_h5": str(input_h5),
            "initialization": initialization,
            "initial_pdb": str(initial_pdb) if initial_pdb is not None else None,
            "initial_pdb_sha256": _sha256(initial_pdb) if initial_pdb is not None else None,
            "domain": handle.name,
            "temperature_k": args.temperature,
            "replica": args.replica,
            "initial_frame": args.frame,
            "initial_coordinates_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
            "residue_slice": [start, stop],
            "num_residues": crop_length,
            "delta_ns": args.delta_ns,
            "requested_rollout_steps": args.steps,
            "saved_rollout_steps": len(safe) - 1,
            "saved_states_including_initial": len(safe),
            "stop_reason": stop_reason,
            "mode": args.mode,
            "ode_steps": args.ode_steps,
            "seed": args.seed,
            "gate": args.gate,
            "acceptance_rate": acceptance_rate,
            "integrator": args.integrator,
            "tau_max": args.tau_max,
            "terminal_denoise": args.terminal_denoise,
            "drift_anchor": args.drift_anchor,
            "project_v_atom_mask": args.project_v_atom_mask,
            "pdb": str(output),
            "interactive_pml": str(interactive_pml),
            "render_pml": str(render_pml),
        }
        metadata_path = output.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
    finally:
        handle.close()


if __name__ == "__main__":
    main()
