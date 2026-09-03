# Compaction is not folding: the formal 500k model's extended-start endpoint
# against the real native structure.
#
# Verified against PyMOL 3.1.8. Note the object naming split_states produces:
# with prefix=final and state 601 the object is `final0601` -- state number, no
# underscore. Guessing that name wrong is why an earlier version of this script
# failed with "Invalid selection name".
#
#   /Applications/PyMOL.app/Contents/bin/pymol docs/pymol/compaction_vs_folding.pml
#
# Inputs are produced by:
#   scripts/export_rollout_pdb.py --initial-pdb <extended> --steps 600 --mode ode

reinitialize

load runs/visualization/folding_probe_20260902/1hw7A02_from_extended.pdb, rollout
load runs/visualization/folding_probe_20260902/1hw7A02_native.pdb, native

# Endpoint of the rollout, and two waypoints showing the contraction.
split_states rollout, 1, 1, prefix=step
split_states rollout, 76, 76, prefix=step
split_states rollout, 601, 601, prefix=step
delete rollout

hide everything
show cartoon, native step0601
color marine, native
color firebrick, step0601

# Superpose the model endpoint onto the native fold. On 1hw7A02 this lands at
# ~13 A: the two are the same size and a different shape.
align step0601, native

set cartoon_transparency, 0.2, native
set cartoon_fancy_helices, on
set cartoon_smooth_loops, on
set orthoscopic, on
set antialias, 2
bg_color white
set ray_opaque_background, 1
orient native

# Blue: real native structure, with helices and a packed core.
# Red: what the model produces from a fully extended chain. Native-scale radius
# of gyration (51.95 -> 10.96 A), no clashes, correct bond lengths -- and no
# secondary structure. See docs/FINDING_no_sequence_specific_folding_20260902.md.
