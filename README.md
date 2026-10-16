# EpNet-Si

EpNet-Si is an empirical pseudopotential method (EPM) for silicon in which an equivariant network predicts each atom's
plane-wave form factor from its local environment: a species potential plus monopole, dipole and quadrupole
corrections, mixed by a learned router over local atomic densities, together with an *s/p* nonlocal projector. The
band structure follows from a single diagonalization in a small plane-wave basis (8 Ry, centred on each **k**), with no
self-consistency. The model has 6,195 parameters and is trained through the eigensolver on PBE band energies of 77 cells.

This repository contains the code, the trained model, the 500 PBE reference band structures and the results behind
the paper ([arXiv:2610.14992](https://arxiv.org/abs/2610.14992)). File and folder names carry the internal version tag
`v2j`, which is the released EpNet-Si.

| Test group | Structures | Gapped: band / gap error | Pass / near miss / fail (gapped) | Metals: band error |
|---|---|---|---|---|
| Distortions of the trained structure types | 231 | 15 / 8 meV | 142 / 0 / 0 | 15 meV |
| Structure types absent from training | 192 | 38 / 49 meV | 120 / 20 / 25 | 21 meV |

## Layout

```
epnet/                    the package: plane-wave Hamiltonian, encoder, form factors, s/p projector, training utilities
models/
  epnet_si.pt             the trained EpNet-Si model
  precursor_init.pt       initial weights of the training run
  protocol_precursor.json, protocol_v2j.json   cells, stages and options of the two training phases
data/
  bands/<id>.npz          500 PBE band structures (lattice, positions, k path, eigenvalues)
  metal_grid/<id>.npz     uniform-grid PBE bands of the 145 metallic cells (116 test, 29 training)
  qe_inputs/<id>/         Quantum ESPRESSO pw.x inputs of every reference calculation
  pseudo/                 the PAW dataset used (Si.pbe-n-kjpaw_psl.1.0.0, pslibrary)
  manifest.json           all 500 cells: metadata, role (train / test), test group and source
  manifest_precursor.json, manifest_v2j.json   training manifests of the two phases (training split: diag_fit)
results/
  test_groups/            per-cell scores on the 423 test structures: EpNet-Si (v2j), the four leave-one-family-out
                          models (v2j_lofo_*) and the one-fixed-potential baseline (v2j_fixed)
  training/               training histories and full-path scores of the precursor and final phases
  ablation/               component ablation of the trained model
  derived_physics/        effective masses, gap pressure coefficients, deformation potentials, supercell consistency,
                          conduction-edge and bond-charge analyses
  lofo_protocols/         training protocols of the leave-one-family-out and fixed-potential runs
  figure_data/            summaries behind the figures
scripts/
  predict_bands.py        bands of a structure with EpNet-Si, compared with PBE when available
  evaluate_test_groups.py score a checkpoint on the 423 test structures (seen-distorted and unseen groups)
  train_epnet_si.py       retrain EpNet-Si with the paper's recipe (also --drop-family F, --fixed)
  train_si_pbe_dataset.py, rescore_si_pbe_fit.py   the training and full-path scoring engines
  scoring.py              metrics and pass / near-miss / fail grades
  check_supercell_consistency.py   primitive vs supercell bands (k-centred basis)
tests/
```

## Installation

```
pip install -e '.[test]'    # numpy, scipy, torch >= 2.0, e3nn, spglib; pytest, ase, pymatgen for the tests
pytest                      # 116 tests
```

## Use

```
python scripts/predict_bands.py si-clathrate_i-parent          # any id of data/manifest.json, or an NPZ with lattice,
                                                                # fractional_positions and kpoints
python scripts/evaluate_test_groups.py --output eval.json       # all 423 test structures, both groups
python scripts/evaluate_test_groups.py --ids si-diamond-hydro_m0.02 --compare   # compare with the stored scores
```

## Training

```
python scripts/train_epnet_si.py --out runs/v2j                 # precursor (66 cells, 3 stages) then final (77 cells, 2 stages)
python scripts/train_epnet_si.py --out runs/lofo_beta_tin --drop-family beta_tin
python scripts/train_epnet_si.py --out runs/fixed --fixed
python scripts/train_epnet_si.py --smoke                        # one step on two cells, CPU
```

The precursor takes about 13 h and the final phase about 7 h on one NVIDIA L40S; diagonalizing the largest cage cells
dominates the step time.

## Reference data

All reference band structures were computed with Quantum ESPRESSO 7.6, PBE, the PAW dataset in `data/pseudo/`,
40 / 320 Ry cutoffs, a 4x4x4 SCF grid and SeeK-path band paths (inputs in `data/qe_inputs/`). Cells whose path gap is
at most 0.02 eV were relabelled as metals with a Marzari-Vanderbilt smeared SCF and a uniform-grid non-SCF calculation
(`data/metal_grid/`, inputs under `qe_inputs/<id>/metal_relabel/`). Structures come from catalog distortions of
twelve Si phases, OQMD, SACADA (carbon nets converted to Si), the low-density frameworks of Amsler et al., AFLOW and
new stacking polytypes.

## Citation

If you use EpNet-Si, its data or its results, please cite:

Q. Zhu, *EpNet-Si: Atomic Form Factors Learned from Band Structures*, arXiv:2610.14992 (2026).
<https://arxiv.org/abs/2610.14992>

```bibtex
@misc{zhu2026epnetsi,
  title         = {EpNet-Si: Atomic Form Factors Learned from Band Structures},
  author        = {Qiang Zhu},
  year          = {2026},
  eprint        = {2610.14992},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mtrl-sci},
  url           = {https://arxiv.org/abs/2610.14992}
}
```
