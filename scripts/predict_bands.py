#!/usr/bin/env python3
"""Band structure of a Si structure from EpNet-Si, compared with its PBE reference when one is available.

Input: a cell id of data/manifest.json (structure, k path and PBE bands) or any NPZ with 'lattice',
'fractional_positions' and 'kpoints' (fractional, reciprocal-lattice units); n_occupied = 2 per Si atom.
Output: eigenvalues (eV) on the k path, VBM-aligned, as an NPZ; with a PBE reference, the band MAE in a +-2 eV window
around midgap and the gap error are printed.
Usage: predict_bands.py ID|FILE.npz [--checkpoint models/epnet_si.pt] [--bands 8] [--output pred.npz]
"""
import argparse,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from epnet.data.structures import Structure
from epnet.models.si_epm import load_gated_si_epm
from scoring import predict_path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('cell');ap.add_argument('--checkpoint',default=str(ROOT/'models/epnet_si.pt'))
    ap.add_argument('--bands',type=int,default=8,help='bands above the occupied ones');ap.add_argument('--output',type=Path);a=ap.parse_args()
    f=Path(a.cell) if a.cell.endswith('.npz') else ROOT/'data/bands'/f'{a.cell}.npz';z=np.load(f)
    nat=len(z['fractional_positions']);no=2*nat
    s=Structure(np.asarray(z['lattice']),['Si']*nat,np.asarray(z['fractional_positions']),{'n_occupied':no})
    model=load_gated_si_epm(a.checkpoint)[0].eval();model.subset_ng_min=100000
    E=predict_path(model,s,z['kpoints'],no+a.bands);E=E-E[:,no-1].max()
    gap=float(E[:,no].min()-E[:,no-1].max());print(f'{f.stem}: {nat} atoms, {len(E)} k-points, EpNet-Si gap {gap:.3f} eV')
    if 'energies' in z.files:
        T=np.asarray(z['energies'])[:,:E.shape[1]];T=T-T[:,no-1].max();tg=float(T[:,no].min()-T[:,no-1].max());mid=0.5*tg
        w=np.abs(T-mid)<=2.0;print(f'  PBE gap {tg:.3f} eV | gap error {1e3*abs(gap-max(tg,0)):.1f} meV | band MAE (+-2 eV of midgap) {1e3*np.abs(E-T)[w].mean():.1f} meV')
    if a.output:np.savez(a.output,kpoints=z['kpoints'],energies=E);print('wrote',a.output)

if __name__=='__main__':main()
