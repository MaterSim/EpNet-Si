#!/usr/bin/env python3
"""Score an EpNet-Si checkpoint on the 423 test structures of the paper, in its two groups.

Group seen_distorted: distortions of the twelve trained structure types; group unseen: structure types with no cell in
training (clathrate II, SACADA / Amsler / OQMD / AFLOW frameworks, new polytypes). Membership is the metadata
'test_group' of data/manifest.json. Gapped cells (PBE path gap > 0.02 eV) are scored on their reference k path,
metals on the uniform grid in data/metal_grid/ (scripts/scoring.py has the metrics and grade definitions).
With --compare, each cell is also compared with the stored v2j result (results/test_groups/v2j/).
Usage: evaluate_test_groups.py [--checkpoint models/epnet_si.pt] [--output results/my_eval.json] [--ids ID ...] [--compare] [--threads 16]
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from epnet.data.manifest import NPZDataset
from epnet.models.si_epm import load_gated_si_epm
from train_si_pbe_dataset import _metrics
from scoring import grade_gapped,grade_metal,metal_metrics,predict,predict_path

def summary(rows):
    for grp in ('seen_distorted','unseen'):
        sel=[r for r in rows if r['test_group']==grp];g=[r for r in sel if r['gapped']];m=[r for r in sel if not r['gapped']]
        gr=lambda v:'/'.join(str(sum(r['grade']==k for r in v)) for k in ('pass','near_miss','fail'))
        print(f"{grp:15s} n={len(sel):3d} | gapped {len(g):3d}: band {1e3*np.mean([r['band_eV'] for r in g]) if g else np.nan:5.1f} gap "
              f"{1e3*np.mean([r['gap_eV'] for r in g]) if g else np.nan:5.1f} meV  pass/near/fail {gr(g)} | metals {len(m):3d}: band "
              f"{1e3*np.mean([r['band_eV'] for r in m]) if m else np.nan:5.1f} meV  {gr(m)}")

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default=str(ROOT/'models/epnet_si.pt'));ap.add_argument('--output',type=Path)
    ap.add_argument('--ids',nargs='*');ap.add_argument('--compare',action='store_true')
    ap.add_argument('--threads',type=int,default=min(16,len(os.sched_getaffinity(0))));a=ap.parse_args()
    torch.set_num_threads(a.threads)
    model=load_gated_si_epm(a.checkpoint)[0].eval();model.subset_ng_min=100000
    ds=NPZDataset(ROOT/'data/manifest.json');stored={}
    if a.compare:
        for f in (ROOT/'results/test_groups/v2j').glob('*.json'):
            for r in json.loads(f.read_text())['cells']:stored[r['id']]=r
    rows=[]
    for i,e in enumerate(ds.entries):
        md=e['metadata']
        if md.get('role')!='test' or (a.ids and e['id'] not in a.ids):continue
        s=ds[i];no=int(md['n_occupied']);tgt=np.asarray(s.bands.energies);pgap=float(tgt[:,no].min()-tgt[:,no-1].max())
        row=dict(id=e['id'],test_group=md['test_group'],test_source=md.get('test_source'),family=md.get('family'),pbe_gap_eV=pgap)
        if pgap>0.02:
            p=predict_path(model,s.structure,s.bands.kpoints,tgt.shape[1])[:,:tgt.shape[1]];m=_metrics(p,tgt,no,True)
            row.update(gapped=True,band_eV=m['window_mae_eV'],gap_eV=m['gap_error_eV'],grade=grade_gapped(m['window_mae_eV'],m['gap_error_eV']))
        else:
            z=np.load(ROOT/'data/metal_grid'/f"{e['id']}.npz");nb=z['energies'].shape[1]
            mm=metal_metrics(predict(model,s.structure,z['kpoints'],nb,'cpu')[:,:nb],z['energies'],float(z['nelec']))
            row.update(gapped=False,band_eV=mm['ef_band_mae_eV'],dos_l1=mm['dos_l1_rel'],grade=grade_metal(mm))
        if a.compare and e['id'] in stored:row['stored_band_eV']=stored[e['id']].get('band_eV');row['stored_gap_eV']=stored[e['id']].get('gap_eV')
        rows.append(row);print(row,flush=True)
    summary(rows)
    if a.compare:
        d=[abs(r['band_eV']-r['stored_band_eV']) for r in rows if r.get('stored_band_eV') is not None]
        print('max |band error - stored v2j| over %d cells: %.2e eV'%(len(d),max(d) if d else float('nan')))
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(checkpoint=str(a.checkpoint),cells=rows),indent=1)+'\n')

if __name__=='__main__':main()
