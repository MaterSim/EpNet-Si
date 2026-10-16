#!/usr/bin/env python3
"""Retrain EpNet-Si with the recipe of the paper (the v2j model in models/epnet_si.pt).

Two phases, each a sequence of learning-rate stages of scripts/train_si_pbe_dataset.py (AdamW, full batches, one step
per epoch, family-balanced loss):
  precursor  66 cells (data/manifest_precursor.json, split diag_fit), three stages, from models/precursor_init.pt
  final      77 cells (data/manifest_v2j.json), the last two stages, warm-started from the precursor's last stage
Stage settings, cell lists and model options come from models/protocol_precursor.json and models/protocol_v2j.json.
Variants of the paper:
  --drop-family F   leave one family out (its training cells get split 'lc_unused'), as in the LOFO test
  --fixed           one fixed Si potential (residual readouts and per-atom s/p head frozen at zero), the baseline
Outputs: <out>/<phase>/stage<i>.{pt,json}; the final model is <out>/final/stage2.pt.
About 13 h (precursor) + 7 h (final) on one NVIDIA L40S; --smoke runs one step on two cells on the CPU.
Usage: train_epnet_si.py [--out runs/v2j] [--phase both|precursor|final] [--drop-family F] [--fixed] [--smoke]
"""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def manifest_for(pr,out,drop):
    """The phase manifest, with the dropped family's training cells moved out of the training split."""
    man=ROOT/pr['manifest']
    if not drop:return man,pr['train_ids']
    m=json.loads(man.read_text());fam={s['id']:s['metadata'].get('family') for s in m['samples']};train=[i for i in pr['train_ids'] if fam[i]!=drop]
    assert len(train)<len(pr['train_ids']),f'no training cell of family {drop}'
    for s in m['samples']:
        if s['id'] in pr['train_ids'] and s['id'] not in train:s['metadata']=dict(s['metadata'],split='lc_unused')
        s['path']=str((man.parent/s['path']).resolve())
    p=out/f'manifest_{Path(pr["manifest"]).stem}_no_{drop}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(m,indent=1));return p,train

def run_phase(name,pr,init,out,a):
    man,train=manifest_for(pr,out,a.drop_family);plan=pr['plans'][0];ck=str(init);d=out/name;d.mkdir(parents=True,exist_ok=True)
    print(f'{name}: {len(train)} training cells, {len(pr["stages"])} stages, init {ck}',flush=True)
    for i,st in enumerate(pr['stages'],1):
        res=d/('smoke.json' if a.smoke else f'stage{i}.json')
        if res.exists() and res.with_suffix('.pt').exists() and not a.smoke:ck=str(res.with_suffix('.pt'));print('  stage',i,'exists, skipped');continue
        cmd=[sys.executable,'-u',str(ROOT/'scripts/train_si_pbe_dataset.py'),'--manifest',str(man),'--model-init',ck,'--splits','diag_fit',
             '--exclude-ids',*pr['exclude_ids'],'--epochs','1' if a.smoke else str(st['epochs']),'--steps-per-epoch',str(pr['steps_per_epoch']),
             '--seed',str(pr['seed']),'--ecut-ry','8','--monopole-mode','free','--sphere-lr',str(st['sphere_lr']),'--lr',str(st['lr']),
             '--compact-lr',str(st['lr']),'--projector-lr',str(st['projector_lr']),'--residual-anchor-l2','0','--env-l2','0',
             '--gap-weight',str(pr['gap_weight']),'--k-stride',str(pr['k_stride']),'--cell-workers','1','--threads-per-worker','8',
             '--device','cpu' if a.smoke else a.device,'--log-every','10','--eval-every','50',
             '--watch-families','diamond','clathrate_i','oqmd_frameworks',*plan['extra_args'],'--output',str(res)]
        if a.fixed:cmd.append('--freeze-residual')
        if a.smoke:cmd+=['--max-samples','2']
        subprocess.run(cmd,cwd=ROOT,check=True)
        if a.smoke:print('SMOKE_OK',name);return None
        got={r['id'] for r in json.loads(res.read_text())['metrics']};assert got==set(train),'trained cells differ from the protocol'
        ck=str(res.with_suffix('.pt'))
    return ck

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'runs/v2j');ap.add_argument('--phase',choices=['both','precursor','final'],default='both')
    ap.add_argument('--drop-family');ap.add_argument('--fixed',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',default='cuda')
    a=ap.parse_args()
    pre=json.loads((ROOT/'models/protocol_precursor.json').read_text());fin=json.loads((ROOT/'models/protocol_v2j.json').read_text())
    ck=ROOT/'models/precursor_init.pt'
    if a.phase in ('both','precursor'):
        ck=run_phase('precursor',pre,ck,a.out,a)
        if ck is None:return
    else:ck=a.out/'precursor'/f'stage{len(pre["stages"])}.pt'
    if a.phase in ('both','final'):
        ck=run_phase('final',fin,ck,a.out,a)
        if ck:print('final model:',ck,'\nevaluate with: python scripts/evaluate_test_groups.py --checkpoint',ck)

if __name__=='__main__':main()
