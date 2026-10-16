#!/usr/bin/env python3
"""Primitive vs supercell consistency of EpNet-Si: the same crystal described by a larger cell must give the same
bands (folded) and the same gap.

Cells: diamond primitive (2 atoms), conventional cubic (8), 2x2x2 supercell (16); clathrate I primitive (46),
2x1x1 supercell (92). For a set of k-points (cartesian, units of 2*pi/a), the supercell spectrum at k must equal the
union of the primitive spectra at k + G_sc over the det(P) supercell reciprocal vectors G_sc that are distinct modulo
the primitive reciprocal lattice ("unfolding test"). We report, per supercell:
  * the gap min_k CBM - max_k VBM over the k set (primitive: over all images), and its difference;
  * the largest eigenvalue difference among states within 3 eV of the gap (VBM-aligned);
  * the largest difference of the encoder invariants h_a and router weights between equivalent atoms.
With the k-centred basis {|k+G| <= G_max} of EpNet-Si the supercell basis is the union of the primitive bases, so
the two agree to float32 precision at the model's 8 Ry (stored: results/derived_physics/supercell_check/).
Usage: check_supercell_consistency.py [--ecut 8 12 ...] [--only diamond|clathrate] [--k-centred] [--ckpt PATH]
Output: results/supercell_check/<system>_<ecut>Ry.json and a summary print.
"""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from epnet.data.manifest import NPZDataset
from epnet.data.structures import Structure
from epnet.models.si_epm import load_gated_si_epm,pack_structure
CKPT=ROOT/'models/epnet_si.pt';OUT=ROOT/'results/supercell_check'
MAN=ROOT/'data/manifest.json'

def supercell(s,P):
    """Supercell with lattice P @ A (P integer 3x3)."""
    A=np.asarray(s.lattice);P=np.asarray(P);As=P@A;n=int(round(abs(np.linalg.det(P))))
    r=range(-4,5);T=np.array([[i,j,k] for i in r for j in r for k in r])
    cart=(np.asarray(s.fractional_positions)[None,:,:]+T[:,None,:]).reshape(-1,3)@A
    f=cart@np.linalg.inv(As);f=f[np.all((f>-1e-6)&(f<1-1e-6),axis=1)]
    f=np.unique(np.round(f%1.0,7)%1.0,axis=0);assert len(f)==n*s.n_atoms,(len(f),n*s.n_atoms)
    md=dict(s.metadata);md['n_occupied']=int(md['n_occupied'])*n
    return Structure(As,['Si']*len(f),f,md)

def recip(A):return 2*np.pi*np.linalg.inv(np.asarray(A)).T

def images(Ap,As):
    """Supercell reciprocal vectors distinct modulo the primitive reciprocal lattice (cartesian)."""
    Bp,Bs=recip(Ap),recip(As);r=range(-4,5);out={}
    for n in [(i,j,k) for i in r for j in r for k in r]:
        g=np.array(n)@Bs;key=tuple(np.round((g@np.linalg.inv(Bp))%1.0,6)%1.0)
        if key not in out or np.linalg.norm(g)<np.linalg.norm(out[key]):out[key]=g
    return np.array(list(out.values()))

def bands(model,s,kcart,nb):
    p=pack_structure(s);B=recip(s.lattice);kf=torch.as_tensor(kcart@np.linalg.inv(B),dtype=p.lattice.dtype);out=[]
    with torch.no_grad():
        for i in range(len(kf)):
            kk=kf[i:i+1];out.append(model.band_path(p,kk,n_bands=nb,cache=model.make_pw_cache(p,kk),refresh_sphere=True))
    return torch.cat(out).double().numpy()

def atom_features(model,s):
    p=pack_structure(s)
    with torch.no_grad():f=model.encode(p);w=model.expert_weights(p)
    return f['scalar'].double().numpy(),w.double().numpy()

def compare(model,prim,sc,klist,label):
    no_p=int(prim.metadata['n_occupied']);no_s=int(sc.metadata['n_occupied']);ndet=no_s//no_p
    G=images(prim.lattice,sc.lattice);assert len(G)==ndet,(len(G),ndet)
    nb_p=no_p+8;nwin=no_s+4;nb_s=nwin+4
    t=time.time();Es=bands(model,sc,klist,nb_s);ts=time.time()-t
    kimg=(klist[:,None,:]+G[None]).reshape(-1,3);t=time.time();Ep=bands(model,prim,kimg,nb_p).reshape(len(klist),ndet,nb_p);tp=time.time()-t
    Eu=np.sort(Ep.reshape(len(klist),-1),axis=1)[:,:nwin]  # unfolded union
    gap=lambda E,no:float(E[:,no].min()-E[:,no-1].max())
    g_s,g_u=gap(Es[:,:nwin],no_s),gap(Eu,no_s);vbm=Eu[:,no_s-1].max()
    sel=np.abs(Eu-vbm-0.5*g_u)<3.0+0.5*g_u;d=np.abs((Es[:,:nwin]-Es[:,no_s-1].max())-(Eu-vbm))
    hp,wp=atom_features(model,prim);hs,ws=atom_features(model,sc)
    # every supercell atom is a copy of a primitive atom: compare against its best-matching primitive atom
    dh=float(np.max([np.min(np.abs(hp-h).max(1)) for h in hs]));dw=float(np.max([np.min(np.abs(wp-w).max(1)) for w in ws]))
    r=dict(label=label,n_atoms=[prim.n_atoms,sc.n_atoms],n_images=ndet,n_k=len(klist),gap_primitive_eV=g_u,gap_supercell_eV=g_s,
           gap_diff_meV=1e3*(g_s-g_u),max_state_diff_meV_within_3eV=1e3*float(d[sel].max()),mean_state_diff_meV_within_3eV=1e3*float(d[sel].mean()),
           max_diff_h=dh,max_diff_router=dw,time_s=[tp,ts],
           per_k=[dict(k=list(map(float,k)),vbm_diff_meV=1e3*float(Es[i,no_s-1]-Eu[i,no_s-1]),cbm_diff_meV=1e3*float(Es[i,no_s]-Eu[i,no_s])) for i,k in enumerate(klist)])
    print(f"{label:34s} gap prim {g_u:.4f}  sc {g_s:.4f}  diff {1e3*(g_s-g_u):+7.2f} meV | states max {r['max_state_diff_meV_within_3eV']:6.2f} mean {r['mean_state_diff_meV_within_3eV']:5.2f} meV | dh {dh:.1e} dpi {dw:.1e}",flush=True)
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ecut',type=float,nargs='+',default=[8.0]);ap.add_argument('--only',default=None);ap.add_argument('--k-centred',action='store_true',help='use the k-centred basis {|k+G|<=G_max}');ap.add_argument('--ckpt',default=None);a=ap.parse_args()
    torch.set_num_threads(max(1,len(os.sched_getaffinity(0))))
    ck=Path(a.ckpt) if a.ckpt else CKPT;model=load_gated_si_epm(ck)[0].eval();model.subset_ng_min=100000;ecut0=model.ecut_ry
    if a.k_centred:model.k_centred_basis=True
    tag=('_kc' if model.k_centred_basis else '')+(f'_{ck.parent.parent.name}' if a.ckpt else '')  # e.g. _kc_general_v2h
    ds=NPZDataset(MAN,allow_external=True);idx={e['id']:i for i,e in enumerate(ds.entries)}
    dia=ds[idx['si-diamond-parent']].structure;cla=ds[idx['si-clathrate_i-parent']].structure
    a_d=np.sqrt(2)*np.linalg.norm(np.asarray(dia.lattice)[0]);Pconv=np.round(a_d*np.linalg.inv(np.asarray(dia.lattice))).astype(int)
    assert np.allclose(Pconv@np.asarray(dia.lattice),a_d*np.eye(3),atol=1e-4),Pconv
    a_c=float(np.linalg.norm(np.asarray(cla.lattice)[0]))
    rng=np.random.default_rng(0)
    kd=2*np.pi/a_d*np.array([[0,0,0],[1,0,0],[.5,.5,.5],[.85,0,0],[1,.5,0],[.75,.75,0],*rng.uniform(-1,1,(2,3))])
    kc=2*np.pi/a_c*np.array([[0,0,0],[.5,0,0],[.5,.5,0],[.5,.5,.5],[.25,0,0],*rng.uniform(-.5,.5,(1,3))])
    OUT.mkdir(parents=True,exist_ok=True)
    print('model',ck,'trained at',ecut0,'Ry; fold_k',model.fold_k,'k-centred basis',model.k_centred_basis,flush=True)
    for ec in a.ecut:
        model.ecut_ry=float(ec);print(f'--- E_cut = {ec:g} Ry',flush=True)
        if a.only in (None,'diamond'):
            res=[compare(model,dia,supercell(dia,Pconv),kd,f'diamond 2 -> 8 (conventional)'),
                 compare(model,dia,supercell(dia,2*np.eye(3,dtype=int)),kd,'diamond 2 -> 16 (2x2x2)')]
            (OUT/f'diamond_{ec:g}Ry{tag}.json').write_text(json.dumps(res,indent=1)+'\n')
        if a.only in (None,'clathrate'):
            res=[compare(model,cla,supercell(cla,np.diag([2,1,1])),kc,'clathrate I 46 -> 92 (2x1x1)')]
            (OUT/f'clathrate_{ec:g}Ry{tag}.json').write_text(json.dumps(res,indent=1)+'\n')
    model.ecut_ry=ecut0

if __name__=='__main__':main()
