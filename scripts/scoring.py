#!/usr/bin/env python3
"""Shared scoring for EpNet-Si against the PBE reference (as used for the paper's test groups).

Gapped cells: band energies on the reference k path, VBM-aligned window MAE and gap error (train_si_pbe_dataset._metrics).
Metals: uniform-grid bands, E_F from the PBE electron count, model shifted by its median offset within +-1 eV; band MAE
within +-1 eV of E_F, relative L1 DOS difference (0.1 eV broadening) and misplaced-electron fraction (metal_metrics).
Grades: gapped pass if band <= 100 meV and gap <= 50 meV, near miss if 50 < gap <= 100 meV, otherwise fail; metals
pass if band <= 50 meV, DOS <= 5% and misplaced <= 0.5%, fail if band > 100 meV or DOS > 10%, otherwise near miss.
"""
import numpy as np,torch
from scipy.special import erfc
from epnet.models.si_epm import pack_structure
_trapz=getattr(np,"trapezoid",None) or np.trapz  # np.trapz was removed in NumPy 2.4

def occupations(e,ef,sigma):return 0.5*erfc((e-ef)/(np.sqrt(2)*sigma))  # Gaussian smearing, per spin
def fermi_level(e,nelec,sigma=0.05):
    lo,hi=e.min()-1,e.max()+1;w=2.0/e.shape[0]  # uniform k weights, spin-degenerate
    for _ in range(100):
        mid=0.5*(lo+hi);n=w*occupations(e,mid,sigma).sum()
        lo,hi=(mid,hi) if n<nelec else (lo,mid)
    return 0.5*(lo+hi)
def dos(e,grid,width=0.1):
    x=(grid[:,None]-e.reshape(1,-1))/width
    return 2.0/e.shape[0]*np.exp(-0.5*x*x).sum(1)/(np.sqrt(2*np.pi)*width)  # states / eV / cell

def metal_metrics(pred,ref,nelec,*,window=1.0,sigma=0.05):
    """Near-E_F quality and E_F self-consistency, reported separately.

    Near-E_F metrics use the PBE E_F (electron count) and shift the model by the median offset in the
    +-window around it: bands, DOS and Fermi sea near E_F. The model's own counted E_F also depends on
    deep occupied bands the training window never sees; ef_selfconsistency_eV = (model-counted E_F)
    minus (window-aligned E_F) measures that, and bandwidth_err_eV compares the occupied bandwidth.
    """
    efr,efp_count=fermi_level(ref,nelec,sigma),fermi_level(pred,nelec,sigma)
    r=ref-efr;near=np.abs(r)<=window
    efp=float(np.median((pred-ref)[near]))+efr   # E_F of the model after the best near-E_F shift
    p=pred-efp
    sel=np.abs(r)<=window
    grid=np.linspace(-window,window,401);gr,gp=dos(r,grid),dos(p,grid)
    fr,fp=occupations(r,0.0,sigma),occupations(p,0.0,sigma)
    return dict(ef_ref_eV=efr,ef_pred_aligned_eV=efp,ef_pred_count_eV=efp_count,ef_selfconsistency_eV=float(efp_count-efp),
                bandwidth_err_eV=float((efp_count-pred.min())-(efr-ref.min())),
                ef_band_mae_eV=float(np.abs(p-r)[sel].mean()),
                dos_l1_rel=float(_trapz(np.abs(gp-gr),grid)/_trapz(gr,grid)),
                n_ef_ref=float(dos(r,np.array([0.0]))[0]),n_ef_pred=float(dos(p,np.array([0.0]))[0]),
                n_ef_rel_err=float(abs(dos(p,np.array([0.0]))[0]/dos(r,np.array([0.0]))[0]-1)),
                misplaced_electron_frac=float((2.0/ref.shape[0])*np.abs(fp-fr).sum()/2/nelec),
                n_points_window=int(sel.sum()))

def fold_to_first_zone(k_frac,lattice):
    """Shortest equivalent k+G of each fractional k. The model's plane-wave set is a fixed |G| sphere (not
    |k+G| as in QE), so k far outside the first Brillouin zone would see an off-centre basis."""
    recip=2*np.pi*np.linalg.inv(np.asarray(lattice,float)).T
    k=np.asarray(k_frac,float);k=k-np.round(k)
    shifts=np.array([[i,j,l] for i in(-1,0,1) for j in(-1,0,1) for l in(-1,0,1)],float)
    cand=k[:,None,:]+shifts[None];norm=np.linalg.norm(cand@recip,axis=-1)
    return cand[np.arange(len(k)),norm.argmin(1)]

def predict(model,structure,k,nb,dev,chunk=256):
    k=fold_to_first_zone(k,structure.lattice)
    packed=pack_structure(structure)
    for n in('positions','lattice','fractional_positions','species','volume'):setattr(packed,n,getattr(packed,n).to(dev))
    k=torch.as_tensor(k,dtype=packed.lattice.dtype,device=dev);out=[]
    with torch.no_grad():
        for i in range(0,len(k),chunk):
            kk=k[i:i+chunk];out.append(model.band_path(packed,kk,n_bands=nb,cache=model.make_pw_cache(packed,kk),refresh_sphere=True).cpu())
    return torch.cat(out).numpy()

def predict_path(model,structure,k,nb,chunk=12):
    """EpNet-Si eigenvalues (eV) at fractional k of the structure, lowest nb bands."""
    packed=pack_structure(structure);k=torch.as_tensor(np.asarray(k),dtype=packed.lattice.dtype);out=[]
    with torch.no_grad():
        for i in range(0,len(k),chunk):
            kk=k[i:i+chunk];out.append(model.band_path(packed,kk,n_bands=nb,cache=model.make_pw_cache(packed,kk),refresh_sphere=True))
    return torch.cat(out).numpy()

def grade_gapped(band,gap):
    if band>0.10 or gap>0.10:return 'fail'
    return 'near_miss' if gap>0.05 else 'pass'

def grade_metal(m):
    if m['ef_band_mae_eV']>0.10 or m['dos_l1_rel']>0.10:return 'fail'
    return 'pass' if m['ef_band_mae_eV']<=0.05 and m['dos_l1_rel']<=0.05 and m['misplaced_electron_frac']<=0.005 else 'near_miss'
