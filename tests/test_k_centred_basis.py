"""k-centred plane-wave basis {|k+G| <= G_max}: bands are periodic in k and identical in a supercell."""
import numpy as np
import torch

from epnet.data.structures import Structure
from epnet.models.si_epm import build_gated_si_epm, pack_structure

A = 5.431


def _model(k_centred: bool, seed: int = 0, smooth: bool = False, buffer: float = 0.0):
    torch.manual_seed(seed)
    model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", cutoff=4.0, sp_nonlocal=True,
                               k_centred_basis=k_centred, smooth_cutoff=smooth, smooth_buffer=buffer).double().eval()
    with torch.no_grad():  # nonzero readouts and s/p strengths, so every term of H(k) is exercised
        for name, p in model.named_parameters():
            if "readout" in name or "sp_projector" in name:
                p.add_(0.05 * torch.randn_like(p))
    model.subset_ng_min = 100000
    return model


def _diamond(P=None):
    prim = np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]]) * A
    frac = np.array([[0, 0, 0], [.25, .25, .25]]) + 0.01 * np.array([[0, 0, 0], [1, -2, 3]])  # break T_d
    if P is None:
        return Structure(prim, ["Si"] * 2, frac, {"n_occupied": 4})
    P = np.asarray(P)
    r = range(-3, 4)
    cart = (frac[None] + np.array([[i, j, k] for i in r for j in r for k in r])[:, None]).reshape(-1, 3) @ prim
    f = cart @ np.linalg.inv(P @ prim)
    f = np.unique(np.round(f[np.all((f > -1e-9) & (f < 1 - 1e-9), axis=1)], 9) % 1.0, axis=0)
    n = int(round(abs(np.linalg.det(P))))
    return Structure(P @ prim, ["Si"] * len(f), f, {"n_occupied": 4 * n})


def _bands(model, s, kcart, nb):
    p = pack_structure(s, dtype=torch.float64)
    kf = torch.as_tensor(kcart @ np.asarray(s.lattice).T / (2 * np.pi), dtype=torch.float64)
    with torch.no_grad():
        return model.band_path(p, kf, n_bands=nb).numpy()


import pytest


@pytest.mark.parametrize("smooth,buffer", [(False, 0.0), (True, 0.0), (True, 0.05)])
def test_bands_are_periodic_in_k(smooth, buffer):
    model = _model(True, smooth=smooth, buffer=buffer)
    model.fold_k = False  # periodicity must come from the basis itself
    s = _diamond()
    B = 2 * np.pi * np.linalg.inv(np.asarray(s.lattice)).T
    k = np.array([[0.11, 0.23, -0.07], [0.4, 0.1, 0.3]]) @ B
    e0 = _bands(model, s, k, 8)
    e1 = _bands(model, s, k + np.array([1, -1, 2]) @ B, 8)
    np.testing.assert_allclose(e0, e1, atol=1e-8)


@pytest.mark.parametrize("smooth,buffer", [(False, 0.0), (True, 0.0), (True, 0.05)])
def test_supercell_reproduces_folded_primitive_bands(smooth, buffer):
    model = _model(True, smooth=smooth, buffer=buffer)
    prim, sc = _diamond(), _diamond(np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]))  # conventional cube
    Bp = 2 * np.pi * np.linalg.inv(np.asarray(prim.lattice)).T
    Bs = 2 * np.pi * np.linalg.inv(np.asarray(sc.lattice)).T
    imgs = {}
    for n in [(i, j, l) for i in range(-2, 3) for j in range(-2, 3) for l in range(-2, 3)]:
        g = np.array(n) @ Bs
        imgs.setdefault(tuple(np.round((g @ np.linalg.inv(Bp)) % 1.0, 6) % 1.0), g)
    G = np.array(list(imgs.values()))
    assert len(G) == 4
    k = np.array([[0.0, 0.0, 0.0], [0.37, 0.11, -0.2]]) * 2 * np.pi / A
    e_sc = _bands(model, sc, k, 20)
    e_p = np.stack([_bands(model, prim, k + g, 10) for g in G], axis=1).reshape(len(k), -1)
    np.testing.assert_allclose(e_sc, np.sort(e_p, axis=1)[:, :20], atol=1e-7)


def test_default_is_off_and_gradients_flow():
    assert not build_gated_si_epm().k_centred_basis
    model = _model(True)
    s = _diamond()
    p = pack_structure(s, dtype=torch.float64)
    e = model.band_path(p, torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]], dtype=torch.float64), n_bands=6)
    e.sum().backward()
    grads = [q.grad for q in model.parameters() if q.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_residual_envelope_band_limits_the_form_factor():
    s = _diamond()
    p = pack_structure(s, dtype=torch.float64)
    g = torch.tensor([[1.0, 0.0, 0.0], [0.0, 5.0, 3.0], [8.5, 0.0, 0.0], [0.0, 0.0, 12.0]], dtype=torch.float64)
    for env in (False, True):
        torch.manual_seed(0)
        model = build_gated_si_epm(channels=4, ecut_ry=4.0, monopole_mode="free", cutoff=4.0,
                                   residual_envelope=env).double().eval()
        with torch.no_grad():
            for name, q in model.named_parameters():
                if "readout" in name:
                    q.add_(0.1 * torch.randn_like(q))
            v = model.residual_form_factor(p, g)
        assert model.residual_envelope == env
        assert torch.all(v[:2].abs() > 0)
        beyond = v[2:].abs()  # |G| = 8.5 and 12 Å^-1, past the 8 Å^-1 envelope
        assert torch.all(beyond == 0) if env else torch.all(beyond > 0)


def test_smooth_cutoff_makes_bands_continuous_in_k():
    """Hard |k+G| cutoff: E(k) jumps where plane waves enter the basis. Smooth cutoff: no jumps (2nd differences ~0)."""
    s = _diamond()
    B = 2 * np.pi * np.linalg.inv(np.asarray(s.lattice)).T
    t = np.linspace(0.0, 0.3, 301)
    k = np.array([0.2, 0.1, 0.05]) @ B + t[:, None] * np.array([1.0, 0.3, 0.0])
    jumps = {}
    for key, (smooth, buffer) in {"hard": (False, 0.0), "inside": (True, 0.0), "buffer": (True, 0.05)}.items():
        e = _bands(_model(True, smooth=smooth, buffer=buffer), s, k, 6)
        jumps[key] = np.abs(np.diff(e, 2, axis=0)).max()
    # remaining second differences are kinks where sorted bands cross, not jumps
    assert jumps["hard"] > 1e-2, jumps
    for key in ("inside", "buffer"):
        assert jumps[key] < 1e-3 and jumps[key] < 0.02 * jumps["hard"], jumps
