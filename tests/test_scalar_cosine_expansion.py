import importlib.util
from pathlib import Path
import torch
from epnet.models.multipole_form_factor import MultipoleFormFactor
from epnet.models.si_epm import build_gated_si_epm,load_gated_si_epm,model_payload_flags
from epnet.models.baselines.radial import CosineRadialEP


def test_zero_expansion_and_nonzero_fourier_symmetries():
    old=MultipoleFormFactor(3)
    new=MultipoleFormFactor(3,scalar_cosine_basis=12)
    new.load_state_dict(old.state_dict(),strict=False)
    pos=torch.tensor([[.2,.3,-.1],[.7,-.2,.5]])
    g=torch.tensor([[0.,0.,0.],[.5,-.8,1.2],[-.5,.8,-1.2],[8.,0.,0.],[9.,0.,0.]])
    features={'scalar':torch.randn(2,3)}
    torch.testing.assert_close(old(pos,g,features),new(pos,g,features),rtol=0,atol=0)
    q=g.norm(dim=-1)
    torch.testing.assert_close(new.cosine_basis(q),CosineRadialEP(1,12,8.).basis(q),rtol=0,atol=0)
    with torch.no_grad():
        new.scalar_cosine.weight.normal_(); new.scalar_cosine.bias.normal_()
    value=new(pos,g,features)
    assert value[0]==0
    torch.testing.assert_close(value[1],value[2].conj())
    rotation,_=torch.linalg.qr(torch.randn(3,3))
    torch.testing.assert_close(new(pos@rotation,g@rotation,features),value,atol=2e-5,rtol=2e-5)
    value.abs().square().sum().backward()
    assert torch.isfinite(new.scalar_cosine.weight.grad).all()
    assert new.scalar_cosine.weight.grad.abs().sum()>0


def test_checkpoint_roundtrip_nonzero_both_branches(tmp_path):
    model=build_gated_si_epm(channels=3,monopole_mode='free',residual_pack_gate=True,
        pack_gate_windows=['cn10<146@2'],residual_anchor=True,residual_scalar_cosine_basis=12)
    for r in [model.readout,model.readout_frozen]:
        assert r.scalar_cosine.weight.requires_grad
        with torch.no_grad():
            r.scalar_cosine.weight.normal_(); r.scalar_cosine.bias.normal_()
    path=tmp_path/'model.pt'
    torch.save(dict(state_dict=model.state_dict(),**model_payload_flags(model)),path)
    restored,_=load_gated_si_epm(path)
    assert restored.residual_scalar_cosine_basis==12
    assert restored.env_radial is None
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
    assert restored.readout_frozen.scalar_cosine.weight.requires_grad
    model.zero_residual_readout()
    assert not torch.count_nonzero(model.readout.scalar_cosine.weight)
