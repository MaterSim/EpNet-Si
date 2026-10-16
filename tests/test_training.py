import torch

from epnet.train import ComplexLinear, TrainConfig, potential_loss, train_model


def test_potential_loss_excludes_g0_and_balances_samples():
    g = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]],
                      [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]])
    target = torch.zeros(2, 3, dtype=torch.complex64)
    prediction = torch.tensor([[1000.0, 1.0, 1.0], [9000.0, 2.0, 2.0]]).to(torch.complex64)
    loss = potential_loss(prediction, target, g, kind="mae")
    torch.testing.assert_close(loss, loss.new_tensor(1.5))


def test_cell_atom_normalization_is_intensive():
    g = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
    pred = torch.tensor([[0.0, 2.0], [0.0, 4.0]])
    target = torch.zeros_like(pred)
    # Raw extensive values differ, but volume/n_atoms converts both to 10.
    loss = potential_loss(pred, target, g, n_atoms=torch.tensor([2.0, 8.0]),
                          volumes=torch.tensor([10.0, 20.0]), kind="mae")
    torch.testing.assert_close(loss, torch.tensor(10.0))


def test_training_and_checkpoint_resume(tmp_path):
    x = torch.linspace(-1, 1, 20).reshape(1, 20, 1)
    g = torch.ones(1, 20, 3)
    batch = {"inputs": x, "targets": torch.complex(2 * x[..., 0], -x[..., 0]), "g_vectors": g}
    checkpoint = tmp_path / "model.pt"
    model = ComplexLinear(1)
    history = train_model(model, [batch], TrainConfig(epochs=80, learning_rate=0.05, seed=3),
                          checkpoint_path=checkpoint)
    assert checkpoint.exists()
    assert history["train_loss"][-1] < history["train_loss"][0] * 0.02
    resumed = ComplexLinear(1)
    resumed_history = train_model(
        resumed, [batch], TrainConfig(epochs=81, learning_rate=0.05, seed=3),
        checkpoint_path=checkpoint,
    )
    assert len(resumed_history["train_loss"]) == 81
