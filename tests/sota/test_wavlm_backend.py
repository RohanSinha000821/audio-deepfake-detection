import inspect
from pathlib import Path

import pytest
import torch
import yaml

from audio_deepfake_detection.sota.wavlm_backend.model import (
    WavLMWA,
    WavLMWABackend,
    masked_temporal_mean,
    normalize_waveforms,
    select_wa_representations,
    spoof_score,
    weighted_layer_aggregation,
)
from audio_deepfake_detection.sota.wavlm_backend.training import (
    EarlyStoppingState,
    create_weighted_cross_entropy,
    save_checkpoint,
)


def test_wa_backend_has_paper_parameter_count():
    backend = WavLMWABackend()

    assert sum(
        parameter.numel()
        for parameter in backend.parameters()
        if parameter.requires_grad
    ) == 1551
    assert backend.layer_weights.numel() == 13


def test_wa_representations_use_projected_frontend_and_layer_outputs():
    projected = torch.randn(1, 2, 768, requires_grad=True)
    encoder_states = tuple(
        torch.randn(1, 2, 768, requires_grad=True)
        for _ in range(13)
    )

    representations = select_wa_representations(
        projected,
        encoder_states,
    )

    assert len(representations) == 13
    assert representations[0] is projected
    assert representations[1] is encoder_states[1]
    assert representations[12] is encoder_states[12]

    sum(item.sum() for item in representations).backward()
    assert projected.grad is not None
    assert encoder_states[1].grad is not None
    assert encoder_states[12].grad is not None


def test_reproduction_defaults_disable_normalization_and_specaugment():
    signature = inspect.signature(WavLMWA.__init__)

    assert signature.parameters["input_normalization"].default is False
    assert signature.parameters["apply_spec_augment"].default is False

    class DummyConfig:
        hidden_size = 768
        num_hidden_layers = 12
        add_adapter = False
        apply_spec_augment = True

    class DummyWavLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = DummyConfig()

    model = WavLMWA(DummyWavLM())

    assert model.input_normalization is False
    assert model.wavlm.config.apply_spec_augment is False

    config_path = (
        Path(__file__).parents[2] / "configs" / "sota" / "wavlm_wa.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["input_normalization"] is False
    assert config["apply_spec_augment"] is False


def test_weighted_layer_aggregation_shape_and_values():
    hidden_states = (
        torch.ones(2, 3, 4),
        torch.full((2, 3, 4), 3.0),
    )
    weights = torch.tensor([0.25, 0.75])

    result = weighted_layer_aggregation(
        hidden_states,
        weights,
        normalization="none",
    )

    assert result.shape == (2, 3, 4)
    assert torch.allclose(
        result,
        torch.full((2, 3, 4), 2.5),
    )


def test_masked_temporal_mean_ignores_padding():
    features = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [100.0, 200.0],
            ]
        ]
    )
    mask = torch.tensor([[True, True, False]])

    result = masked_temporal_mean(features, mask)

    assert torch.equal(result, torch.tensor([[2.0, 3.0]]))


def test_waveform_normalization_ignores_and_zeros_padding():
    waveform = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    attention_mask = torch.tensor([[True, True, True, False]])

    normalized = normalize_waveforms(waveform, attention_mask)

    assert normalized[0, :3].mean().item() == pytest.approx(0.0)
    assert normalized[0, :3].var(unbiased=False).item() == pytest.approx(1.0)
    assert normalized[0, 3].item() == 0.0


def test_spoof_score_increases_with_spoof_logit():
    lower = spoof_score(torch.tensor([[0.5, 0.6]]))
    higher = spoof_score(torch.tensor([[0.5, 1.6]]))

    assert higher.item() > lower.item()


def test_checkpoint_metadata_distinguishes_best_and_last(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.95,
    )
    early_stopping = EarlyStoppingState()
    config = {"seed": 1234}

    for role in ("best", "last"):
        save_checkpoint(
            tmp_path / f"{role}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            early_stopping=early_stopping,
            config=config,
            checkpoint_role=role,
        )

    best = torch.load(tmp_path / "best.pt", weights_only=False)
    last = torch.load(tmp_path / "last.pt", weights_only=False)

    assert best["checkpoint_role"] == "best"
    assert last["checkpoint_role"] == "last"
    assert best["epoch"] == last["epoch"] == 3
    assert best["config"]["seed"] == last["config"]["seed"] == 1234


def test_weighted_cross_entropy_class_mapping():
    loss = create_weighted_cross_entropy()

    assert torch.equal(
        loss.weight,
        torch.tensor([9.0, 1.0]),
    )
