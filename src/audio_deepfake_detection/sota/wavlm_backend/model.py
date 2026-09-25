from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import WavLMModel


WAVLM_BASE_HIDDEN_SIZE = 768
WAVLM_BASE_NUM_LAYERS = 12
WAVLM_WA_NUM_REPRESENTATIONS = 13


def weighted_layer_aggregation(
    hidden_states: Sequence[torch.Tensor],
    layer_weights: torch.Tensor,
    *,
    normalization: str,
) -> torch.Tensor:
    """Combine same-shaped WavLM layer representations."""
    if len(hidden_states) != layer_weights.numel():
        raise ValueError(
            "Number of hidden states and layer weights must match"
        )

    if not hidden_states:
        raise ValueError("At least one hidden state is required")

    reference_shape = hidden_states[0].shape

    if any(state.shape != reference_shape for state in hidden_states):
        raise ValueError("All hidden states must have the same shape")

    if normalization == "softmax":
        weights = torch.softmax(layer_weights, dim=0)
    elif normalization == "none":
        weights = layer_weights
    else:
        raise ValueError(
            "layer weight normalization must be 'softmax' or 'none'"
        )

    stacked = torch.stack(tuple(hidden_states), dim=0)
    weight_shape = (weights.shape[0],) + (1,) * (stacked.ndim - 1)

    return (stacked * weights.view(weight_shape)).sum(dim=0)


def select_wa_representations(
    projected_features: torch.Tensor,
    encoder_hidden_states: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Map projected CNN features and encoder states to paper WA inputs.

    Transformers returns the input to layer 1 at hidden-state index 0,
    followed by the outputs of Transformer layers 1 through 12. The
    paper instead requires the projected CNN/front-end representation
    followed by those 12 layer outputs, so index 0 is replaced here.
    """
    if len(encoder_hidden_states) != WAVLM_WA_NUM_REPRESENTATIONS:
        raise ValueError(
            "Expected encoder input plus 12 layer outputs, got "
            f"{len(encoder_hidden_states)} states"
        )

    if (
        projected_features.ndim != 3
        or projected_features.shape[-1] != WAVLM_BASE_HIDDEN_SIZE
    ):
        raise ValueError(
            "Projected frontend features must have shape [B, T, 768]"
        )

    representations = (
        projected_features,
        *encoder_hidden_states[1:],
    )

    if any(
        representation.shape != projected_features.shape
        for representation in representations
    ):
        raise ValueError(
            "All 13 WA representations must have matching shapes"
        )

    return representations


def masked_temporal_mean(
    features: torch.Tensor,
    feature_mask: torch.Tensor,
) -> torch.Tensor:
    """Average valid feature frames while excluding padded frames."""
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, time, hidden]")

    if feature_mask.ndim != 2:
        raise ValueError("feature_mask must have shape [batch, time]")

    if features.shape[:2] != feature_mask.shape:
        raise ValueError("features and feature_mask shapes do not align")

    mask = feature_mask.to(
        device=features.device,
        dtype=features.dtype,
    )
    valid_counts = mask.sum(dim=1)

    if torch.any(valid_counts == 0):
        raise ValueError("Every example must have a valid feature frame")

    return (
        features * mask.unsqueeze(-1)
    ).sum(dim=1) / valid_counts.unsqueeze(-1)


def normalize_waveforms(
    waveform: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    epsilon: float = 1.0e-7,
) -> torch.Tensor:
    """Match Hugging Face's valid-sample waveform normalization."""
    if waveform.ndim != 2 or attention_mask.shape != waveform.shape:
        raise ValueError(
            "waveform and attention_mask must have shape [batch, samples]"
        )

    mask = attention_mask.to(device=waveform.device, dtype=waveform.dtype)
    valid_counts = mask.sum(dim=1, keepdim=True)

    if torch.any(valid_counts == 0):
        raise ValueError("Every waveform must contain a valid sample")

    means = (waveform * mask).sum(dim=1, keepdim=True) / valid_counts
    centered = (waveform - means) * mask
    variances = centered.square().sum(dim=1, keepdim=True) / valid_counts

    return centered * torch.rsqrt(variances + epsilon)


def spoof_score(logits: torch.Tensor) -> torch.Tensor:
    """Return a scalar score where larger values are more spoof-like."""
    if logits.shape[-1] != 2:
        raise ValueError("Expected logits with final dimension 2")

    return logits[..., 1] - logits[..., 0]


class WavLMWABackend(nn.Module):
    """The 1,551-parameter WA pooling and classification back-end."""

    def __init__(
        self,
        *,
        hidden_size: int = WAVLM_BASE_HIDDEN_SIZE,
        num_representations: int = WAVLM_WA_NUM_REPRESENTATIONS,
        layer_weight_normalization: str = "softmax",
    ) -> None:
        super().__init__()

        if layer_weight_normalization not in {"softmax", "none"}:
            raise ValueError(
                "layer_weight_normalization must be 'softmax' or 'none'"
            )

        self.layer_weight_normalization = layer_weight_normalization
        initial_weight = (
            0.0
            if layer_weight_normalization == "softmax"
            else 1.0 / num_representations
        )
        self.layer_weights = nn.Parameter(
            torch.full((num_representations,), initial_weight)
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def normalized_layer_weights(self) -> torch.Tensor:
        if self.layer_weight_normalization == "softmax":
            return torch.softmax(self.layer_weights, dim=0)

        return self.layer_weights

    def forward(
        self,
        hidden_states: Sequence[torch.Tensor],
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        frame_features = weighted_layer_aggregation(
            hidden_states,
            self.layer_weights,
            normalization=self.layer_weight_normalization,
        )
        utterance_features = masked_temporal_mean(
            frame_features,
            feature_mask,
        )

        return self.classifier(utterance_features)


class WavLMWA(nn.Module):
    """WavLM Base with the Stourbe et al. weighted-average back-end.

    Representation 0 is captured from ``feature_projection``: the
    768-D projected CNN/front-end output before SpecAugment and before
    encoder positional convolution, normalization, and dropout.
    Representations 1 through 12 are the actual outputs of Transformer
    layers 1 through 12, obtained as ``outputs.hidden_states[1:]``.
    """

    def __init__(
        self,
        wavlm: WavLMModel,
        *,
        layer_weight_normalization: str = "softmax",
        input_normalization: bool = False,
        apply_spec_augment: bool = False,
    ) -> None:
        super().__init__()

        if wavlm.config.hidden_size != WAVLM_BASE_HIDDEN_SIZE:
            raise ValueError(
                "WA reproduction requires WavLM Base hidden size 768"
            )

        if wavlm.config.num_hidden_layers != WAVLM_BASE_NUM_LAYERS:
            raise ValueError(
                "WA reproduction requires 12 Transformer layers"
            )

        if getattr(wavlm.config, "add_adapter", False):
            raise ValueError("WavLM adapters are not part of this baseline")

        self.wavlm = wavlm
        self.input_normalization = bool(input_normalization)
        self.wavlm.config.apply_spec_augment = bool(apply_spec_augment)
        self.backend = WavLMWABackend(
            layer_weight_normalization=layer_weight_normalization,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "microsoft/wavlm-base",
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        layer_weight_normalization: str = "softmax",
        input_normalization: bool = False,
        apply_spec_augment: bool = False,
    ) -> WavLMWA:
        wavlm = WavLMModel.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=local_files_only,
        )

        return cls(
            wavlm,
            layer_weight_normalization=layer_weight_normalization,
            input_normalization=input_normalization,
            apply_spec_augment=apply_spec_augment,
        )

    def feature_mask(
        self,
        attention_mask: torch.Tensor,
        feature_length: int,
    ) -> torch.Tensor:
        return self.wavlm._get_feature_vector_attention_mask(
            feature_length,
            attention_mask,
            add_adapter=False,
        )

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.input_normalization:
            waveform = normalize_waveforms(waveform, attention_mask)

        projected_features: torch.Tensor | None = None

        def capture_projected_features(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, torch.Tensor],
        ) -> None:
            nonlocal projected_features
            # Clone preserves autograd while isolating the exact projection
            # output from later in-place SpecAugment/padding operations.
            projected_features = output[0].clone()

        hook = self.wavlm.feature_projection.register_forward_hook(
            capture_projected_features
        )

        try:
            outputs = self.wavlm(
                input_values=waveform,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            hook.remove()

        hidden_states = outputs.hidden_states

        if hidden_states is None:
            raise RuntimeError("WavLM did not return hidden states")

        if projected_features is None:
            raise RuntimeError(
                "WavLM feature projection hook captured no output"
            )

        representations = select_wa_representations(
            projected_features,
            hidden_states,
        )
        feature_mask = self.feature_mask(
            attention_mask,
            representations[0].shape[1],
        )

        return self.backend(representations, feature_mask)

    def score(
        self,
        waveform: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return spoof_score(self(waveform, attention_mask))
