from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torch import Tensor

from src.protify.base_models import get_base_models
from src.protify.base_models import vec2vec


def _tiny_config(**overrides: object) -> vec2vec.Vec2VecConfig:
    values: dict[str, object] = {
        "encoder_names": ["ESM2-8", "ESM2-35"],
        "encoder_paths": ["Synthyra/ESM2-8M", "Synthyra/ESM2-35M"],
        "encoder_dims": [8, 10],
        "d_adapter": 8,
        "d_hidden": 8,
        "d_transform": 8,
        "adapter_depth": 1,
        "transform_depth": 1,
        "disc_dim": 8,
        "disc_depth": 1,
        "norm_style": "layer",
    }
    values.update(overrides)
    return vec2vec.Vec2VecConfig(**values)


class _TensorSourceEncoder(nn.Module):
    """FastPLMs-style source encoder with no tokenizer attribute."""

    def __init__(self) -> None:
        super().__init__()
        self.last_kwargs: dict[str, object] | None = None

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # input_ids: (b, l); attention_mask: (b, l) or None
        self.last_kwargs = kwargs
        hidden_states = (  # (b, l, d=4)
            input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 4)
        )
        if kwargs.get("output_attentions"):
            return hidden_states, torch.ones(1)  # (b, l, d), (1,)
        return hidden_states  # (b, l, d)


def test_build_uses_protify_source_factory_and_forwards_full_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config()
    translator = vec2vec.Vec2VecModel(config)
    source_model = _TensorSourceEncoder()
    source_tokenizer = object()
    model_load_call: dict[str, object] = {}

    def fake_get_base_model(
        name: str,
        masked_lm: bool = False,
        dtype: torch.dtype | None = None,
        model_path: str | None = None,
    ) -> tuple[nn.Module, object]:
        model_load_call.update(name=name, dtype=dtype, model_path=model_path)
        return source_model, source_tokenizer

    monkeypatch.setattr(
        vec2vec,
        "_load_vec2vec_config_for_inference",
        lambda path: (config, False),
    )
    monkeypatch.setattr(get_base_models, "get_base_model", fake_get_base_model)
    monkeypatch.setattr(
        vec2vec.Vec2VecModel,
        "from_pretrained",
        classmethod(lambda cls, path, config: translator),
    )

    model, tokenizer = vec2vec.build_vec2vec_model(
        "unused",
        dtype=torch.bfloat16,
        model_path="checkpoint",
    )
    output = model(  # (b=2, d_b=10)
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        output_attentions=True,
        sequence_ids=torch.tensor([[0, 0, 0], [1, 1, 1]]),
    )

    assert tokenizer is source_tokenizer
    assert model_load_call == {
        "name": "ESM2-8",
        "dtype": torch.bfloat16,
        "model_path": "Synthyra/ESM2-8M",
    }
    assert source_model.last_kwargs is not None
    assert source_model.last_kwargs["output_attentions"] is True
    assert "sequence_ids" in source_model.last_kwargs
    assert output.shape == (2, 10)
    assert output.dtype == next(translator.parameters()).dtype


def test_source_encoder_output_must_be_residue_embeddings() -> None:
    class PooledSource(nn.Module):
        def forward(self, input_ids: Tensor, **kwargs: object) -> Tensor:
            return torch.ones(input_ids.shape[0], 4)  # (b, d=4)

    config = _tiny_config()
    model = vec2vec.Vec2VecForEmbedding(
        config,
        PooledSource(),
        vec2vec.Vec2VecModel(config),
        "ESM2-8",
        "ESM2-35",
    )

    with pytest.raises(ValueError, match="residue embeddings"):
        model(input_ids=torch.ones(2, 3, dtype=torch.long))
